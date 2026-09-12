from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from app.services.human_research_transfer import (
    SHARED_THERMAL_SHAPE_FEATURES,
    extract_offset_invariant_thermal_shape,
)


def _norm(value: object) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", str(value))
        if not unicodedata.combining(char)
    ).upper()


def infer_label(path: Path) -> tuple[int | None, str | None]:
    parts = [_norm(part) for part in path.parts]
    if any("DOENTES" in part for part in parts):
        return 1, "CANCER"
    if any("SAUDAVEIS" in part for part in parts):
        return 0, "HEALTHY"
    return None, None


def patient_from_name(name: str) -> str:
    match = re.search(r"PAC[_-]?(\d+)", name, flags=re.I)
    if match:
        return match.group(1)
    bits = name.split("_")
    return bits[1] if len(bits) > 1 else Path(name).stem


def matrix_name_for_segmentation(name: str) -> str:
    converted = re.sub(r"-(dir|esq)\.png$", ".txt", name, flags=re.I)
    if converted == name:
        raise ValueError(f"Unsupported segmentation filename: {name}")
    return converted


def build_manifest(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for segdir in root.rglob("Segmentadas"):
        label, label_name = infer_label(segdir)
        if label is None or label_name is None:
            continue
        patient_root = segdir.parent
        matrix_dirs = [patient_root / "Matrizes", patient_root / "Matrizes de Temperatura"]
        for png in sorted(segdir.glob("*.png")):
            try:
                base = matrix_name_for_segmentation(png.name)
            except ValueError:
                continue
            matrix = next((directory / base for directory in matrix_dirs if (directory / base).exists()), None)
            if matrix is None:
                continue
            subject_id = f"DMR_{patient_from_name(png.name)}"
            rows.append(
                {
                    "subject_id": subject_id,
                    "label": label_name,
                    "y": label,
                    "segmentation_path": str(png),
                    "matrix_path": str(matrix),
                    "record": png.name,
                    "side": "RIGHT" if "-dir.png" in png.name.lower() else "LEFT",
                }
            )

    if not rows:
        raise ValueError("No matched DMR-IR segmentation/matrix records were found")
    manifest = pd.DataFrame(rows).drop_duplicates(
        subset=["segmentation_path", "matrix_path"]
    ).reset_index(drop=True)

    labels_per_subject = manifest.groupby("subject_id")["label"].nunique()
    bad = labels_per_subject[labels_per_subject > 1].index.tolist()
    if bad:
        raise ValueError(f"Subjects with inconsistent labels: {bad[:20]}")
    return manifest


def load_supported_values(segmentation_path: str | Path, matrix_path: str | Path) -> np.ndarray:
    support = np.asarray(Image.open(segmentation_path).convert("L")) > 0
    matrix = np.loadtxt(matrix_path, dtype=np.float32)
    if matrix.shape != support.shape:
        raise ValueError(f"shape mismatch {matrix.shape} vs {support.shape}")
    values = matrix[support & np.isfinite(matrix)]
    if values.size < 100:
        raise ValueError("too few valid breast pixels")
    return values.astype(np.float64, copy=False)


def features_from_values(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 100 or not np.isfinite(values).all():
        raise ValueError("expected at least 100 finite thermal values")
    # Reuse the exact target-domain feature implementation by presenting the
    # supported source values as a 2-D field. This prevents source/target drift.
    field = values.reshape(1, -1)
    features = extract_offset_invariant_thermal_shape(field)
    return {name: float(features[name]) for name in SHARED_THERMAL_SHAPE_FEATURES}


def build_feature_table(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    errors: list[dict] = []
    for item in manifest.itertuples(index=False):
        try:
            values = load_supported_values(item.segmentation_path, item.matrix_path)
            features = features_from_values(values)
            rows.append(
                {
                    "subject_id": item.subject_id,
                    "label": item.label,
                    "record": item.record,
                    "side": item.side,
                    **features,
                }
            )
        except Exception as exc:  # preserve row-level provenance; never impute
            errors.append({"record": item.record, "error": str(exc)})

    feature_table = pd.DataFrame(rows)
    if feature_table.empty:
        raise ValueError("No usable DMR-IR thermal records remained after feature extraction")
    if not set(SHARED_THERMAL_SHAPE_FEATURES).issubset(feature_table.columns):
        raise RuntimeError("Shared thermal feature contract is incomplete")
    return feature_table, pd.DataFrame(errors, columns=["record", "error"])


def build(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    manifest = build_manifest(root)
    features, errors = build_feature_table(manifest)
    subject_labels = features.groupby("subject_id")["label"].first()
    label_counts = subject_labels.value_counts().to_dict()
    summary = {
        "dataset": "DMR-IR",
        "source_domain": "human infrared thermography",
        "feature_contract": "offset_invariant_thermal_shape_v0",
        "feature_names": list(SHARED_THERMAL_SHAPE_FEATURES),
        "records_manifested": int(len(manifest)),
        "records_usable": int(len(features)),
        "records_failed": int(len(errors)),
        "subjects_usable": int(features["subject_id"].nunique()),
        "subject_label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "absolute_temperature_used": False,
        "clinical_claim": "NONE",
        "semantics": "auxiliary source-domain research features only; not MumGuard validation",
    }
    return manifest, features, errors, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DMR-IR shared thermal shape features")
    parser.add_argument("root", type=Path, help="Extracted DMR-IR dataset root")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    manifest, features, errors, summary = build(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_dir / "dmr_ir_original_manifest.csv", index=False)
    features.to_csv(args.output_dir / "dmr_ir_shared_shape_features.csv", index=False)
    errors.to_csv(args.output_dir / "dmr_ir_shared_shape_errors.csv", index=False)
    (args.output_dir / "dmr_ir_shared_shape_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
