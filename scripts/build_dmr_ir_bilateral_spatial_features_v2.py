from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from app.services.human_research_transfer_v2 import (
    BILATERAL_SPATIAL_FEATURES,
    FEATURE_CONTRACT,
    extract_dimensionless_bilateral_spatial_features,
    resample_supported_field,
)
from scripts.build_dmr_ir_shared_shape_features import build_manifest


def load_supported_field(segmentation_path: str | Path, matrix_path: str | Path) -> np.ndarray:
    support = np.asarray(Image.open(segmentation_path).convert("L")) > 0
    matrix = np.loadtxt(matrix_path, dtype=np.float32)
    if matrix.shape != support.shape:
        raise ValueError(f"shape mismatch {matrix.shape} vs {support.shape}")
    field = np.full(matrix.shape, np.nan, dtype=np.float64)
    valid = support & np.isfinite(matrix)
    if int(valid.sum()) < 100:
        raise ValueError("too few valid breast pixels")
    field[valid] = matrix[valid]
    return field


def _aggregate_side(records: pd.DataFrame, *, size: int) -> tuple[np.ndarray, int, list[dict]]:
    grids: list[np.ndarray] = []
    errors: list[dict] = []
    for item in records.itertuples(index=False):
        try:
            raw = load_supported_field(item.segmentation_path, item.matrix_path)
            grids.append(resample_supported_field(raw, size=size))
        except Exception as exc:
            errors.append({"record": item.record, "error": str(exc)})
    if not grids:
        raise ValueError("no usable records remained for side")

    stack = np.stack(grids, axis=0)
    with np.errstate(all="ignore"):
        aggregate = np.nanmedian(stack, axis=0)
    if int(np.isfinite(aggregate).sum()) < 64:
        raise ValueError("aggregated side has insufficient support")
    return aggregate, len(grids), errors


def build_subject_table(manifest: pd.DataFrame, *, size: int = 64) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    errors: list[dict] = []

    for subject_id, subject in manifest.groupby("subject_id", sort=True):
        labels = subject["label"].drop_duplicates().tolist()
        if len(labels) != 1:
            errors.append({"subject_id": subject_id, "stage": "label", "error": f"labels={labels}"})
            continue
        left_records = subject[subject["side"] == "LEFT"]
        right_records = subject[subject["side"] == "RIGHT"]
        if left_records.empty or right_records.empty:
            errors.append({
                "subject_id": subject_id,
                "stage": "bilateral",
                "error": f"missing side: left={len(left_records)} right={len(right_records)}",
            })
            continue
        try:
            left, n_left, left_errors = _aggregate_side(left_records, size=size)
            right, n_right, right_errors = _aggregate_side(right_records, size=size)
            for item in left_errors + right_errors:
                errors.append({"subject_id": subject_id, "stage": "record", **item})
            features = extract_dimensionless_bilateral_spatial_features(left, right, size=size)
            rows.append({
                "subject_id": subject_id,
                "label": labels[0],
                "left_records_usable": int(n_left),
                "right_records_usable": int(n_right),
                **features,
            })
        except Exception as exc:
            errors.append({"subject_id": subject_id, "stage": "subject", "error": str(exc)})

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No usable bilateral DMR-IR subjects remained")
    if not set(BILATERAL_SPATIAL_FEATURES).issubset(table.columns):
        raise RuntimeError("bilateral spatial feature contract is incomplete")
    return table, pd.DataFrame(errors)


def build(root: Path, *, size: int = 64) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    manifest = build_manifest(root)
    subjects, errors = build_subject_table(manifest, size=size)
    counts = subjects["label"].value_counts().to_dict()
    summary = {
        "dataset": "DMR-IR",
        "source_domain": "human infrared thermography",
        "feature_contract": FEATURE_CONTRACT,
        "feature_names": list(BILATERAL_SPATIAL_FEATURES),
        "subjects_usable": int(len(subjects)),
        "subject_label_counts": {str(key): int(value) for key, value in counts.items()},
        "manifest_records": int(len(manifest)),
        "errors_logged": int(len(errors)),
        "spatial_geometry_preserved": True,
        "bilateral_features_used": True,
        "offset_invariant": True,
        "positive_scale_invariant": True,
        "absolute_temperature_used": False,
        "clinical_claim": "NONE",
        "semantics": "auxiliary source-domain research features only; not MumGuard validation",
    }
    return manifest, subjects, errors, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DMR-IR bilateral spatial transfer features v0.2")
    parser.add_argument("root", type=Path, help="Extracted DMR-IR dataset root")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()

    manifest, subjects, errors, summary = build(args.root, size=args.size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_dir / "dmr_ir_original_manifest.csv", index=False)
    subjects.to_csv(args.output_dir / "dmr_ir_bilateral_spatial_subject_features_v2.csv", index=False)
    errors.to_csv(args.output_dir / "dmr_ir_bilateral_spatial_errors_v2.csv", index=False)
    (args.output_dir / "dmr_ir_bilateral_spatial_summary_v2.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
