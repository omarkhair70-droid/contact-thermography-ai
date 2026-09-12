from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from app.services.human_research_transfer_v21 import (
    FEATURE_CONTRACT,
    LOCAL_SPATIAL_FEATURES,
    extract_dimensionless_local_spatial_features,
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


def build_record_table(manifest: pd.DataFrame, *, size: int = 64) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    errors: list[dict] = []
    for item in manifest.itertuples(index=False):
        try:
            field = load_supported_field(item.segmentation_path, item.matrix_path)
            features = extract_dimensionless_local_spatial_features(field, size=size)
            rows.append({
                "subject_id": item.subject_id,
                "label": item.label,
                "record": item.record,
                "side": item.side,
                **features,
            })
        except Exception as exc:
            errors.append({
                "subject_id": item.subject_id,
                "record": item.record,
                "side": item.side,
                "error": str(exc),
            })

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No usable DMR-IR records remained")
    if not set(LOCAL_SPATIAL_FEATURES).issubset(table.columns):
        raise RuntimeError("local spatial feature contract is incomplete")

    labels_per_subject = table.groupby("subject_id")["label"].nunique()
    bad = labels_per_subject[labels_per_subject > 1].index.tolist()
    if bad:
        raise ValueError(f"Subjects with inconsistent labels: {bad[:20]}")
    return table, pd.DataFrame(errors)


def build(root: Path, *, size: int = 64) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    manifest = build_manifest(root)
    records, errors = build_record_table(manifest, size=size)
    subject_labels = records.groupby("subject_id")["label"].first()
    side_coverage = (
        records.groupby(["label", "side"])["subject_id"]
        .nunique()
        .rename("subjects")
        .reset_index()
        .to_dict(orient="records")
    )
    counts = subject_labels.value_counts().to_dict()
    summary = {
        "dataset": "DMR-IR",
        "source_domain": "human infrared thermography",
        "feature_contract": FEATURE_CONTRACT,
        "feature_names": list(LOCAL_SPATIAL_FEATURES),
        "records_usable": int(len(records)),
        "records_failed": int(len(errors)),
        "subjects_usable": int(records["subject_id"].nunique()),
        "subject_label_counts": {str(key): int(value) for key, value in counts.items()},
        "side_coverage_by_label": side_coverage,
        "bilateral_availability_used_as_feature": False,
        "bilateral_required_for_source_training": False,
        "spatial_geometry_preserved": True,
        "offset_invariant": True,
        "positive_scale_invariant": True,
        "absolute_temperature_used": False,
        "clinical_claim": "NONE",
        "semantics": (
            "local source-domain spatial morphology only; MumGuard bilateral asymmetry remains a separate target-domain lane"
        ),
    }
    return manifest, records, errors, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DMR-IR local spatial transfer features v0.2.1")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()

    manifest, records, errors, summary = build(args.root, size=args.size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_dir / "dmr_ir_original_manifest.csv", index=False)
    records.to_csv(args.output_dir / "dmr_ir_local_spatial_record_features_v21.csv", index=False)
    errors.to_csv(args.output_dir / "dmr_ir_local_spatial_errors_v21.csv", index=False)
    (args.output_dir / "dmr_ir_local_spatial_summary_v21.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
