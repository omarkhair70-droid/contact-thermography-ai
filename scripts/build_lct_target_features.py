from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from app.services.target_feature_builder import (
    FEATURE_CONTRACT_VERSION,
    build_target_feature_vector,
    feature_columns,
    feature_schema,
)


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_embedding_lookup(
    embedding_npy: Path | None,
    embedding_index_csv: Path | None,
) -> dict[str, np.ndarray] | None:
    if embedding_npy is None and embedding_index_csv is None:
        return None
    if embedding_npy is None or embedding_index_csv is None:
        raise ValueError("--embedding-npy and --embedding-index-csv must be supplied together")

    matrix = np.load(embedding_npy, allow_pickle=False)
    index = pd.read_csv(embedding_index_csv, keep_default_na=False)
    if matrix.ndim != 2 or matrix.shape[1] != 384:
        raise ValueError(f"Expected Nx384 embedding matrix, received {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix contains NaN/Inf")
    if "source_image" not in index.columns:
        raise ValueError("Embedding index CSV requires source_image")
    if len(index) != len(matrix):
        raise ValueError("Embedding index row count must match embedding matrix")

    if "embedding_row" in index.columns:
        rows = pd.to_numeric(index["embedding_row"], errors="raise").astype(int).to_numpy()
    else:
        rows = np.arange(len(index), dtype=int)
    if sorted(rows.tolist()) != list(range(len(index))):
        raise ValueError("embedding_row must be a permutation of 0..N-1")

    return {
        str(source_image): matrix[int(row)].astype(np.float32, copy=False)
        for source_image, row in zip(index["source_image"], rows)
    }


def _resolve_image(image_uri: str, image_dir: Path) -> Path:
    value = str(image_uri).strip()
    if value.startswith("client-private://"):
        return image_dir / Path(value.removeprefix("client-private://")).name
    path = Path(value)
    if path.is_absolute():
        return path
    return image_dir / path.name


def build_feature_table(
    manifest_path: Path,
    image_dir: Path,
    *,
    source_id: str,
    embedding_lookup: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    required = {
        "subject_id",
        "image_path",
        "source_id",
        "species",
        "label",
        "label_provenance",
        "tlc_profile_id",
        "device_profile_id",
        "use_role",
        "train_eligible",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Missing manifest columns: {missing}")

    selected = manifest[manifest["source_id"] == source_id].copy()
    if selected.empty:
        raise ValueError(f"No manifest rows found for source_id={source_id}")

    records = []
    columns = feature_columns()
    for row in selected.itertuples(index=False):
        image_path = _resolve_image(row.image_path, image_dir)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing target image for {row.subject_id}: {image_path}")
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not decode image: {image_path}")

        embedding = None
        if embedding_lookup is not None:
            embedding = embedding_lookup.get(image_path.name)
            if embedding is None:
                raise ValueError(f"No DINO embedding mapped for {image_path.name}")

        result = build_target_feature_vector(
            bgr,
            str(row.tlc_profile_id),
            embedding=embedding,
        )
        record = {
            "subject_id": str(row.subject_id),
            "source_image": image_path.name,
            "source_id": str(row.source_id),
            "label": str(row.label),
            "label_provenance": str(row.label_provenance),
            "species": str(row.species),
            "tlc_profile_id": str(row.tlc_profile_id),
            "device_profile_id": str(row.device_profile_id),
            "use_role": str(row.use_role),
            "train_eligible": _as_bool(row.train_eligible),
            "morphology_descriptor": result.morphology_descriptor,
            "qc_status": result.qc_status,
            "qc_flags": ";".join(result.qc_flags),
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "clinical_claim": "NONE",
        }
        record.update(
            {name: float(value) for name, value in zip(columns, result.vector)}
        )
        records.append(record)

    frame = pd.DataFrame(records)
    if frame["subject_id"].duplicated().any():
        dup = frame.loc[frame["subject_id"].duplicated(), "subject_id"].tolist()
        raise ValueError(f"Duplicate feature subject IDs: {dup[:20]}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--source-id", default="client-mice-2026")
    parser.add_argument("--embedding-npy", type=Path)
    parser.add_argument("--embedding-index-csv", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--schema-json", type=Path)
    args = parser.parse_args()

    lookup = _load_embedding_lookup(args.embedding_npy, args.embedding_index_csv)
    frame = build_feature_table(
        args.manifest,
        args.image_dir,
        source_id=args.source_id,
        embedding_lookup=lookup,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)

    schema_path = args.schema_json or args.output_csv.with_suffix(".schema.json")
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(feature_schema(), indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "rows": int(len(frame)),
                "subjects": int(frame["subject_id"].nunique()),
                "feature_contract_version": FEATURE_CONTRACT_VERSION,
                "feature_count": len(feature_columns()),
                "morphology": frame["morphology_descriptor"].value_counts().to_dict(),
                "qc_status": frame["qc_status"].value_counts().to_dict(),
                "clinical_claim": "NONE",
                "output_csv": str(args.output_csv),
                "schema_json": str(schema_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
