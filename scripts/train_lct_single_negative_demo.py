from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

POSITIVE_LABELS = {"CANCER", "MALIGNANT", "TUMOR_BEARING", "TUMOR_LIKE"}
NEGATIVE_LABELS = {"HEALTHY", "BENIGN", "NO_TUMOR_LIKE"}
FEATURE_CONTRACT_VERSION = "lct-target-v1"
TLC_PROFILE_ID = "client-device-tlc-pending"
DINO_DIM = 384
DINO_OFFSET = 33
FULL_FEATURE_COUNT = 417


def _binary_label(value: object) -> int:
    label = str(value).strip().upper()
    if label in POSITIVE_LABELS:
        return 1
    if label in NEGATIVE_LABELS:
        return 0
    raise ValueError(f"unsupported binary label: {value}")


def _load_embedding_index(path: Path, rows: int) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if "source_image" not in frame.columns:
        raise ValueError("embedding index CSV requires source_image")
    if len(frame) != rows:
        raise ValueError("embedding index row count must match embedding matrix")
    if "embedding_row" in frame.columns:
        order = pd.to_numeric(frame["embedding_row"], errors="raise").astype(int)
        if sorted(order.tolist()) != list(range(rows)):
            raise ValueError("embedding_row must be a permutation of 0..N-1")
        frame = frame.assign(_row=order).sort_values("_row").drop(columns="_row")
    return frame.reset_index(drop=True)


def _load_ground_truth(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"subject_id", "image_id", "tumor_status"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"ground-truth CSV missing columns: {missing}")
    if frame["subject_id"].duplicated().any():
        raise ValueError("ground truth contains duplicate subject_id values")
    if frame["image_id"].duplicated().any():
        raise ValueError("ground truth contains duplicate image_id values")
    frame = frame.copy()
    frame["y"] = frame["tumor_status"].map(_binary_label)
    return frame


def train_demo(
    embedding_npy: Path,
    embedding_index_csv: Path,
    ground_truth_csv: Path,
    output_json: Path,
    *,
    acknowledge_single_negative: bool,
    seed: int = 20260911,
) -> dict:
    if not acknowledge_single_negative:
        raise ValueError(
            "single-negative demo training is intentionally locked; pass the explicit "
            "acknowledgement flag only for internal research/demo use"
        )

    matrix = np.load(embedding_npy, allow_pickle=False).astype(np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != DINO_DIM:
        raise ValueError(f"expected Nx{DINO_DIM} DINOv2 embeddings, received {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("embedding matrix contains NaN/Inf")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("embedding matrix contains zero-norm rows")
    matrix = matrix / norms

    index = _load_embedding_index(embedding_index_csv, len(matrix))
    truth = _load_ground_truth(ground_truth_csv)
    joined = index[["source_image"]].copy()
    joined["embedding_row"] = np.arange(len(joined), dtype=int)
    joined = joined.merge(
        truth[["subject_id", "image_id", "tumor_status", "y"]],
        left_on="source_image",
        right_on="image_id",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(truth) or len(joined) != len(matrix):
        missing_truth = sorted(set(index["source_image"]) - set(truth["image_id"]))
        missing_embedding = sorted(set(truth["image_id"]) - set(index["source_image"]))
        raise ValueError(
            "ground truth and embedding index must map one-to-one; "
            f"without_truth={missing_truth[:10]}, without_embedding={missing_embedding[:10]}"
        )

    y = joined["y"].to_numpy(dtype=int)
    positive = int((y == 1).sum())
    negative = int((y == 0).sum())
    if negative != 1:
        raise ValueError(f"this demo lane requires exactly one negative subject; found {negative}")
    if positive < 2:
        raise ValueError(f"this demo lane requires at least two positive subjects; found {positive}")

    X = matrix[joined["embedding_row"].to_numpy(dtype=int)]
    model = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=5000,
        solver="liblinear",
        random_state=seed,
    ).fit(X, y)
    score = model.predict_proba(X)[:, 1]

    records = [
        {
            "id": str(subject_id),
            "source_image": str(source_image),
            "y": int(label),
            "score": round(float(value), 6),
        }
        for subject_id, source_image, label, value in zip(
            joined["subject_id"], joined["source_image"], y, score
        )
    ]
    payload = {
        "name": "lct-native-binary",
        "version": "0.1.0-exp8v1-dino",
        "model_type": "logistic_regression_linear_json",
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "feature_block": "dinov2",
        "feature_offset": DINO_OFFSET,
        "feature_count": DINO_DIM,
        "full_feature_count": FULL_FEATURE_COUNT,
        "dinov2_backbone": "dinov2_vits14",
        "tlc_profile_id": TLC_PROFILE_ID,
        "device_profile_id": "client-device-unknown",
        "species": "mouse",
        "classes": ["NO_TUMOR_LIKE", "TUMOR_LIKE"],
        "decision_threshold": 0.5,
        "intercept": float(model.intercept_[0]),
        "coefficients": [float(v) for v in model.coef_[0]],
        "training": {
            "subjects": int(len(joined)),
            "positive_subjects": positive,
            "negative_subjects": negative,
            "positive_ids": joined.loc[y == 1, "subject_id"].astype(str).tolist(),
            "negative_ids": joined.loc[y == 0, "subject_id"].astype(str).tolist(),
            "label_source": "client class map relayed by project owner on 2026-09-11",
            "class_weight": "balanced",
            "C": 1.0,
            "solver": "liblinear",
            "seed": seed,
            "validation_status": "NOT_VALIDATED_SINGLE_NEGATIVE",
            "resubstitution_scores": records,
        },
        "semantics": (
            "experimental internal research/demo classifier; model score is not "
            "cancer probability, diagnosis, or tumor-size estimate"
        ),
        "clinical_claim": "NONE",
        "clinical_use": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-npy", type=Path, required=True)
    parser.add_argument("--embedding-index-csv", type=Path, required=True)
    parser.add_argument("--ground-truth-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--acknowledge-single-negative", action="store_true")
    args = parser.parse_args()
    payload = train_demo(
        args.embedding_npy,
        args.embedding_index_csv,
        args.ground_truth_csv,
        args.output_json,
        acknowledge_single_negative=args.acknowledge_single_negative,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "artifact": str(args.output_json),
                "version": payload["version"],
                "subjects": payload["training"]["subjects"],
                "positive_subjects": payload["training"]["positive_subjects"],
                "negative_subjects": payload["training"]["negative_subjects"],
                "validation_status": payload["training"]["validation_status"],
                "clinical_claim": "NONE",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
