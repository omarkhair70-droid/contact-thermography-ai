from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def run(path: Path, dino_dims: int = 384, folds: int = 5, seed: int = 20260910) -> dict:
    frame = pd.read_csv(path)
    required = {"subject_id", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing required columns: {sorted(required - set(frame.columns))}")

    feature_cols = [c for c in frame.columns if c.startswith("feature_")]
    if len(feature_cols) <= dino_dims:
        raise ValueError("feature table does not contain an explicit thermal block after DINO dimensions")

    label_counts = frame.groupby("subject_id")["label"].nunique()
    if (label_counts > 1).any():
        raise ValueError("inconsistent labels inside subject IDs")

    subject_x = frame.groupby("subject_id")[feature_cols].mean()
    subject_label = frame.groupby("subject_id")["label"].first().reindex(subject_x.index)
    y = subject_label.isin(["CANCER", "MALIGNANT", "TUMOR_BEARING"]).astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("binary labels are required")

    x = subject_x.to_numpy(dtype=np.float64)
    blocks = {
        "dino_only": x[:, :dino_dims],
        "thermal_only": x[:, dino_dims:],
        "fused": x,
    }

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    results = {}
    for block_name, block_x in blocks.items():
        p = np.full(len(y), np.nan)
        for train_idx, test_idx in cv.split(block_x, y):
            fitted = clone(model)
            fitted.fit(block_x[train_idx], y[train_idx])
            p[test_idx] = fitted.predict_proba(block_x[test_idx])[:, 1]
        results[block_name] = _metrics(y, p)

    return {
        "subjects": int(len(y)),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "dino_dims": int(dino_dims),
        "explicit_dims": int(x.shape[1] - dino_dims),
        "folds": int(folds),
        "seed": int(seed),
        "results": results,
        "clinical_claim": "NONE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_table", type=Path)
    parser.add_argument("--dino-dims", type=int, default=384)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    result = run(args.feature_table, args.dino_dims, args.folds, args.seed)
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
