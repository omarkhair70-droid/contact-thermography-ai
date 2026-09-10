from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
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
from sklearn.svm import LinearSVC

POSITIVE_LABELS = {"CANCER", "MALIGNANT", "TUMOR_LIKE"}
NEGATIVE_LABELS = {"HEALTHY", "BENIGN", "NO_TUMOR_LIKE"}


def _binary_label(value: str) -> int:
    value = str(value).strip().upper()
    if value in POSITIVE_LABELS:
        return 1
    if value in NEGATIVE_LABELS:
        return 0
    raise ValueError(f"Unsupported binary label: {value}")


def load_subject_features(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"subject_id", "label"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    feature_cols = [c for c in frame.columns if c.startswith("feature_")]
    if not feature_cols:
        raise ValueError("No feature_* columns found")

    label_counts = frame.groupby("subject_id")["label"].nunique()
    bad = label_counts[label_counts > 1].index.tolist()
    if bad:
        raise ValueError(f"Subjects with inconsistent labels: {bad[:20]}")

    numeric = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Features contain NaN/Inf/non-numeric values")
    frame[feature_cols] = numeric

    labels = frame.groupby("subject_id", sort=True)["label"].first()
    features = frame.groupby("subject_id", sort=True)[feature_cols].mean()
    subjects = features.reset_index()
    subjects["label"] = labels.reindex(subjects["subject_id"]).to_numpy()
    subjects["y"] = subjects["label"].map(_binary_label)
    return subjects, feature_cols


def _metrics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    pred = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "subjects": int(len(y_true)),
        "positives": int(y_true.sum()),
        "negatives": int((1 - y_true).sum()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auroc": float(roc_auc_score(y_true, probability)),
        "auprc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "clinical_claim": "NONE",
    }


def evaluate_models(subjects: pd.DataFrame, feature_cols: list[str], folds: int = 5, seed: int = 20260910) -> tuple[dict, pd.DataFrame, dict[str, object]]:
    X = subjects[feature_cols].to_numpy(dtype=np.float64)
    y = subjects["y"].to_numpy(dtype=int)
    class_counts = np.bincount(y, minlength=2)
    max_folds = int(class_counts.min())
    folds = min(int(folds), max_folds)
    if folds < 2:
        raise ValueError(f"Need at least two subjects in each class; counts={class_counts.tolist()}")

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    models = {
        "logistic": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed)),
        ]),
        "linear_svm_calibrated": Pipeline([
            ("scale", StandardScaler()),
            ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=seed), cv=3, method="sigmoid")),
        ]),
    }

    all_metrics: dict[str, dict] = {}
    oof = subjects[["subject_id", "label", "y"]].copy()
    fitted: dict[str, object] = {}

    for name, template in models.items():
        probs = np.full(len(subjects), np.nan, dtype=float)
        for train_idx, test_idx in cv.split(X, y):
            model = clone(template)
            model.fit(X[train_idx], y[train_idx])
            probs[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        if not np.isfinite(probs).all():
            raise RuntimeError(f"Non-finite OOF probabilities for {name}")
        all_metrics[name] = _metrics(y, probs)
        oof[f"{name}_tumor_like_probability"] = probs
        final_model = clone(template)
        final_model.fit(X, y)
        fitted[name] = final_model

    summary = {
        "evaluation": "subject-level out-of-fold auxiliary infrared baseline",
        "folds": folds,
        "features": len(feature_cols),
        "models": all_metrics,
        "clinical_claim": "NONE",
        "semantics": "auxiliary research classification only; not validated for contact-LCT or clinical diagnosis",
    }
    return summary, oof, fitted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    subjects, feature_cols = load_subject_features(args.features_csv)
    summary, oof, fitted = evaluate_models(subjects, feature_cols, folds=args.folds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aux_binary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    oof.to_csv(args.output_dir / "aux_binary_oof_predictions.csv", index=False)
    for name, model in fitted.items():
        joblib.dump({"model": model, "feature_columns": feature_cols, "clinical_claim": "NONE"}, args.output_dir / f"{name}.joblib")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
