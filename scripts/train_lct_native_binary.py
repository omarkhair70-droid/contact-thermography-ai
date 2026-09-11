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
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

POSITIVE_LABELS = {"CANCER", "MALIGNANT", "TUMOR_BEARING", "TUMOR_LIKE"}
NEGATIVE_LABELS = {"HEALTHY", "BENIGN", "NO_TUMOR_LIKE"}


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _binary_label(value: str) -> int:
    label = str(value).strip().upper()
    if label in POSITIVE_LABELS:
        return 1
    if label in NEGATIVE_LABELS:
        return 0
    raise ValueError(f"unsupported binary label: {value}")


def readiness(manifest_path: Path, min_per_class: int = 2) -> dict:
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    required = {
        "subject_id", "modality", "species", "label", "split_group", "tlc_profile_id",
        "device_profile_id", "use_role", "train_eligible", "label_provenance",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing manifest columns: {missing}")

    train = frame[frame["train_eligible"].map(_as_bool)].copy()
    reasons: list[str] = []
    warnings: list[str] = []
    if train.empty:
        reasons.append("no train_eligible target-domain rows")
    if not train.empty and set(train["modality"]) != {"contact-LCT"}:
        reasons.append("all native rows must be contact-LCT")
    if not train.empty and train["species"].nunique() != 1:
        reasons.append("trainable native pool must contain one species")
    if not train.empty and train["tlc_profile_id"].nunique() != 1:
        reasons.append("trainable native pool must contain one TLC profile")
    if not train.empty and train["device_profile_id"].nunique() != 1:
        reasons.append("trainable native pool must contain one device profile")
    if not train.empty and not (train["use_role"] == "TRAIN_CANDIDATE").all():
        reasons.append("trainable rows must use TRAIN_CANDIDATE role")

    positive = int(train["label"].isin(POSITIVE_LABELS).sum())
    negative = int(train["label"].isin(NEGATIVE_LABELS).sum())
    unknown = int(len(train) - positive - negative)
    if positive < min_per_class:
        reasons.append(f"need at least {min_per_class} positive subjects; found {positive}")
    if negative < min_per_class:
        reasons.append(f"need at least {min_per_class} negative subjects; found {negative}")
    if unknown:
        reasons.append(f"{unknown} trainable rows have unsupported/unknown binary labels")
    if min(positive, negative) < 5 and positive and negative:
        warnings.append(
            "very small class counts: metrics are exploratory internal research only, not independent validation"
        )

    duplicate_subjects = train[train["subject_id"].duplicated()]["subject_id"].tolist()
    if duplicate_subjects:
        reasons.append(f"duplicate trainable subject IDs: {duplicate_subjects[:20]}")

    return {
        "binary_training_ready": not reasons,
        "trainable_subjects": int(train["subject_id"].nunique()),
        "positive_subjects": positive,
        "negative_subjects": negative,
        "species": sorted(train["species"].unique().tolist()),
        "tlc_profiles": sorted(train["tlc_profile_id"].unique().tolist()),
        "device_profiles": sorted(train["device_profile_id"].unique().tolist()),
        "reasons": reasons,
        "warnings": warnings,
        "clinical_claim": "NONE",
    }


def _metrics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    pred = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "auroc": float(roc_auc_score(y_true, probability)),
        "auprc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "clinical_claim": "NONE",
    }


def train(manifest_path: Path, features_path: Path, output_dir: Path, min_per_class: int = 2, folds: int = 5, seed: int = 20260910) -> dict:
    gate = readiness(manifest_path, min_per_class=min_per_class)
    if not gate["binary_training_ready"]:
        raise ValueError("target-domain training gate closed: " + "; ".join(gate["reasons"]))

    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    manifest = manifest[manifest["train_eligible"].map(_as_bool)].copy()
    features = pd.read_csv(features_path, keep_default_na=False)
    feature_cols = [c for c in features.columns if c.startswith("feature_")]
    if "subject_id" not in features or not feature_cols:
        raise ValueError("features CSV requires subject_id and feature_* columns")

    numeric = features[feature_cols].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("features contain NaN/Inf/non-numeric values")
    features[feature_cols] = numeric
    subject_features = features.groupby("subject_id", sort=True)[feature_cols].mean().reset_index()

    joined = manifest[["subject_id", "label"]].merge(subject_features, on="subject_id", how="left", validate="one_to_one")
    if joined[feature_cols].isna().any().any():
        missing = joined.loc[joined[feature_cols].isna().any(axis=1), "subject_id"].tolist()
        raise ValueError(f"missing features for trainable subjects: {missing[:20]}")

    X = joined[feature_cols].to_numpy(dtype=np.float64)
    y = joined["label"].map(_binary_label).to_numpy(dtype=int)
    class_counts = np.bincount(y, minlength=2)
    min_class_count = int(class_counts.min())
    n_folds = min(int(folds), min_class_count)
    if n_folds < 2:
        raise ValueError(f"insufficient class counts for cross-validation: {class_counts.tolist()}")

    models = {
        "logistic": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed)),
        ]),
    }
    skipped_models: dict[str, str] = {}
    if min_class_count >= 3:
        models["linear_svm_calibrated"] = Pipeline([
            ("scale", StandardScaler()),
            ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=seed), cv=2, method="sigmoid")),
        ])
    else:
        skipped_models["linear_svm_calibrated"] = (
            "requires at least 3 subjects in each class so every outer-CV training fold can support 2-fold calibration"
        )

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = joined[["subject_id", "label"]].copy()
    result = {
        "evaluation": "subject-level contact-LCT native research baseline",
        "readiness": gate,
        "folds": n_folds,
        "feature_count": len(feature_cols),
        "models": {},
        "skipped_models": skipped_models,
        "clinical_claim": "NONE",
        "semantics": "research classification only; not a clinical diagnosis or cancer probability",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, template in models.items():
        probability = np.full(len(joined), np.nan, dtype=float)
        for train_idx, test_idx in cv.split(X, y):
            model = clone(template)
            model.fit(X[train_idx], y[train_idx])
            probability[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        result["models"][name] = _metrics(y, probability)
        oof[f"{name}_tumor_like_probability"] = probability
        final_model = clone(template).fit(X, y)
        joblib.dump({
            "model": final_model,
            "feature_columns": feature_cols,
            "species": gate["species"][0],
            "tlc_profile_id": gate["tlc_profiles"][0],
            "device_profile_id": gate["device_profiles"][0],
            "clinical_claim": "NONE",
        }, output_dir / f"{name}.joblib")

    (output_dir / "lct_native_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    oof.to_csv(output_dir / "lct_native_oof_predictions.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-per-class", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--readiness-only", action="store_true")
    args = parser.parse_args()

    if args.readiness_only:
        print(json.dumps(readiness(args.manifest, args.min_per_class), indent=2))
        return
    if args.features is None or args.output_dir is None:
        parser.error("--features and --output-dir are required unless --readiness-only is used")
    print(json.dumps(train(args.manifest, args.features, args.output_dir, args.min_per_class, args.folds), indent=2))


if __name__ == "__main__":
    main()
