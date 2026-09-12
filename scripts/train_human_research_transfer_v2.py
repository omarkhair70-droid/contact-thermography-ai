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

from app.services.human_research_transfer_v2 import BILATERAL_SPATIAL_FEATURES, FEATURE_CONTRACT


POSITIVE_LABELS = {"CANCER", "MALIGNANT", "TUMOR_LIKE", "SICK"}
NEGATIVE_LABELS = {"HEALTHY", "NO_TUMOR_LIKE", "CONTROL"}


def _label(value: object) -> int:
    normalized = str(value).strip().upper()
    if normalized in POSITIVE_LABELS:
        return 1
    if normalized in NEGATIVE_LABELS:
        return 0
    raise ValueError(f"Unsupported source-domain label: {value!r}")


def load_subject_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"subject_id", "label", *BILATERAL_SPATIAL_FEATURES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame["subject_id"].duplicated().any():
        duplicate = frame.loc[frame["subject_id"].duplicated(), "subject_id"].tolist()
        raise ValueError(f"v0.2 expects one row per subject; duplicates={duplicate[:20]}")

    numeric = frame[list(BILATERAL_SPATIAL_FEATURES)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Bilateral spatial features contain NaN/Inf/non-numeric values")
    frame[list(BILATERAL_SPATIAL_FEATURES)] = numeric
    frame["y"] = frame["label"].map(_label)
    return frame.sort_values("subject_id").reset_index(drop=True)


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
    }


def _fit_ood_gate(model: Pipeline, X: np.ndarray, quantile: float) -> tuple[np.ndarray, float, dict]:
    scaler = model.named_steps["scale"]
    source = np.asarray(scaler.transform(X), dtype=np.float64)
    distances = np.linalg.norm(source[:, None, :] - source[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    threshold = float(np.quantile(nearest, quantile))
    return source, threshold, {
        "metric": "standardized_euclidean_nearest_source",
        "threshold_quantile": float(quantile),
        "threshold": threshold,
        "source_count": int(len(source)),
        "source_nn_median": float(np.median(nearest)),
        "source_nn_q95": float(np.quantile(nearest, 0.95)),
    }


def train(
    subjects: pd.DataFrame,
    *,
    folds: int = 5,
    seed: int = 20260913,
    ood_quantile: float = 0.95,
) -> tuple[dict, pd.DataFrame, dict]:
    X = subjects[list(BILATERAL_SPATIAL_FEATURES)].to_numpy(dtype=np.float64)
    y = subjects["y"].to_numpy(dtype=int)
    counts = np.bincount(y, minlength=2)
    folds = min(int(folds), int(counts.min()))
    if folds < 2:
        raise ValueError(f"Need at least two subjects in each class; counts={counts.tolist()}")
    if not 0.5 <= ood_quantile < 1.0:
        raise ValueError("ood_quantile must be in [0.5, 1.0)")

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    templates = {
        "logistic": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed)),
        ]),
        "linear_svm_calibrated": Pipeline([
            ("scale", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(class_weight="balanced", random_state=seed),
                cv=3,
                method="sigmoid",
            )),
        ]),
    }

    metrics: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    oof = subjects[["subject_id", "label", "y"]].copy()
    for name, template in templates.items():
        probability = np.full(len(subjects), np.nan, dtype=np.float64)
        for train_idx, test_idx in cv.split(X, y):
            model = clone(template)
            model.fit(X[train_idx], y[train_idx])
            probability[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        if not np.isfinite(probability).all():
            raise RuntimeError(f"Non-finite OOF predictions for {name}")
        metrics[name] = _metrics(y, probability)
        oof[f"{name}_research_concern"] = probability
        final_model = clone(template)
        final_model.fit(X, y)
        fitted[name] = final_model

    selected_name = sorted(
        metrics,
        key=lambda name: (
            -metrics[name]["balanced_accuracy"],
            metrics[name]["brier"],
            0 if name == "logistic" else 1,
        ),
    )[0]
    selected_model = fitted[selected_name]
    source_standardized, ood_threshold, ood = _fit_ood_gate(selected_model, X, ood_quantile)

    summary = {
        "model_id": "mumguard-human-research-transfer-v2",
        "model_version": "0.2.0",
        "source_domain": "DMR-IR_AUXILIARY_HUMAN_INFRARED",
        "target_domain": "MUMGUARD_CONTACT_LCT_RELATIVE_SIGNAL",
        "feature_contract": FEATURE_CONTRACT,
        "feature_names": list(BILATERAL_SPATIAL_FEATURES),
        "subjects": int(len(subjects)),
        "folds": int(folds),
        "selection_metric": "out_of_fold_balanced_accuracy_then_brier",
        "selected_model": selected_name,
        "models": metrics,
        "ood_gate": ood,
        "clinical_claim": "NONE",
        "score_semantics": (
            "research transfer concern index only; not a cancer probability and not validation of MumGuard"
        ),
        "transfer_boundary": (
            "features preserve bilateral/spatial morphology while removing pooled additive offset and positive scale; "
            "no absolute Celsius threshold is transferred into Contact-TLC"
        ),
    }
    bundle = {
        "status": "RESEARCH_TRANSFER_CANDIDATE",
        "model_id": summary["model_id"],
        "model_version": summary["model_version"],
        "source_domain": summary["source_domain"],
        "target_domain": summary["target_domain"],
        "feature_contract": summary["feature_contract"],
        "feature_names": list(BILATERAL_SPATIAL_FEATURES),
        "model": selected_model,
        "decision_threshold": 0.5,
        "ood_source_standardized": source_standardized,
        "ood_threshold": ood_threshold,
        "ood_gate": ood,
        "clinical_claim": "NONE",
        "score_semantics": summary["score_semantics"],
    }
    return summary, oof, bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MumGuard human research transfer v0.2")
    parser.add_argument("features_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--ood-quantile", type=float, default=0.95)
    args = parser.parse_args()

    subjects = load_subject_table(args.features_csv)
    summary, oof, bundle = train(
        subjects,
        folds=args.folds,
        seed=args.seed,
        ood_quantile=args.ood_quantile,
    )
    summary["artifact_status"] = "RESEARCH_TRANSFER_CANDIDATE"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary_v2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    oof.to_csv(args.output_dir / "oof_predictions_v2.csv", index=False)
    joblib.dump(bundle, args.output_dir / "mumguard-human-research-transfer-v2.joblib")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
