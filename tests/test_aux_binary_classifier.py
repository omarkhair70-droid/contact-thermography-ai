from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train_aux_binary_from_embeddings import evaluate_models, load_subject_features


def _make_separable_frame() -> pd.DataFrame:
    rows = []
    for i in range(12):
        positive = i >= 6
        center = 2.0 if positive else -2.0
        for view in range(2):
            rows.append(
                {
                    "subject_id": f"S{i:02d}",
                    "label": "CANCER" if positive else "HEALTHY",
                    "feature_000": center + 0.02 * view,
                    "feature_001": center * 0.5 + 0.01 * i,
                }
            )
    return pd.DataFrame(rows)


def test_subject_rows_are_aggregated_before_evaluation(tmp_path: Path):
    path = tmp_path / "features.csv"
    _make_separable_frame().to_csv(path, index=False)
    subjects, feature_cols = load_subject_features(path)
    assert len(subjects) == 12
    assert feature_cols == ["feature_000", "feature_001"]
    assert subjects["y"].sum() == 6


def test_auxiliary_models_produce_finite_subject_level_oof_predictions(tmp_path: Path):
    path = tmp_path / "features.csv"
    _make_separable_frame().to_csv(path, index=False)
    subjects, feature_cols = load_subject_features(path)
    summary, oof, fitted = evaluate_models(subjects, feature_cols, folds=3)
    assert summary["clinical_claim"] == "NONE"
    assert set(summary["models"]) == {"logistic", "linear_svm_calibrated"}
    assert len(oof) == 12
    for column in [c for c in oof.columns if c.endswith("_probability")]:
        values = oof[column].to_numpy(dtype=float)
        assert np.isfinite(values).all()
        assert ((values >= 0.0) & (values <= 1.0)).all()
    assert fitted


def test_inconsistent_subject_label_is_rejected(tmp_path: Path):
    frame = _make_separable_frame()
    frame.loc[1, "label"] = "CANCER"
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    try:
        load_subject_features(path)
    except ValueError as exc:
        assert "inconsistent labels" in str(exc).lower()
    else:
        raise AssertionError("expected inconsistent subject labels to fail")
