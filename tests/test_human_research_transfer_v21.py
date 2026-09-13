import numpy as np
import pandas as pd

from app.services.human_research_transfer_v21 import (
    LOCAL_SPATIAL_FEATURES,
    extract_dimensionless_local_spatial_features,
)
from scripts.train_human_research_transfer_v21 import (
    audit_side_grouped,
    build_subject_table,
    load_record_table,
    train,
)


def _field():
    y, x = np.mgrid[0:64, 0:64]
    base = 0.02 * x + 0.01 * y
    hotspot = 1.5 * np.exp(-((x - 44) ** 2 + (y - 24) ** 2) / (2 * 5.0**2))
    return base + hotspot


def _source_like_rows(subjects_per_class: int = 6):
    rows = []
    total = subjects_per_class * 2
    for i in range(total):
        label = "HEALTHY" if i < subjects_per_class else "CANCER"
        subject = f"S{i}"
        # Mirror the real DMR-IR coverage problem: controls can expose two sides,
        # while cancer source subjects can expose only one. This must not become
        # a learned disease feature.
        sides = ["LEFT", "RIGHT"] if label == "HEALTHY" else ["LEFT"]
        class_shift = 0.0 if label == "HEALTHY" else 0.7
        for side_index, side in enumerate(sides):
            row = {"subject_id": subject, "label": label, "side": side}
            for j, name in enumerate(LOCAL_SPATIAL_FEATURES):
                row[name] = class_shift + 0.04 * i + 0.01 * j + 0.003 * side_index
            rows.append(row)
    return rows


def test_local_spatial_features_are_positive_affine_invariant():
    field = _field()
    first = extract_dimensionless_local_spatial_features(field)
    shifted_scaled = extract_dimensionless_local_spatial_features(field * 7.5 + 31.0)
    assert tuple(first) == LOCAL_SPATIAL_FEATURES
    for name in LOCAL_SPATIAL_FEATURES:
        assert np.isclose(first[name], shifted_scaled[name], atol=1e-5)


def test_local_spatial_hotspot_changes_morphology():
    y, x = np.mgrid[0:64, 0:64]
    flat = 0.02 * x + 0.01 * y + 0.03 * np.sin(x / 6)
    hot = flat + 1.7 * np.exp(-((x - 46) ** 2 + (y - 18) ** 2) / (2 * 4.0**2))
    a = extract_dimensionless_local_spatial_features(flat)
    b = extract_dimensionless_local_spatial_features(hot)
    assert b["local_contrast_p95"] > a["local_contrast_p95"]
    assert b["hotspot_peak_z"] > a["hotspot_peak_z"]


def test_subject_training_does_not_require_bilateral_source_rows(tmp_path):
    path = tmp_path / "features.csv"
    pd.DataFrame(_source_like_rows()).to_csv(path, index=False)
    records = load_record_table(path)
    subjects = build_subject_table(records)

    assert len(subjects) == 12
    assert subjects["y"].value_counts().to_dict() == {0: 6, 1: 6}

    summary, _, bundle = train(subjects, folds=2)
    assert summary["subjects"] == 12
    assert summary["source_side_availability_used"] is False
    assert summary["source_bilateral_features_used"] is False
    assert bundle["status"] == "RESEARCH_TRANSFER_CANDIDATE"
    assert bundle["clinical_claim"] == "NONE"


def test_side_grouped_audit_never_splits_one_subject_across_model_semantics(tmp_path):
    path = tmp_path / "features.csv"
    pd.DataFrame(_source_like_rows()).to_csv(path, index=False)
    records = load_record_table(path)
    audit = audit_side_grouped(records, folds=3)

    assert audit["method"] == "side_level_grouped_oof_logistic_subject_weighted"
    assert audit["side_identity_used_as_feature"] is False
    assert audit["side_count_used_as_feature"] is False
    assert audit["subject_total_weight"] == 1.0
    assert audit["side_metrics"]["subjects"] == 18
    assert audit["subject_mean_probability_metrics"]["subjects"] == 12
    assert 0.0 <= audit["subject_mean_probability_metrics"]["balanced_accuracy"] <= 1.0
