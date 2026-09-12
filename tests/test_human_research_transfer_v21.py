import numpy as np
import pandas as pd

from app.services.human_research_transfer_v21 import (
    LOCAL_SPATIAL_FEATURES,
    extract_dimensionless_local_spatial_features,
)
from scripts.train_human_research_transfer_v21 import load_subject_table, train


def _field():
    y, x = np.mgrid[0:64, 0:64]
    base = 0.02 * x + 0.01 * y
    hotspot = 1.5 * np.exp(-((x - 44) ** 2 + (y - 24) ** 2) / (2 * 5.0**2))
    return base + hotspot


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
    rows = []
    for i in range(8):
        label = "HEALTHY" if i < 4 else "CANCER"
        subject = f"S{i}"
        # Healthy source rows can have both sides while cancer source rows may
        # have one side only. Side availability must not enter the model.
        sides = ["LEFT", "RIGHT"] if label == "HEALTHY" else ["LEFT"]
        for side in sides:
            row = {"subject_id": subject, "label": label, "side": side}
            for j, name in enumerate(LOCAL_SPATIAL_FEATURES):
                row[name] = 0.2 * i + 0.01 * j
            rows.append(row)
    path = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    subjects = load_subject_table(path)
    assert len(subjects) == 8
    assert subjects["y"].value_counts().to_dict() == {0: 4, 1: 4}
    summary, _, bundle = train(subjects, folds=2)
    assert summary["subjects"] == 8
    assert summary["source_side_availability_used"] is False
    assert summary["source_bilateral_features_used"] is False
    assert bundle["status"] == "RESEARCH_TRANSFER_CANDIDATE"
