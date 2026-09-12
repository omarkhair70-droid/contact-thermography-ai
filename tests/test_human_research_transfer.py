import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.services.human_research_transfer import (
    SHARED_THERMAL_SHAPE_FEATURES,
    extract_offset_invariant_thermal_shape,
    infer_research_transfer,
)


def _active_bundle():
    X = np.asarray([
        [0.10, 0.25, 0.12, 0.13, 0.40, -0.01, 0.08, 0.07],
        [0.12, 0.28, 0.14, 0.14, 0.44, 0.00, 0.09, 0.07],
        [0.14, 0.31, 0.15, 0.16, 0.48, 0.01, 0.10, 0.07],
        [0.48, 0.82, 0.40, 0.42, 1.20, 0.07, 0.18, 0.20],
        [0.52, 0.88, 0.43, 0.45, 1.28, 0.08, 0.19, 0.21],
        [0.56, 0.94, 0.46, 0.48, 1.36, 0.09, 0.20, 0.22],
    ], dtype=np.float64)
    y = np.asarray([0, 0, 0, 1, 1, 1], dtype=int)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=7)),
    ])
    model.fit(X, y)
    source = model.named_steps["scale"].transform(X)
    return {
        "status": "ACTIVE_RESEARCH_TRANSFER",
        "model_id": "test-human-transfer",
        "model_version": "0-test",
        "source_domain": "SYNTHETIC_TEST_ONLY",
        "feature_names": list(SHARED_THERMAL_SHAPE_FEATURES),
        "model": model,
        "decision_threshold": 0.5,
        "ood_source_standardized": source,
        "ood_threshold": 2.0,
        "clinical_claim": "NONE",
    }, X


def _mapping(row):
    return {name: float(value) for name, value in zip(SHARED_THERMAL_SHAPE_FEATURES, row)}


def test_offset_invariant_shape_features_ignore_constant_temperature_offset():
    y, x = np.mgrid[0:40, 0:40]
    field = x * 0.03 + y * 0.02 + np.sin(x / 4.0) * 0.1
    first = extract_offset_invariant_thermal_shape(field)
    shifted = extract_offset_invariant_thermal_shape(field + 37.5)
    assert tuple(first) == SHARED_THERMAL_SHAPE_FEATURES
    for name in SHARED_THERMAL_SHAPE_FEATURES:
        assert np.isclose(first[name], shifted[name], atol=1e-10)


def test_research_transfer_fails_closed_without_active_model():
    result = infer_research_transfer(
        {name: 0.2 for name in SHARED_THERMAL_SHAPE_FEATURES},
        None,
        measurement_status="OK",
    )
    assert result.research_decision == "INCONCLUSIVE"
    assert result.research_concern_score is None
    assert result.domain_status == "TRANSFER_MODEL_UNAVAILABLE"
    assert result.clinical_claim == "NONE"


def test_research_transfer_fails_closed_when_measurement_is_not_ok():
    bundle, X = _active_bundle()
    result = infer_research_transfer(_mapping(X[0]), bundle, measurement_status="LIMITED")
    assert result.research_decision == "INCONCLUSIVE"
    assert result.domain_status == "MEASUREMENT_INSUFFICIENT"
    assert result.research_concern_score is None


def test_research_transfer_abstains_out_of_domain():
    bundle, X = _active_bundle()
    far = X[-1] + np.asarray([8, 8, 8, 8, 8, 8, 8, 8], dtype=np.float64)
    result = infer_research_transfer(_mapping(far), bundle, measurement_status="OK")
    assert result.research_decision == "INCONCLUSIVE"
    assert result.domain_status == "ABSTAIN_OOD"
    assert result.research_concern_score is None
    assert result.nearest_source_distance > result.ood_threshold


def test_research_transfer_emits_research_labels_in_domain_only():
    bundle, X = _active_bundle()
    low = infer_research_transfer(_mapping(X[0]), bundle, measurement_status="OK")
    high = infer_research_transfer(_mapping(X[-1]), bundle, measurement_status="OK")

    assert low.domain_status == "IN_DOMAIN"
    assert high.domain_status == "IN_DOMAIN"
    assert low.research_decision == "NOT_SUSPICIOUS_RESEARCH"
    assert high.research_decision == "SUSPICIOUS_RESEARCH"
    assert 0.0 <= low.research_concern_score <= 100.0
    assert 0.0 <= high.research_concern_score <= 100.0
    assert high.research_concern_score > low.research_concern_score
    assert "not a clinical" in high.score_semantics
