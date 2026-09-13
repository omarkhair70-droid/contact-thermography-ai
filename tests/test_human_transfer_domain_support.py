import numpy as np

from app.services.human_research_transfer_v21 import (
    LOCAL_SPATIAL_FEATURES,
    extract_dimensionless_local_spatial_features,
)
from app.services.human_transfer_domain_support import (
    evaluate_session_source_support,
    load_research_model,
    load_source_support,
)


def _field():
    y, x = np.mgrid[0:64, 0:64]
    return (
        0.02 * x
        + 0.01 * y
        + 1.2 * np.exp(-((x - 43) ** 2 + (y - 22) ** 2) / (2 * 5.0**2))
    )


def _artifact(center_vector, *, threshold=1.0):
    n = len(LOCAL_SPATIAL_FEATURES)
    return {
        "feature_names": list(LOCAL_SPATIAL_FEATURES),
        "feature_contract": "dimensionless_local_spatial_v0_2_1",
        "scaler_mean": list(center_vector),
        "scaler_scale": [1.0] * n,
        "standardized_source_mean": [0.0] * n,
        "standardized_source_covariance": np.eye(n).tolist(),
        "covariance_regularization": 1e-6,
        "threshold": threshold,
        "source_domain": "TEST_SOURCE",
        "model_id": "test-support",
        "model_version": "0-test",
        "distance_metric": "mahalanobis_to_source_centroid_regularized_covariance",
        "source_count": 10,
    }


def _research_artifact(*, intercept):
    return {
        "model_id": "test-research-transfer",
        "model_version": "0-test",
        "feature_names": list(LOCAL_SPATIAL_FEATURES),
        "feature_contract": "dimensionless_local_spatial_v0_2_1",
        "coefficients": [0.0] * len(LOCAL_SPATIAL_FEATURES),
        "intercept": float(intercept),
        "decision_threshold": 0.5,
        "requires_source_support_status": "IN_SOURCE_SUPPORT",
        "target_domain_calibration": "NOT_AVAILABLE",
        "clinical_claim": "NONE",
        "score_semantics": "research concern only",
        "source_oof_metrics": {"subjects": 10},
    }


def test_default_artifacts_keep_research_and_clinical_semantics_separate():
    support = load_source_support()
    model = load_research_model()
    assert support["decision_model_included"] is False
    assert support["clinical_claim"] == "NONE"
    assert tuple(support["feature_names"]) == LOCAL_SPATIAL_FEATURES
    assert model["clinical_claim"] == "NONE"
    assert model["target_domain_calibration"] == "NOT_AVAILABLE"
    assert model["requires_source_support_status"] == "IN_SOURCE_SUPPORT"
    assert model["source_oof_metrics"]["subjects"] == 56


def test_in_domain_session_can_emit_suspicious_research():
    field = _field()
    features = extract_dimensionless_local_spatial_features(field)
    center = [features[name] for name in LOCAL_SPATIAL_FEATURES]
    result = evaluate_session_source_support(
        field,
        field,
        measurement_status="OK",
        artifact=_artifact(center),
        research_artifact=_research_artifact(intercept=8.0),
    )
    assert result["status"] == "IN_SOURCE_SUPPORT"
    assert result["distance"] == 0.0
    assert result["research_decision"] == "SUSPICIOUS_RESEARCH"
    assert result["research_concern_score"] > 99.0
    assert result["decision_model_executed"] is True
    assert result["clinical_risk"] is None
    assert result["clinical_claim"] == "NONE"


def test_in_domain_session_can_emit_not_suspicious_research():
    field = _field()
    features = extract_dimensionless_local_spatial_features(field)
    center = [features[name] for name in LOCAL_SPATIAL_FEATURES]
    result = evaluate_session_source_support(
        field,
        field,
        measurement_status="OK",
        artifact=_artifact(center),
        research_artifact=_research_artifact(intercept=-8.0),
    )
    assert result["status"] == "IN_SOURCE_SUPPORT"
    assert result["research_decision"] == "NOT_SUSPICIOUS_RESEARCH"
    assert result["research_concern_score"] < 1.0
    assert result["decision_model_executed"] is True
    assert result["clinical_risk"] is None


def test_source_support_abstains_when_far_from_source_geometry():
    field = _field()
    features = extract_dimensionless_local_spatial_features(field)
    far_center = [features[name] + 10.0 for name in LOCAL_SPATIAL_FEATURES]
    result = evaluate_session_source_support(
        field,
        field,
        measurement_status="OK",
        artifact=_artifact(far_center, threshold=2.0),
        research_artifact=_research_artifact(intercept=8.0),
    )
    assert result["status"] == "ABSTAIN_OOD"
    assert result["distance"] > result["threshold"]
    assert result["research_decision"] == "INCONCLUSIVE"
    assert result["research_concern_score"] is None
    assert result["decision_model_executed"] is False


def test_source_support_fails_closed_when_measurement_is_not_ok():
    field = np.full((64, 64), np.nan)
    result = evaluate_session_source_support(
        field,
        field,
        measurement_status="LIMITED",
    )
    assert result["status"] == "MEASUREMENT_INSUFFICIENT"
    assert result["distance"] is None
    assert result["research_decision"] == "INCONCLUSIVE"
    assert result["research_concern_score"] is None
    assert result["decision_model_executed"] is False
    assert result["clinical_claim"] == "NONE"
