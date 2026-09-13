from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from app.services.human_research_transfer_v21 import (
    FEATURE_CONTRACT,
    LOCAL_SPATIAL_FEATURES,
    extract_dimensionless_local_spatial_features,
)


DEFAULT_SUPPORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research_artifacts"
    / "human_transfer_v21_source_support.json"
)
DEFAULT_RESEARCH_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "research_artifacts"
    / "human_transfer_v21_research_model.json"
)


def load_source_support(path: str | Path = DEFAULT_SUPPORT_PATH) -> dict:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("clinical_claim") != "NONE":
        raise ValueError("source support artifact must keep clinical_claim=NONE")
    if artifact.get("decision_model_included") is not False:
        raise ValueError("source support artifact must not include a disease decision model")
    if artifact.get("feature_contract") != FEATURE_CONTRACT:
        raise ValueError("source support feature contract mismatch")
    names = tuple(str(name) for name in artifact.get("feature_names", ()))
    if names != LOCAL_SPATIAL_FEATURES:
        raise ValueError("source support feature names mismatch")
    return artifact


def load_research_model(path: str | Path = DEFAULT_RESEARCH_MODEL_PATH) -> dict:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("clinical_claim") != "NONE":
        raise ValueError("research transfer artifact must keep clinical_claim=NONE")
    if artifact.get("target_domain_calibration") != "NOT_AVAILABLE":
        raise ValueError("research transfer artifact must not claim target-domain calibration")
    if artifact.get("feature_contract") != FEATURE_CONTRACT:
        raise ValueError("research transfer feature contract mismatch")
    names = tuple(str(name) for name in artifact.get("feature_names", ()))
    if names != LOCAL_SPATIAL_FEATURES:
        raise ValueError("research transfer feature names mismatch")
    coefficients = np.asarray(artifact.get("coefficients"), dtype=np.float64)
    if coefficients.shape != (len(LOCAL_SPATIAL_FEATURES),) or not np.isfinite(coefficients).all():
        raise ValueError("research transfer coefficients are invalid")
    intercept = float(artifact.get("intercept"))
    threshold = float(artifact.get("decision_threshold"))
    if not np.isfinite(intercept):
        raise ValueError("research transfer intercept is invalid")
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("research transfer decision threshold is invalid")
    return artifact


def _feature_vector(features: Mapping[str, float], names: tuple[str, ...]) -> np.ndarray:
    missing = [name for name in names if name not in features]
    if missing:
        raise ValueError(f"missing source-support features: {missing}")
    vector = np.asarray([features[name] for name in names], dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("source-support feature vector contains NaN/Inf")
    return vector


def _standardize(vector: np.ndarray, artifact: Mapping[str, object]) -> np.ndarray:
    mean = np.asarray(artifact.get("scaler_mean"), dtype=np.float64)
    scale = np.asarray(artifact.get("scaler_scale"), dtype=np.float64)
    if mean.shape != vector.shape or scale.shape != vector.shape:
        raise ValueError("source support scaler shape mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("source support scaler is invalid")
    return (vector - mean) / scale


def _mahalanobis_distance(vector: np.ndarray, artifact: Mapping[str, object]) -> float:
    standardized = _standardize(vector, artifact)
    source_mean = np.asarray(artifact.get("standardized_source_mean"), dtype=np.float64)
    covariance = np.asarray(artifact.get("standardized_source_covariance"), dtype=np.float64)
    regularization = float(artifact.get("covariance_regularization", 1e-6))

    if source_mean.shape != vector.shape:
        raise ValueError("source support centroid shape mismatch")
    if covariance.shape != (vector.size, vector.size):
        raise ValueError("source support covariance shape mismatch")
    if not np.isfinite(source_mean).all() or not np.isfinite(covariance).all():
        raise ValueError("source support geometry contains NaN/Inf")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("source support covariance regularization is invalid")

    inverse = np.linalg.inv(covariance + np.eye(vector.size, dtype=np.float64) * regularization)
    delta = standardized - source_mean
    squared = float(delta @ inverse @ delta)
    if not np.isfinite(squared):
        raise ValueError("source support distance is invalid")
    return float(np.sqrt(max(squared, 0.0)))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return float(1.0 / (1.0 + np.exp(-value)))
    exp_value = float(np.exp(value))
    return float(exp_value / (1.0 + exp_value))


def _apply_research_transfer(
    vector: np.ndarray,
    support_artifact: Mapping[str, object],
    research_artifact: Mapping[str, object],
) -> tuple[float, str]:
    support_names = tuple(str(name) for name in support_artifact.get("feature_names", ()))
    research_names = tuple(str(name) for name in research_artifact.get("feature_names", ()))
    if support_names != research_names or support_names != LOCAL_SPATIAL_FEATURES:
        raise ValueError("research transfer and source-support feature contracts differ")
    if research_artifact.get("requires_source_support_status") != "IN_SOURCE_SUPPORT":
        raise ValueError("research transfer artifact must require source-support gating")

    standardized = _standardize(vector, support_artifact)
    coefficients = np.asarray(research_artifact.get("coefficients"), dtype=np.float64)
    intercept = float(research_artifact.get("intercept"))
    logit = float(intercept + standardized @ coefficients)
    if not np.isfinite(logit):
        raise ValueError("research transfer logit is invalid")
    score = _sigmoid(logit)
    threshold = float(research_artifact.get("decision_threshold"))
    status = "SUSPICIOUS_RESEARCH" if score >= threshold else "NOT_SUSPICIOUS_RESEARCH"
    return score, status


def _inconclusive_payload(status: str, reason: str) -> dict:
    return {
        "status": status,
        "distance": None,
        "threshold": None,
        "feature_contract": FEATURE_CONTRACT,
        "research_decision": "INCONCLUSIVE",
        "research_concern_score": None,
        "source_transfer_score_0_1": None,
        "decision_model_executed": False,
        "target_domain_calibration": "NOT_AVAILABLE",
        "clinical_risk": None,
        "clinical_claim": "NONE",
        "reason": reason,
    }


def evaluate_session_source_support(
    left_field: np.ndarray,
    right_field: np.ndarray,
    *,
    left_support: np.ndarray | None = None,
    right_support: np.ndarray | None = None,
    measurement_status: str,
    artifact: Mapping[str, object] | None = None,
    research_artifact: Mapping[str, object] | None = None,
) -> dict:
    """Gate and, when supported, execute the auxiliary human research transfer head.

    The gate compares dimensionless local morphology from the MumGuard relative-TLC
    fields with the auxiliary DMR-IR human source domain. The reviewed logistic
    transfer head is executed only when that source-support gate passes. Its output
    is a research concern index and research label, never a cancer probability,
    target-domain calibration, clinical risk, or diagnosis.
    """

    status = str(measurement_status or "UNKNOWN")
    if status != "OK":
        return _inconclusive_payload(
            "MEASUREMENT_INSUFFICIENT",
            "MumGuard measurement support is insufficient for research transfer.",
        )

    try:
        support_artifact = dict(artifact) if artifact is not None else load_source_support()
        names = tuple(str(name) for name in support_artifact["feature_names"])
        left_features = extract_dimensionless_local_spatial_features(
            left_field,
            left_support,
        )
        right_features = extract_dimensionless_local_spatial_features(
            right_field,
            right_support,
        )
        left_vector = _feature_vector(left_features, names)
        right_vector = _feature_vector(right_features, names)
        # The DMR-IR candidate was trained after subject-level averaging of local
        # morphology. MumGuard always has LEFT and RIGHT, so average the two local
        # vectors for transfer without using side availability as a disease proxy.
        session_vector = (left_vector + right_vector) / 2.0
        distance = _mahalanobis_distance(session_vector, support_artifact)
        threshold = float(support_artifact["threshold"])
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("source support threshold is invalid")
    except Exception as exc:
        return _inconclusive_payload("SUPPORT_CHECK_UNAVAILABLE", str(exc))

    in_domain = distance <= threshold
    base = {
        "status": "IN_SOURCE_SUPPORT" if in_domain else "ABSTAIN_OOD",
        "distance": round(distance, 6),
        "threshold": round(threshold, 6),
        "distance_over_threshold": round(distance / threshold, 6),
        "feature_contract": FEATURE_CONTRACT,
        "feature_names": list(names),
        "session_features": {
            name: round(float(value), 8)
            for name, value in zip(names, session_vector)
        },
        "left_features": {name: round(float(left_features[name]), 8) for name in names},
        "right_features": {name: round(float(right_features[name]), 8) for name in names},
        "source_domain": support_artifact.get("source_domain"),
        "source_model_id": support_artifact.get("model_id"),
        "source_model_version": support_artifact.get("model_version"),
        "distance_metric": support_artifact.get("distance_metric"),
        "source_count": support_artifact.get("source_count"),
        "target_domain_calibration": "NOT_AVAILABLE",
        "clinical_risk": None,
        "clinical_claim": "NONE",
    }

    if not in_domain:
        return {
            **base,
            "research_decision": "INCONCLUSIVE",
            "research_concern_score": None,
            "source_transfer_score_0_1": None,
            "decision_model_executed": False,
            "reason": (
                "Session morphology is outside the auxiliary human-IR source support; "
                "the research transfer head abstains."
            ),
        }

    try:
        model_artifact = (
            dict(research_artifact)
            if research_artifact is not None
            else load_research_model()
        )
        score, research_decision = _apply_research_transfer(
            session_vector,
            support_artifact,
            model_artifact,
        )
    except Exception as exc:
        return {
            **base,
            "research_decision": "INCONCLUSIVE",
            "research_concern_score": None,
            "source_transfer_score_0_1": None,
            "decision_model_executed": False,
            "reason": f"Research transfer model unavailable: {exc}",
        }

    return {
        **base,
        "research_decision": research_decision,
        "research_concern_score": round(score * 100.0, 2),
        "source_transfer_score_0_1": round(score, 6),
        "decision_model_executed": True,
        "decision_model_id": model_artifact.get("model_id"),
        "decision_model_version": model_artifact.get("model_version"),
        "decision_threshold": float(model_artifact.get("decision_threshold")),
        "source_oof_metrics": model_artifact.get("source_oof_metrics"),
        "score_semantics": model_artifact.get("score_semantics"),
        "reason": (
            "Auxiliary human-IR morphology is inside source support; the reviewed "
            "v0.2.1 research transfer head produced this research-only indication."
        ),
    }
