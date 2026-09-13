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


def _feature_vector(features: Mapping[str, float], names: tuple[str, ...]) -> np.ndarray:
    missing = [name for name in names if name not in features]
    if missing:
        raise ValueError(f"missing source-support features: {missing}")
    vector = np.asarray([features[name] for name in names], dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("source-support feature vector contains NaN/Inf")
    return vector


def _mahalanobis_distance(vector: np.ndarray, artifact: Mapping[str, object]) -> float:
    mean = np.asarray(artifact.get("scaler_mean"), dtype=np.float64)
    scale = np.asarray(artifact.get("scaler_scale"), dtype=np.float64)
    source_mean = np.asarray(artifact.get("standardized_source_mean"), dtype=np.float64)
    covariance = np.asarray(artifact.get("standardized_source_covariance"), dtype=np.float64)
    regularization = float(artifact.get("covariance_regularization", 1e-6))

    if mean.shape != vector.shape or scale.shape != vector.shape or source_mean.shape != vector.shape:
        raise ValueError("source support scaler shape mismatch")
    if covariance.shape != (vector.size, vector.size):
        raise ValueError("source support covariance shape mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("source support scaler is invalid")
    if not np.isfinite(source_mean).all() or not np.isfinite(covariance).all():
        raise ValueError("source support geometry contains NaN/Inf")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("source support covariance regularization is invalid")

    standardized = (vector - mean) / scale
    inverse = np.linalg.inv(covariance + np.eye(vector.size, dtype=np.float64) * regularization)
    delta = standardized - source_mean
    squared = float(delta @ inverse @ delta)
    if not np.isfinite(squared):
        raise ValueError("source support distance is invalid")
    return float(np.sqrt(max(squared, 0.0)))


def evaluate_session_source_support(
    left_field: np.ndarray,
    right_field: np.ndarray,
    *,
    left_support: np.ndarray | None = None,
    right_support: np.ndarray | None = None,
    measurement_status: str,
    artifact: Mapping[str, object] | None = None,
) -> dict:
    """Check whether a MumGuard session resembles the auxiliary DMR-IR feature domain.

    This function intentionally performs **no disease classification**. It only
    asks whether the dimensionless local morphology extracted from the MumGuard
    relative-TLC fields lies inside a conservative source-domain support region.
    """

    status = str(measurement_status or "UNKNOWN")
    if status != "OK":
        return {
            "status": "MEASUREMENT_INSUFFICIENT",
            "distance": None,
            "threshold": None,
            "feature_contract": FEATURE_CONTRACT,
            "research_decision": "INCONCLUSIVE",
            "decision_model_executed": False,
            "clinical_claim": "NONE",
            "reason": "MumGuard measurement support is insufficient for source-domain comparison.",
        }

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
        # The DMR-IR candidate artifact was trained after averaging available
        # local morphology per subject. MumGuard always supplies both sides, so
        # average the two local vectors for this source-support comparison only.
        session_vector = (left_vector + right_vector) / 2.0
        distance = _mahalanobis_distance(session_vector, support_artifact)
        threshold = float(support_artifact["threshold"])
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("source support threshold is invalid")
    except Exception as exc:
        return {
            "status": "SUPPORT_CHECK_UNAVAILABLE",
            "distance": None,
            "threshold": None,
            "feature_contract": FEATURE_CONTRACT,
            "research_decision": "INCONCLUSIVE",
            "decision_model_executed": False,
            "clinical_claim": "NONE",
            "reason": str(exc),
        }

    in_domain = distance <= threshold
    return {
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
        "research_decision": "INCONCLUSIVE",
        "decision_model_executed": False,
        "clinical_claim": "NONE",
        "reason": (
            "Session morphology is inside the auxiliary human-IR source support; no disease decision is implied."
            if in_domain
            else "Session morphology is outside the auxiliary human-IR source support; transferred disease inference must abstain."
        ),
    }
