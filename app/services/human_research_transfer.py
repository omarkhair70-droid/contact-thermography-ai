from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np


RESEARCH_DECISIONS = {
    "SUSPICIOUS_RESEARCH",
    "NOT_SUSPICIOUS_RESEARCH",
    "INCONCLUSIVE",
}

# Deliberately excludes absolute temperature. DMR-IR experiments showed that
# absolute temperature carries strong, potentially protocol-confounded signal,
# while live MumGuard Contact-TLC currently exposes relative signal rather than
# calibrated Celsius. These shape/spread features therefore form the first
# cross-modality transfer contract.
SHARED_THERMAL_SHAPE_FEATURES = (
    "std",
    "p90_minus_p10",
    "p50_minus_p10",
    "p90_minus_p50",
    "max_minus_min",
    "mean_minus_median",
    "p10_minus_min",
    "max_minus_p90",
)


@dataclass(frozen=True)
class ResearchTransferResult:
    research_decision: str
    research_concern_score: float | None
    domain_status: str
    measurement_status: str
    reason: str
    model_id: str | None
    model_version: str | None
    source_domain: str | None
    nearest_source_distance: float | None
    ood_threshold: float | None
    feature_names: tuple[str, ...]
    clinical_claim: str = "NONE"
    score_semantics: str = (
        "research transfer concern index from an auxiliary human thermal source model; "
        "not a cancer probability and not a clinical diagnosis"
    )

    def as_dict(self) -> dict:
        return {
            "research_decision": self.research_decision,
            "research_concern_score": self.research_concern_score,
            "domain_status": self.domain_status,
            "measurement_status": self.measurement_status,
            "reason": self.reason,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "source_domain": self.source_domain,
            "nearest_source_distance": self.nearest_source_distance,
            "ood_threshold": self.ood_threshold,
            "feature_names": list(self.feature_names),
            "clinical_claim": self.clinical_claim,
            "score_semantics": self.score_semantics,
        }


def extract_offset_invariant_thermal_shape(
    field: np.ndarray,
    support: np.ndarray | None = None,
) -> dict[str, float]:
    """Extract offset-invariant thermal shape features from a 2-D field.

    This helper is intentionally usable by both source-domain human thermal
    preprocessing and MumGuard relative-TLC fields. Adding a constant offset to
    the field must not change these features.
    """

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("thermal transfer features require a 2-D field")

    valid = np.isfinite(values)
    if support is not None:
        mask = np.asarray(support, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("support mask must match thermal field shape")
        valid &= mask

    selected = values[valid]
    if selected.size < 16:
        raise ValueError("at least 16 finite supported pixels are required")

    p10, p50, p90 = np.percentile(selected, [10, 50, 90])
    minimum = float(np.min(selected))
    maximum = float(np.max(selected))
    mean = float(np.mean(selected))
    std = float(np.std(selected))

    features = {
        "std": std,
        "p90_minus_p10": float(p90 - p10),
        "p50_minus_p10": float(p50 - p10),
        "p90_minus_p50": float(p90 - p50),
        "max_minus_min": float(maximum - minimum),
        "mean_minus_median": float(mean - p50),
        "p10_minus_min": float(p10 - minimum),
        "max_minus_p90": float(maximum - p90),
    }
    if not np.isfinite(list(features.values())).all():
        raise ValueError("thermal transfer features contain NaN/Inf")
    return features


def feature_vector(
    features: Mapping[str, float | int],
    feature_names: Sequence[str] = SHARED_THERMAL_SHAPE_FEATURES,
) -> np.ndarray:
    missing = [name for name in feature_names if name not in features]
    if missing:
        raise ValueError(f"missing transfer features: {missing}")
    vector = np.asarray([features[name] for name in feature_names], dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("transfer feature vector contains NaN/Inf")
    return vector


def load_transfer_bundle(path: str | Path) -> dict:
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, dict):
        raise ValueError("research transfer artifact must be a dictionary bundle")
    return bundle


def _validate_bundle(bundle: Mapping[str, object]) -> tuple[object, tuple[str, ...], np.ndarray, float, float]:
    if str(bundle.get("status")) != "ACTIVE_RESEARCH_TRANSFER":
        raise ValueError("research transfer artifact is not ACTIVE_RESEARCH_TRANSFER")
    if str(bundle.get("clinical_claim", "NONE")) != "NONE":
        raise ValueError("research transfer artifact must keep clinical_claim=NONE")

    model = bundle.get("model")
    if model is None or not hasattr(model, "predict_proba"):
        raise ValueError("research transfer artifact is missing a probability-producing model")

    names = tuple(str(name) for name in bundle.get("feature_names", ()))
    if not names:
        raise ValueError("research transfer artifact has no feature contract")

    source = np.asarray(bundle.get("ood_source_standardized"), dtype=np.float64)
    if source.ndim != 2 or source.shape[0] < 3 or source.shape[1] != len(names):
        raise ValueError("research transfer artifact has invalid OOD source features")
    if not np.isfinite(source).all():
        raise ValueError("research transfer OOD source contains NaN/Inf")

    threshold = float(bundle.get("ood_threshold"))
    decision_threshold = float(bundle.get("decision_threshold", 0.5))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("research transfer artifact has invalid OOD threshold")
    if not 0.0 < decision_threshold < 1.0:
        raise ValueError("research transfer artifact has invalid decision threshold")
    return model, names, source, threshold, decision_threshold


def infer_research_transfer(
    features: Mapping[str, float | int],
    bundle: Mapping[str, object] | None,
    *,
    measurement_status: str,
) -> ResearchTransferResult:
    """Return a research-only human transfer decision with explicit abstention.

    The function never falls back to MumGuard's fused measurement score as a
    disease probability. Without an active source-trained artifact, sufficient
    measurement support, or in-domain feature support, it returns INCONCLUSIVE.
    """

    measurement_status = str(measurement_status or "UNKNOWN")
    if measurement_status != "OK":
        return ResearchTransferResult(
            research_decision="INCONCLUSIVE",
            research_concern_score=None,
            domain_status="MEASUREMENT_INSUFFICIENT",
            measurement_status=measurement_status,
            reason="Measurement support is insufficient for research transfer inference.",
            model_id=None,
            model_version=None,
            source_domain=None,
            nearest_source_distance=None,
            ood_threshold=None,
            feature_names=tuple(),
        )

    if bundle is None:
        return ResearchTransferResult(
            research_decision="INCONCLUSIVE",
            research_concern_score=None,
            domain_status="TRANSFER_MODEL_UNAVAILABLE",
            measurement_status=measurement_status,
            reason="No active human-source research transfer artifact is loaded.",
            model_id=None,
            model_version=None,
            source_domain=None,
            nearest_source_distance=None,
            ood_threshold=None,
            feature_names=tuple(),
        )

    try:
        model, names, source, ood_threshold, decision_threshold = _validate_bundle(bundle)
        vector = feature_vector(features, names)
        scaler = model.named_steps.get("scale") if hasattr(model, "named_steps") else None
        if scaler is None or not hasattr(scaler, "transform"):
            raise ValueError("research transfer model must expose a fitted 'scale' step")
        standardized = np.asarray(scaler.transform(vector.reshape(1, -1)), dtype=np.float64)[0]
        distances = np.linalg.norm(source - standardized[None, :], axis=1)
        nearest = float(np.min(distances))
    except (TypeError, ValueError) as exc:
        return ResearchTransferResult(
            research_decision="INCONCLUSIVE",
            research_concern_score=None,
            domain_status="TRANSFER_ARTIFACT_INVALID",
            measurement_status=measurement_status,
            reason=str(exc),
            model_id=str(bundle.get("model_id")) if bundle.get("model_id") else None,
            model_version=str(bundle.get("model_version")) if bundle.get("model_version") else None,
            source_domain=str(bundle.get("source_domain")) if bundle.get("source_domain") else None,
            nearest_source_distance=None,
            ood_threshold=None,
            feature_names=tuple(str(name) for name in bundle.get("feature_names", ())),
        )

    if nearest > ood_threshold:
        return ResearchTransferResult(
            research_decision="INCONCLUSIVE",
            research_concern_score=None,
            domain_status="ABSTAIN_OOD",
            measurement_status=measurement_status,
            reason="MumGuard thermal-shape features are outside the source-domain support gate.",
            model_id=str(bundle.get("model_id")) if bundle.get("model_id") else None,
            model_version=str(bundle.get("model_version")) if bundle.get("model_version") else None,
            source_domain=str(bundle.get("source_domain")) if bundle.get("source_domain") else None,
            nearest_source_distance=nearest,
            ood_threshold=ood_threshold,
            feature_names=names,
        )

    probability = float(model.predict_proba(vector.reshape(1, -1))[0, 1])
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return ResearchTransferResult(
            research_decision="INCONCLUSIVE",
            research_concern_score=None,
            domain_status="MODEL_OUTPUT_INVALID",
            measurement_status=measurement_status,
            reason="Human-source transfer model returned an invalid score.",
            model_id=str(bundle.get("model_id")) if bundle.get("model_id") else None,
            model_version=str(bundle.get("model_version")) if bundle.get("model_version") else None,
            source_domain=str(bundle.get("source_domain")) if bundle.get("source_domain") else None,
            nearest_source_distance=nearest,
            ood_threshold=ood_threshold,
            feature_names=names,
        )

    decision = (
        "SUSPICIOUS_RESEARCH"
        if probability >= decision_threshold
        else "NOT_SUSPICIOUS_RESEARCH"
    )
    return ResearchTransferResult(
        research_decision=decision,
        research_concern_score=round(probability * 100.0, 1),
        domain_status="IN_DOMAIN",
        measurement_status=measurement_status,
        reason=(
            "Research-only source-domain transfer decision passed the OOD gate. "
            "The concern index is not a clinical cancer probability."
        ),
        model_id=str(bundle.get("model_id")) if bundle.get("model_id") else None,
        model_version=str(bundle.get("model_version")) if bundle.get("model_version") else None,
        source_domain=str(bundle.get("source_domain")) if bundle.get("source_domain") else None,
        nearest_source_distance=nearest,
        ood_threshold=ood_threshold,
        feature_names=names,
    )
