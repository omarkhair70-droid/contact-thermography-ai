from __future__ import annotations

import math
from typing import Mapping


DECISION_VERSION = "mumguard-native-research-v0"
NOT_SUSPICIOUS_MAX = 0.52
SUSPICIOUS_MIN = 0.64

# Engineering research fusion only. These weights/thresholds are not clinically
# calibrated and must not be presented as sensitivity/specificity or cancer risk.
_NATIVE_WEIGHTS = {
    "core_hyperthermia": 0.30,
    "abnormal_distribution": 0.25,
    "bilateral_asymmetry": 0.30,
    "core_side_gap": 0.15,
}
DINO_WEIGHT = 0.10
TRANSFER_WEIGHT = 0.15


def _unit(value: object, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 0..1 research-evidence score") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite 0..1 research-evidence score")
    return number


def _optional_unit(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _inconclusive(measurement_status: str, transfer_status: str | None, reason: str) -> dict:
    return {
        "version": DECISION_VERSION,
        "status": "INCONCLUSIVE",
        "research_concern_score": None,
        "native_score_0_1": None,
        "decision_origin": "MUMGUARD_NATIVE_RESEARCH_V0",
        "measurement_status": measurement_status,
        "transfer_reference_status": transfer_status,
        "transfer_used": False,
        "dino_bilateral_used": False,
        "evidence_components": {},
        "decision_bands": {
            "not_suspicious_below": NOT_SUSPICIOUS_MAX,
            "suspicious_at_or_above": SUSPICIOUS_MIN,
        },
        "calibration_status": "ENGINEERING_RESEARCH_THRESHOLDS_NOT_CLINICALLY_CALIBRATED",
        "reason": reason,
        "score_semantics": "research concern only; not a cancer probability, diagnosis, sensitivity, or specificity",
        "clinical_risk": None,
        "clinical_claim": "NONE",
    }


def build_native_research_decision(
    *,
    measurement_status: str,
    measurement_scores: Mapping[str, object],
    dino_bilateral_score: float | None = None,
    transfer_reference: Mapping[str, object] | None = None,
) -> dict:
    """Build the primary human research decision from MumGuard-native evidence.

    The primary decision is intentionally independent of the auxiliary DMR-IR
    source-domain gate. If the transfer lane is out-of-domain it is reported but
    does not force the whole MumGuard examination to INCONCLUSIVE. The transfer
    score can contribute a small auxiliary weight only when its source-support
    gate passed and its research head actually executed.

    This is a transparent engineering evidence-fusion rule, not a trained human
    disease model and not a clinically calibrated cancer probability.
    """

    status = str(measurement_status or "UNKNOWN")
    transfer_status = (
        str(transfer_reference.get("status"))
        if isinstance(transfer_reference, Mapping) and transfer_reference.get("status")
        else None
    )
    if status != "OK":
        return _inconclusive(
            status,
            transfer_status,
            "MumGuard measurement support is insufficient; the native research layer abstains.",
        )

    try:
        core = _unit(measurement_scores.get("core_hyperthermia_score"), name="core_hyperthermia_score")
        bilateral = _unit(measurement_scores.get("bilateral_asymmetry_score"), name="bilateral_asymmetry_score")
        pattern = _unit(measurement_scores.get("abnormal_skin_behavior_score"), name="abnormal_skin_behavior_score")
        left_core = _unit(
            measurement_scores.get("left_core_hyperthermia_score", core),
            name="left_core_hyperthermia_score",
        )
        right_core = _unit(
            measurement_scores.get("right_core_hyperthermia_score", core),
            name="right_core_hyperthermia_score",
        )
    except ValueError as exc:
        return _inconclusive(status, transfer_status, str(exc))

    core_side_gap = abs(left_core - right_core)
    native_thermal = (
        _NATIVE_WEIGHTS["core_hyperthermia"] * core
        + _NATIVE_WEIGHTS["abnormal_distribution"] * pattern
        + _NATIVE_WEIGHTS["bilateral_asymmetry"] * bilateral
        + _NATIVE_WEIGHTS["core_side_gap"] * core_side_gap
    )

    dino = _optional_unit(dino_bilateral_score)
    if dino is None:
        native_with_dino = native_thermal
        dino_used = False
    else:
        native_with_dino = (1.0 - DINO_WEIGHT) * native_thermal + DINO_WEIGHT * dino
        dino_used = True

    transfer_score = None
    transfer_used = False
    if isinstance(transfer_reference, Mapping):
        candidate = _optional_unit(transfer_reference.get("source_transfer_score_0_1"))
        transfer_eligible = (
            transfer_status == "IN_SOURCE_SUPPORT"
            and bool(transfer_reference.get("decision_model_executed"))
            and candidate is not None
        )
        if transfer_eligible:
            transfer_score = candidate
            transfer_used = True

    if transfer_used and transfer_score is not None:
        final_score = (1.0 - TRANSFER_WEIGHT) * native_with_dino + TRANSFER_WEIGHT * transfer_score
    else:
        final_score = native_with_dino
    final_score = min(1.0, max(0.0, float(final_score)))

    if final_score < NOT_SUSPICIOUS_MAX:
        decision = "NOT_SUSPICIOUS_RESEARCH"
        reason = (
            "MumGuard-native thermal/bilateral evidence stayed below the engineering research concern band."
        )
    elif final_score >= SUSPICIOUS_MIN:
        decision = "SUSPICIOUS_RESEARCH"
        reason = (
            "MumGuard-native thermal/bilateral evidence reached the engineering research concern band."
        )
    else:
        decision = "INCONCLUSIVE"
        reason = (
            "MumGuard-native evidence falls in the engineering gray band; the research layer abstains from a binary label."
        )

    if transfer_status == "ABSTAIN_OOD":
        reason += " Auxiliary DMR-IR transfer was out-of-domain and was not used in the primary decision."
    elif transfer_used:
        reason += " In-domain human-IR transfer contributed a small auxiliary evidence weight."

    return {
        "version": DECISION_VERSION,
        "status": decision,
        "research_concern_score": round(final_score * 100.0, 2),
        "native_score_0_1": round(final_score, 6),
        "native_thermal_score_0_1": round(native_thermal, 6),
        "decision_origin": "MUMGUARD_NATIVE_RESEARCH_V0",
        "measurement_status": status,
        "transfer_reference_status": transfer_status,
        "transfer_used": transfer_used,
        "transfer_score_0_1": round(transfer_score, 6) if transfer_score is not None else None,
        "dino_bilateral_used": dino_used,
        "dino_bilateral_score_0_1": round(dino, 6) if dino is not None else None,
        "evidence_components": {
            "core_hyperthermia_score": round(core, 6),
            "left_core_hyperthermia_score": round(left_core, 6),
            "right_core_hyperthermia_score": round(right_core, 6),
            "core_side_gap": round(core_side_gap, 6),
            "bilateral_asymmetry_score": round(bilateral, 6),
            "abnormal_skin_behavior_score": round(pattern, 6),
        },
        "weights": {
            **_NATIVE_WEIGHTS,
            "dino_bilateral_optional": DINO_WEIGHT,
            "in_domain_human_transfer_optional": TRANSFER_WEIGHT,
        },
        "decision_bands": {
            "not_suspicious_below": NOT_SUSPICIOUS_MAX,
            "suspicious_at_or_above": SUSPICIOUS_MIN,
        },
        "calibration_status": "ENGINEERING_RESEARCH_THRESHOLDS_NOT_CLINICALLY_CALIBRATED",
        "reason": reason,
        "score_semantics": "research concern only; not a cancer probability, diagnosis, sensitivity, or specificity",
        "clinical_risk": None,
        "clinical_claim": "NONE",
    }
