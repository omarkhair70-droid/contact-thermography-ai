from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


DECISION_STATUSES = {
    "NOT_CALIBRATED",
    "INCONCLUSIVE",
    "INDICATION_LOW",
    "INDICATION_INTERMEDIATE",
    "INDICATION_HIGH",
}


@dataclass(frozen=True)
class HumanDecisionInput:
    """Downstream contract for the human MumGuard decision layer.

    Measurement evidence is intentionally separated from the eventual calibrated
    human decision model. ``model_score`` must come from a human outcome-linked
    decision head; it is never substituted with the measurement fusion score.
    """

    measurement_status: str
    measurement_scores: Mapping[str, float | int | None]
    ai_evidence_available: bool
    model_score: float | None = None
    calibration: Mapping[str, object] | None = None


@dataclass(frozen=True)
class HumanDecisionResult:
    decision_status: str
    indication: str | None
    risk_score: float | None
    calibrated: bool
    measurement_status: str
    reason: str
    model_id: str | None
    calibration_id: str | None
    clinical_claim: str = "NONE"

    def as_dict(self) -> dict:
        return {
            "decision_status": self.decision_status,
            "status": self.decision_status,
            "indication": self.indication,
            "risk_score": self.risk_score,
            "calibrated": self.calibrated,
            "measurement_status": self.measurement_status,
            "reason": self.reason,
            "model_id": self.model_id,
            "calibration_id": self.calibration_id,
            "clinical_claim": self.clinical_claim,
        }


def _finite_unit_interval(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= number <= 1.0:
        return None
    return number


def _valid_calibration(calibration: Mapping[str, object] | None) -> tuple[float, float, str | None] | None:
    if not calibration or calibration.get("status") != "CALIBRATED":
        return None
    low = _finite_unit_interval(calibration.get("tau_low"))
    high = _finite_unit_interval(calibration.get("tau_high"))
    if low is None or high is None or not low < high:
        return None
    calibration_id = str(calibration.get("calibration_id")) if calibration.get("calibration_id") else None
    return low, high, calibration_id


def build_human_decision(input_data: HumanDecisionInput) -> HumanDecisionResult:
    """Return the honest session-level decision state.

    Current uncalibrated production/research sessions return ``NOT_CALIBRATED``
    (or ``INCONCLUSIVE`` when measurement support is insufficient). A LOW /
    INTERMEDIATE / HIGH indication is only emitted when a separate human model
    score and an explicitly calibrated threshold artifact are supplied.
    """

    measurement_status = str(input_data.measurement_status or "UNKNOWN")
    if measurement_status != "OK":
        return HumanDecisionResult(
            decision_status="INCONCLUSIVE",
            indication=None,
            risk_score=None,
            calibrated=False,
            measurement_status=measurement_status,
            reason="Measurement support is insufficient for a session-level indication.",
            model_id=None,
            calibration_id=None,
        )

    calibration = _valid_calibration(input_data.calibration)
    score = _finite_unit_interval(input_data.model_score)
    if calibration is None or score is None:
        return HumanDecisionResult(
            decision_status="NOT_CALIBRATED",
            indication=None,
            risk_score=None,
            calibrated=False,
            measurement_status=measurement_status,
            reason=(
                "MumGuard session evidence is available, but no validated human "
                "outcome-linked clinical decision score/calibration is active."
            ),
            model_id=None,
            calibration_id=None,
        )

    low, high, calibration_id = calibration
    model_id = None
    if input_data.calibration and input_data.calibration.get("model_id"):
        model_id = str(input_data.calibration.get("model_id"))

    if score < low:
        status = "INDICATION_LOW"
        indication = "LOW"
    elif score >= high:
        status = "INDICATION_HIGH"
        indication = "HIGH"
    else:
        status = "INDICATION_INTERMEDIATE"
        indication = "INTERMEDIATE"

    return HumanDecisionResult(
        decision_status=status,
        indication=indication,
        risk_score=score,
        calibrated=True,
        measurement_status=measurement_status,
        reason="Human outcome-linked decision score interpreted with the active calibration artifact.",
        model_id=model_id,
        calibration_id=calibration_id,
    )
