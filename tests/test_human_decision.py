from app.services.human_decision import HumanDecisionInput, build_human_decision


def test_inconclusive_measurement_never_emits_risk():
    result = build_human_decision(
        HumanDecisionInput(
            measurement_status="INCONCLUSIVE_MEASUREMENT_SUPPORT",
            measurement_scores={"overall_measurement_evidence_score": 0.91},
            ai_evidence_available=True,
        )
    )
    assert result.decision_status == "INCONCLUSIVE"
    assert result.indication is None
    assert result.risk_score is None
    assert result.calibrated is False
    assert result.clinical_claim == "NONE"
    assert result.as_dict()["status"] == "INCONCLUSIVE"
    assert result.as_dict()["decision_status"] == "INCONCLUSIVE"


def test_measurement_score_is_not_relabelled_as_risk_when_uncalibrated():
    result = build_human_decision(
        HumanDecisionInput(
            measurement_status="OK",
            measurement_scores={"overall_measurement_evidence_score": 0.88},
            ai_evidence_available=True,
        )
    )
    assert result.decision_status == "NOT_CALIBRATED"
    assert result.risk_score is None
    assert result.indication is None
    assert result.as_dict()["status"] == "NOT_CALIBRATED"


def test_calibrated_human_score_can_emit_low_intermediate_high():
    calibration = {
        "status": "CALIBRATED",
        "tau_low": 0.25,
        "tau_high": 0.70,
        "model_id": "human-head-v1",
        "calibration_id": "human-calibration-v1",
    }

    low = build_human_decision(
        HumanDecisionInput("OK", {}, True, model_score=0.10, calibration=calibration)
    )
    mid = build_human_decision(
        HumanDecisionInput("OK", {}, True, model_score=0.50, calibration=calibration)
    )
    high = build_human_decision(
        HumanDecisionInput("OK", {}, True, model_score=0.82, calibration=calibration)
    )

    assert low.decision_status == "INDICATION_LOW"
    assert low.indication == "LOW"
    assert mid.decision_status == "INDICATION_INTERMEDIATE"
    assert mid.indication == "INTERMEDIATE"
    assert high.decision_status == "INDICATION_HIGH"
    assert high.indication == "HIGH"
    assert high.risk_score == 0.82
    assert high.model_id == "human-head-v1"
    assert high.calibration_id == "human-calibration-v1"
    assert high.as_dict()["status"] == "INDICATION_HIGH"


def test_invalid_calibration_fails_closed():
    result = build_human_decision(
        HumanDecisionInput(
            measurement_status="OK",
            measurement_scores={},
            ai_evidence_available=False,
            model_score=0.95,
            calibration={"status": "CALIBRATED", "tau_low": 0.8, "tau_high": 0.2},
        )
    )
    assert result.decision_status == "NOT_CALIBRATED"
    assert result.risk_score is None
