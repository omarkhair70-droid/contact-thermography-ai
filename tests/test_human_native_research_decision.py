from app.services.human_native_research_decision import build_native_research_decision


def _scores(*, core, bilateral, pattern, left_core=None, right_core=None):
    return {
        "core_hyperthermia_score": core,
        "bilateral_asymmetry_score": bilateral,
        "abnormal_skin_behavior_score": pattern,
        "left_core_hyperthermia_score": core if left_core is None else left_core,
        "right_core_hyperthermia_score": core if right_core is None else right_core,
    }


def test_measurement_insufficient_remains_inconclusive():
    result = build_native_research_decision(
        measurement_status="INCONCLUSIVE_MEASUREMENT_SUPPORT",
        measurement_scores={},
        transfer_reference={"status": "ABSTAIN_OOD"},
    )
    assert result["status"] == "INCONCLUSIVE"
    assert result["research_concern_score"] is None
    assert result["clinical_risk"] is None
    assert result["clinical_claim"] == "NONE"


def test_low_symmetric_native_evidence_is_not_suspicious_even_when_transfer_is_ood():
    result = build_native_research_decision(
        measurement_status="OK",
        measurement_scores=_scores(core=0.42, bilateral=0.18, pattern=0.40, left_core=0.42, right_core=0.40),
        dino_bilateral_score=0.20,
        transfer_reference={
            "status": "ABSTAIN_OOD",
            "decision_model_executed": False,
            "source_transfer_score_0_1": None,
        },
    )
    assert result["status"] == "NOT_SUSPICIOUS_RESEARCH"
    assert result["transfer_reference_status"] == "ABSTAIN_OOD"
    assert result["transfer_used"] is False
    assert result["research_concern_score"] is not None
    assert result["clinical_claim"] == "NONE"


def test_concordant_high_native_evidence_is_suspicious_without_transfer():
    result = build_native_research_decision(
        measurement_status="OK",
        measurement_scores=_scores(core=0.82, bilateral=0.78, pattern=0.74, left_core=0.84, right_core=0.25),
        dino_bilateral_score=0.72,
        transfer_reference={"status": "ABSTAIN_OOD", "decision_model_executed": False},
    )
    assert result["status"] == "SUSPICIOUS_RESEARCH"
    assert result["research_concern_score"] >= 64.0
    assert result["decision_origin"] == "MUMGUARD_NATIVE_RESEARCH_V0"


def test_gray_band_abstains_instead_of_forcing_binary_result():
    result = build_native_research_decision(
        measurement_status="OK",
        measurement_scores=_scores(core=0.60, bilateral=0.48, pattern=0.60, left_core=0.65, right_core=0.45),
        dino_bilateral_score=0.50,
        transfer_reference={"status": "ABSTAIN_OOD", "decision_model_executed": False},
    )
    assert result["status"] == "INCONCLUSIVE"
    assert result["research_concern_score"] is not None


def test_in_domain_human_transfer_is_auxiliary_not_the_primary_gate():
    native_only = build_native_research_decision(
        measurement_status="OK",
        measurement_scores=_scores(core=0.58, bilateral=0.44, pattern=0.55, left_core=0.62, right_core=0.42),
        transfer_reference={"status": "ABSTAIN_OOD", "decision_model_executed": False},
    )
    with_transfer = build_native_research_decision(
        measurement_status="OK",
        measurement_scores=_scores(core=0.58, bilateral=0.44, pattern=0.55, left_core=0.62, right_core=0.42),
        transfer_reference={
            "status": "IN_SOURCE_SUPPORT",
            "decision_model_executed": True,
            "source_transfer_score_0_1": 0.90,
        },
    )
    assert with_transfer["transfer_used"] is True
    assert with_transfer["research_concern_score"] > native_only["research_concern_score"]
    assert with_transfer["decision_origin"] == "MUMGUARD_NATIVE_RESEARCH_V0"
    assert with_transfer["clinical_risk"] is None


def test_research_score_is_explicitly_not_cancer_probability():
    result = build_native_research_decision(
        measurement_status="OK",
        measurement_scores=_scores(core=0.70, bilateral=0.65, pattern=0.68, left_core=0.75, right_core=0.30),
    )
    assert "not a cancer probability" in result["score_semantics"]
    assert result["calibration_status"] == "ENGINEERING_RESEARCH_THRESHOLDS_NOT_CLINICALLY_CALIBRATED"
    assert result["clinical_claim"] == "NONE"
