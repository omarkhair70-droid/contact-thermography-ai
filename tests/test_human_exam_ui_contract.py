from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "app" / "static" / "human-exam.html"


def test_human_exam_ui_exists_and_uses_bilateral_contract():
    text = UI.read_text(encoding="utf-8")
    assert "MumGuard Human Examination" in text
    assert "LEFT breast captures" in text
    assert "RIGHT breast captures" in text
    assert "species:'human'" in text
    assert "acquisition_type:'contact-LCT'" in text
    assert "POST" not in text or "/api/exams/analyze" in text
    assert "/api/exams/analyze" in text


def test_human_exam_ui_renders_session_and_decision_outputs():
    text = UI.read_text(encoding="utf-8")
    assert "mumguard_session_evidence" in text
    assert "human_decision" in text
    assert "core_hyperthermia_score" in text
    assert "bilateral_asymmetry_score" in text
    assert "abnormal_skin_behavior_score" in text
    assert "overall_measurement_evidence_score" in text
    assert "INCONCLUSIVE" in text
    assert "NOT_CALIBRATED" in text
    assert "INDICATION_" in text


def test_human_exam_ui_does_not_present_measurement_evidence_as_cancer_probability():
    text = UI.read_text(encoding="utf-8")
    assert "percentage cancer probability is not manufactured" in text
    assert "clinical claim: NONE" in text
