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
    assert "/api/exams/analyze" in text


def test_human_exam_ui_records_environment_and_preparation_context():
    text = UI.read_text(encoding="utf-8")
    for token in (
        "room_temperature_c",
        "room_humidity_percent",
        "acclimatization_minutes",
        "lighting_profile_id",
        "camera_profile_id",
        "tlc_batch_id",
        "protocol_revision",
        "preparation_flags",
        "capture_role",
    ):
        assert token in text
    assert "SPATIAL_TILE" in text
    assert "TEMPORAL_FRAME" in text


def test_human_exam_ui_renders_session_decision_measurement_ai_and_maps():
    text = UI.read_text(encoding="utf-8")
    assert "mumguard_session_evidence" in text
    assert "human_decision" in text
    assert "core_hyperthermia_score" in text
    assert "bilateral_asymmetry_score" in text
    assert "abnormal_skin_behavior_score" in text
    assert "overall_measurement_evidence_score" in text
    assert "Measurement support" in text
    assert "Local AI / DINO evidence" in text
    assert "ai_evidence_available" in text
    assert "preview_filenames" in text
    assert "renderPreviews" in text
    assert "LEFT thermal evidence" in text
    assert "Bilateral asymmetry" in text
    assert "INCONCLUSIVE" in text
    assert "NOT_CALIBRATED" in text
    assert "INDICATION_" in text


def test_human_exam_ui_makes_upload_names_unique_and_escapes_display_names():
    text = UI.read_text(encoding="utf-8")
    assert "const uploadName=" in text
    assert "fd.append('files',item.file,name)" in text
    assert "filename:name" in text
    assert "const esc=" in text
    assert "${esc(x.file.name)}" in text


def test_human_exam_ui_does_not_present_measurement_evidence_as_cancer_probability():
    text = UI.read_text(encoding="utf-8")
    assert "percentage cancer probability is not manufactured" in text
    assert "clinical claim: NONE" in text
