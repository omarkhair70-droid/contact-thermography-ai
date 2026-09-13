from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "app" / "static" / "human-exam.html"


def test_human_exam_ui_exists_and_uses_bilateral_contract():
    text = UI.read_text(encoding="utf-8")
    assert "MumGuard Care" in text
    assert "LEFT breast" in text
    assert "RIGHT breast" in text
    assert "species:'human'" in text
    assert "acquisition_type:'contact-LCT'" in text
    assert "/api/human-exams/analyze" in text
    assert "/api/exams/analyze" not in text


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


def test_human_exam_ui_keeps_client_result_and_technical_provenance():
    text = UI.read_text(encoding="utf-8")
    assert "mumguard_session_evidence" in text
    assert "human_decision" in text
    assert "core_hyperthermia_score" in text
    assert "bilateral_asymmetry_score" in text
    assert "abnormal_skin_behavior_score" in text
    assert "overall_measurement_evidence_score" in text
    assert "Measurement quality" in text
    assert "Local AI / DINO evidence" in text
    assert "ai_evidence_available" in text
    assert "preview_filenames" in text
    assert "renderPreviews" in text
    assert "LEFT thermal evidence" in text
    assert "LEFT / RIGHT asymmetry" in text
    assert "Technical details" in text
    assert "Decision source" in text
    assert "Human transfer reference" in text
    assert "Clinical calibration" in text


def test_human_exam_ui_presents_plain_language_research_results():
    text = UI.read_text(encoding="utf-8")
    assert "Tumor-like pattern detected" in text
    assert "No tumor-like pattern detected" in text
    assert "Inconclusive" in text
    assert "Research Concern Score" in text
    assert "not a percentage chance of cancer" in text
    assert "Research-use prototype" in text


def test_human_exam_ui_makes_upload_names_unique_and_escapes_display_names():
    text = UI.read_text(encoding="utf-8")
    assert "const uploadName=" in text
    assert "fd.append('files',item.file,name)" in text
    assert "filename:name" in text
    assert "const esc=" in text
    assert "${esc(x.file.name)}" in text


def test_human_exam_ui_requests_direct_runtime_with_dino_support():
    text = UI.read_text(encoding="utf-8")
    assert "fd.append('include_dino','true')" in text
    assert "fetch('/api/human-exams/analyze'" in text
    assert "await r.text()" in text


def test_human_exam_ui_shows_indeterminate_activity_while_request_is_running():
    text = UI.read_text(encoding="utf-8")
    assert 'id="loading"' in text
    assert 'class="loading-track"' in text
    assert 'class="loading-bar"' in text
    assert "startLoading()" in text
    assert "stopLoading()" in text
    assert "Analyzing…" in text
    assert "Uploading, reconstructing the bilateral scan and running thermal + visual AI analysis." in text
