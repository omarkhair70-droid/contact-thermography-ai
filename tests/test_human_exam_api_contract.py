import json

import pytest
from fastapi import HTTPException

from app import main
from app.main import (
    _default_file_metadata,
    _eligible_mumguard_session_frame,
    parse_metadata,
)


def test_parse_metadata_carries_capture_context_for_human_session():
    payload = json.dumps([
        {
            "filename": "left-01.jpg",
            "side": "LEFT",
            "sequence_index": 1,
            "species": "human",
            "acquisition_type": "contact-LCT",
            "capture_role": "SPATIAL_TILE",
            "room_temperature_c": 25.0,
            "room_humidity_percent": 55,
            "acclimatization_minutes": 15,
            "lighting_profile_id": "light-v1",
            "camera_profile_id": "camera-v1",
            "tlc_batch_id": "tlc-batch-a",
        }
    ])
    item = parse_metadata(payload)["left-01.jpg"]
    assert item["side"] == "LEFT"
    assert item["species"] == "human"
    assert item["acquisition_type"] == "contact-LCT"
    context = item["acquisition_context"]
    assert context["capture_role"] == "SPATIAL_TILE"
    assert context["room_temperature_c"] == 25.0
    assert context["acclimatization_minutes"] == 15.0
    assert context["tlc_batch_id"] == "tlc-batch-a"


def test_parse_metadata_rejects_ambiguous_duplicate_filename_rows():
    payload = json.dumps([
        {"filename": "capture.jpg", "side": "LEFT"},
        {"filename": "capture.jpg", "side": "RIGHT"},
    ])
    with pytest.raises(HTTPException) as exc:
        parse_metadata(payload)
    assert exc.value.status_code == 400
    assert "Duplicate metadata filename" in str(exc.value.detail)


def test_parse_metadata_rejects_invalid_environment_context():
    payload = json.dumps([
        {"filename": "capture.jpg", "side": "LEFT", "room_humidity_percent": 120},
    ])
    with pytest.raises(HTTPException) as exc:
        parse_metadata(payload)
    assert exc.value.status_code == 400
    assert "room_humidity_percent" in str(exc.value.detail)


def test_default_metadata_keeps_unknown_acquisition_context_explicit():
    item = _default_file_metadata()
    assert item["acquisition_context"]["capture_role"] == "UNKNOWN"
    assert item["acquisition_context"]["room_temperature_c"] is None


def test_human_session_eligibility_is_not_bound_to_pending_profile_name():
    base = {"side": "LEFT", "species": "human", "acquisition_type": "contact-LCT"}
    assert _eligible_mumguard_session_frame("client-device-tlc-pending", base)
    assert _eligible_mumguard_session_frame("mumguard-calibrated-v2", base)


def test_human_session_eligibility_rejects_reference_mouse_and_ir_domains():
    assert not _eligible_mumguard_session_frame(
        "reference-publication-unknown",
        {"side": "LEFT", "species": "human", "acquisition_type": "contact-LCT"},
    )
    assert not _eligible_mumguard_session_frame(
        "client-device-tlc-pending",
        {"side": "LEFT", "species": "mouse", "acquisition_type": "contact-LCT"},
    )
    assert not _eligible_mumguard_session_frame(
        "client-device-tlc-pending",
        {"side": "LEFT", "species": "human", "acquisition_type": "radiometric-IR"},
    )


def test_human_exam_route_and_health_flags_are_live(client):
    page = client.get("/human-exam")
    assert page.status_code == 200
    assert "MumGuard Human Examination" in page.text
    health = client.get("/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["human_exam_ui"] is True
    assert payload["human_decision_contract"] is True
    assert payload["mumguard_session_fusion"] is True
    assert payload["human_runtime_direct"] is True


def test_direct_human_exam_4x4_skips_legacy_reference_pipeline(client, sample_png, monkeypatch):
    def forbidden_legacy(*args, **kwargs):
        raise AssertionError("legacy per-image analyzer must not run for direct human exams")

    monkeypatch.setattr(main, "analyze_uploaded_image", forbidden_legacy)

    files = []
    metadata = []
    for side in ("LEFT", "RIGHT"):
        for index in range(1, 5):
            name = f"{side.lower()}-{index:02d}.png"
            files.append(("files", (name, sample_png, "image/png")))
            metadata.append({
                "filename": name,
                "side": side,
                "sequence_index": index,
                "species": "human",
                "acquisition_type": "contact-LCT",
                "capture_role": "SPATIAL_TILE",
                "tlc_profile_id": "client-device-tlc-pending",
                "device_profile_id": "device-test",
            })

    response = client.post(
        "/api/human-exams/analyze",
        files=files,
        data={
            "exam_id": "human-4x4-direct",
            "metadata_json": json.dumps(metadata),
            "tlc_profile_id": "client-device-tlc-pending",
            "device_profile_id": "device-test",
            "include_dino": "false",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["human_runtime"] == "DIRECT_SESSION_ONLY"
    assert body["legacy_reference_analysis_executed"] is False
    assert body["source_images"] == 8
    assert body["plates_detected"] == 0
    assert body["bilateral_pairs_created"] == 0
    assert body["sources"] == []
    assert body["bilateral_analysis"] == []
    assert body["mumguard_session_evidence"] is not None
    assert len(body["mumguard_session_evidence"]["frames"]) == 8
    assert body["model_status"] == "MUMGUARD_SESSION_EVIDENCE_RESEARCH"
    assert body["clinical_risk"] is None
    assert body["clinical_claim"] == "NONE"

    history = client.get("/api/exams")
    assert history.status_code == 200
    assert history.json()["items"][0]["plates_detected"] == 0

    report = client.get("/reports/human-4x4-direct")
    assert report.status_code == 200
    assert "MumGuard session evidence" in report.text
