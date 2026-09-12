import json

import pytest
from fastapi import HTTPException

from app.main import _default_file_metadata, parse_metadata


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
