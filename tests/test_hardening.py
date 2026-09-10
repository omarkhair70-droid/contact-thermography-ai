from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

from app.services.analysis_engine import analyze_uploaded_image, detect_circular_plates
from app.services.json_safety import sanitize_for_json


def test_missing_upload_field_is_rejected(client):
    response = client.post("/api/exams/analyze", data={"exam_id":"no-file"})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "filename,payload",
    [
        ("empty.png", b""),
        ("not-image.txt", b"this is not an image"),
        ("broken.png", b"\x89PNG\r\n\x1a\ntruncated"),
    ],
)
def test_corrupt_empty_and_non_image_inputs_return_400(client, filename, payload):
    response = client.post(
        "/api/exams/analyze",
        files=[("files", (filename, payload, "application/octet-stream"))],
        data={"exam_id":"bad-input"},
    )
    assert response.status_code == 400
    assert response.json()["detail"]


def test_invalid_metadata_json_returns_400_before_analysis(client, sample_png):
    response = client.post(
        "/api/exams/analyze",
        files=[("files", ("left.png", sample_png, "image/png"))],
        data={"exam_id":"bad-meta","metadata_json":"{not-json"},
    )
    assert response.status_code == 400
    assert "Invalid metadata_json" in response.json()["detail"]


def test_metadata_files_member_must_be_list(client, sample_png):
    response = client.post(
        "/api/exams/analyze",
        files=[("files", ("left.png", sample_png, "image/png"))],
        data={"exam_id":"bad-meta-files","metadata_json":json.dumps({"files":{"filename":"left.png"}})},
    )
    assert response.status_code == 400
    assert "files must be a list" in response.json()["detail"]


def test_json_sanitizer_removes_nan_and_infinity_recursively():
    raw = {
        "nan": float("nan"),
        "pos": float("inf"),
        "neg": float("-inf"),
        "nested": [np.float64("nan"), {"ok": np.float32(1.25)}],
    }
    safe = sanitize_for_json(raw)
    assert safe["nan"] is None
    assert safe["pos"] is None
    assert safe["neg"] is None
    assert safe["nested"][0] is None
    assert math.isclose(safe["nested"][1]["ok"], 1.25, rel_tol=1e-6)
    json.dumps(safe, allow_nan=False)


def test_multi_plate_extraction_regression(tmp_path, monkeypatch):
    from app.services import analysis_engine

    image = np.full((360, 760, 3), 255, dtype=np.uint8)
    cv2.circle(image, (180, 180), 110, (10, 10, 10), -1)
    cv2.circle(image, (580, 180), 110, (10, 10, 10), -1)

    detected = detect_circular_plates(image)
    assert len(detected) >= 2

    ok, encoded = cv2.imencode(".png", image)
    assert ok
    monkeypatch.setattr(analysis_engine, "STATIC_GENERATED", tmp_path / "generated")
    result = analyze_uploaded_image(encoded.tobytes(), "two-plates.png", "multi-plate")
    assert result["extraction_mode"] == "detected_circles"
    assert result["plates_detected"] >= 2
    assert result["plates"][0]["plate_id"].endswith("P01")
    assert result["plates"][1]["plate_id"].endswith("P02")
