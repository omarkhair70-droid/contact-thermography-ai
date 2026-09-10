from __future__ import annotations

import json
import re

import pandas as pd

from app import main
from app.services.exam_contracts import DEFAULT_TLC_PROFILE


def _upload(client, sample_png, *, exam_id="qa-exam", metadata=None, second=False):
    files = [("files", ("left.png", sample_png, "image/png"))]
    if second:
        files.append(("files", ("right.png", sample_png, "image/png")))
    data = {"exam_id": exam_id}
    if metadata is not None:
        data["metadata_json"] = json.dumps(metadata)
    return client.post("/api/exams/analyze", files=files, data=data)


def _assert_no_current_clinical_claim(payload):
    forbidden_value_keys = {
        "cancer_probability",
        "cancer_risk",
        "clinical_probability",
        "diagnosis",
        "diagnostic_result",
        "diagnostic_probability",
    }

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                lowered = key.lower()
                if lowered == "clinical_claim":
                    assert item == "NONE"
                if lowered == "clinical_risk":
                    assert item is None
                if lowered in forbidden_value_keys:
                    assert item in (None, False, "", "NONE")
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    text = json.dumps(payload, allow_nan=False).lower()
    assert re.search(r"cancer probability\s*[:=]\s*(?:\d|0\.)", text) is None
    assert re.search(r"diagnos(?:is|tic result)\s*[:=]\s*[^\s\"']+", text) is None


def test_health_endpoint_and_nonclinical_boundary(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["clinical_claim"] == "NONE"
    _assert_no_current_clinical_claim(body)


def test_upload_history_detail_and_report_regression(client, sample_png):
    metadata = [{
        "filename": "left.png",
        "side": "LEFT",
        "position": "P1",
        "tlc_profile_id": "client-device-tlc-pending",
        "device_profile_id": "client-device-qa",
    }]
    response = _upload(client, sample_png, exam_id="qa-history", metadata=metadata)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["exam_id"] == "qa-history"
    assert body["tlc_profile_id"] == "client-device-tlc-pending"
    assert body["profile_provenance"]["domain_status"] == "SEPARATE_TLC_DOMAIN"
    assert body["profile_provenance"]["matches_reference_colour_domain"] is False
    assert body["profile_provenance"]["absolute_temperature_interpretation"].startswith("DISABLED")
    assert body["clinical_claim"] == "NONE"
    assert body["sources"][0]["plates"][0]["tlc_profile_id"] == "client-device-tlc-pending"
    assert body["sources"][0]["plates"][0]["device_profile_id"] == "client-device-qa"
    _assert_no_current_clinical_claim(body)

    history = client.get("/api/exams")
    assert history.status_code == 200
    assert history.json()["count"] == 1
    assert history.json()["items"][0]["exam_id"] == "qa-history"
    _assert_no_current_clinical_claim(history.json())

    detail = client.get("/api/exams/qa-history")
    assert detail.status_code == 200
    saved = detail.json()["result"]
    assert saved["tlc_profile_id"] == "client-device-tlc-pending"
    assert saved["profile_provenance"]["device_profile_ids"] == ["client-device-qa"]
    _assert_no_current_clinical_claim(detail.json())

    report = client.get("/reports/qa-history")
    assert report.status_code == 200
    assert "client-device-tlc-pending" in report.text
    assert "client-device-qa" in report.text
    assert "Clinical claim:</b> NONE" in report.text
    assert "Absolute temperature interpretation remains disabled" in report.text
    assert re.search(r"cancer probability\s*[:=]\s*(?:\d|0\.)", report.text, re.I) is None
    assert re.search(r"diagnosis\s*[:=]", report.text, re.I) is None


def test_default_tlc_profile_is_explicit(client, sample_png):
    response = _upload(client, sample_png, exam_id="qa-default")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["tlc_profile_id"] == DEFAULT_TLC_PROFILE
    assert body["sources"][0]["tlc_profile_id"] == DEFAULT_TLC_PROFILE
    assert body["sources"][0]["plates"][0]["tlc_profile_id"] == DEFAULT_TLC_PROFILE


def test_reference_profile_provenance_is_not_client_domain(client, sample_png):
    metadata = [{
        "filename": "left.png",
        "side": "LEFT",
        "position": "P1",
        "tlc_profile_id": "reference-publication-unknown",
    }]
    response = _upload(client, sample_png, exam_id="qa-reference", metadata=metadata)
    assert response.status_code == 200, response.text
    provenance = response.json()["profile_provenance"]
    assert provenance["domain_status"] == "REFERENCE_PUBLICATION_DOMAIN"
    assert provenance["matches_reference_colour_domain"] is True
    assert provenance["colour_calibration_status"] == "UNKNOWN_REFERENCE_CALIBRATION"


def test_mixed_tlc_profiles_are_rejected_instead_of_silently_combined(client, sample_png):
    metadata = [
        {"filename":"left.png","side":"LEFT","position":"P1","tlc_profile_id":"reference-publication-unknown"},
        {"filename":"right.png","side":"RIGHT","position":"P1","tlc_profile_id":"client-device-tlc-pending"},
    ]
    response = _upload(client, sample_png, exam_id="qa-mixed", metadata=metadata, second=True)
    assert response.status_code == 400
    assert "Mixed TLC profiles are not allowed" in response.json()["detail"]
    assert client.get("/api/exams").json()["count"] == 0


def test_left_right_pairing_uses_position_and_device_provenance(client, sample_png):
    metadata = [
        {"filename":"left.png","side":"L","position":"P1","tlc_profile_id":"client-device-tlc-pending","device_profile_id":"device-a"},
        {"filename":"right.png","side":"right","position":"P1","tlc_profile_id":"client-device-tlc-pending","device_profile_id":"device-a"},
    ]
    response = _upload(client, sample_png, exam_id="qa-pair", metadata=metadata, second=True)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["bilateral_pairs_created"] == 1
    pair = body["bilateral_analysis"][0]
    assert pair["position"] == "P1"
    assert pair["tlc_profile_id"] == "client-device-tlc-pending"
    assert pair["device_profile_id"] == "device-a"
    assert pair["clinical_claim"] == "NONE"


def test_left_right_device_mismatch_is_not_paired(client, sample_png):
    metadata = [
        {"filename":"left.png","side":"LEFT","position":"P1","tlc_profile_id":"client-device-tlc-pending","device_profile_id":"device-a"},
        {"filename":"right.png","side":"RIGHT","position":"P1","tlc_profile_id":"client-device-tlc-pending","device_profile_id":"device-b"},
    ]
    response = _upload(client, sample_png, exam_id="qa-device-mismatch", metadata=metadata, second=True)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["bilateral_pairs_created"] == 0
    assert body["bilateral_pairing_warnings"][0]["position"] == "P1"


def test_dinov2_endpoint_is_json_safe_and_nonclinical(client, monkeypatch):
    monkeypatch.setattr(main.store, "dino_index", pd.DataFrame([
        {"plate_id":"A","score":float("nan"),"upper":float("inf"),"lower":float("-inf")},
    ]))
    monkeypatch.setattr(main.store, "dino_pairs", pd.DataFrame([
        {"bilateral_pair_id":"B","score":float("inf")},
    ]))
    response = client.get("/api/reference/dinov2")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["plates"][0]["score"] is None
    assert body["plates"][0]["upper"] is None
    assert body["plates"][0]["lower"] is None
    assert body["bilateral_pairs"][0]["score"] is None
    _assert_no_current_clinical_claim(body)


def test_reference_endpoints_do_not_surface_current_diagnostic_claims(client):
    for path in ["/api/model", "/api/reference/plates", "/api/reference/pairs", "/api/reference/dinov2", "/api/exams"]:
        response = client.get(path)
        assert response.status_code == 200, (path, response.text)
        _assert_no_current_clinical_claim(response.json())
