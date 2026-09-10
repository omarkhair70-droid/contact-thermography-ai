from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.services import analysis_engine, db, dinov2_service
from app.services.tlc_profiles import TLCProfile, resolve_tlc_profile


def _png_bytes(bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", bgr)
    assert ok
    return encoded.tobytes()


def _image(colour: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(image, (128, 128), 105, colour, -1)
    cv2.line(image, (60, 80), (190, 170), (255, 255, 255), 4)
    return image


@pytest.fixture
def isolated_runtime(tmp_path, monkeypatch):
    generated = tmp_path / "generated"
    monkeypatch.setattr(analysis_engine, "STATIC_GENERATED", generated)
    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "lct.db")

    calls = []

    def fake_encode(rgb):
        calls.append(np.asarray(rgb).copy())
        offset = float(np.asarray(rgb, dtype=np.float32).mean()) / 10_000.0
        vector = np.linspace(-1.0, 1.0, 384, dtype=np.float32) + offset
        return vector / np.linalg.norm(vector)

    monkeypatch.setattr(dinov2_service.runtime, "encode_rgb", fake_encode)
    return TestClient(main.app), calls


def test_canonical_embedding_artifact_is_original_kaggle_output():
    artifact = Path("artifacts/dinov2_reference/dinov2_embeddings.npy")
    embeddings = np.load(artifact, allow_pickle=False)
    assert artifact.stat().st_size == 40_064
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
        "cad838124969a4a9d06ad122f44d5cdfa9d200c780e437a00ce6c3abf8163ee9"
    )
    assert embeddings.shape == (26, 384)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    assert dinov2_service._FUSED_HEAD.input_dim == 402
    assert dinov2_service._PAIR_HEAD.input_dim == 1536
    assert dinov2_service.DINO_HUB_REPOSITORY == (
        "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    )


def test_ten_plate_composite_runs_live_dino_and_preserves_non_default_profile(
    isolated_runtime, monkeypatch
):
    client, calls = isolated_runtime
    circles = [
        ([50 + (index % 5) * 100, 55 + (index // 5) * 110, 40], None)
        for index in range(10)
    ]
    monkeypatch.setattr(analysis_engine, "detect_circular_plates", lambda image: circles)
    composite = np.full((220, 500, 3), 245, dtype=np.uint8)
    for circle, _ in circles:
        cv2.circle(composite, tuple(circle[:2]), circle[2], (30, 130, 210), -1)

    metadata = [
        {
            "filename": "composite.png",
            "tlc_profile_id": "reference-publication-unknown",
            "device_profile_id": "publication-scanner-a",
        }
    ]
    response = client.post(
        "/api/exams/analyze",
        files=[("files", ("composite.png", _png_bytes(composite), "image/png"))],
        data={"exam_id": "ten-plate", "metadata_json": json.dumps(metadata)},
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["plates_detected"] == 10
    assert len(calls) == 10
    assert result["tlc_profile_ids"] == ["reference-publication-unknown"]
    assert result["device_profile_ids"] == ["publication-scanner-a"]
    assert result["clinical_risk"] is None
    assert result["clinical_claim"] == "NONE"
    for plate in result["sources"][0]["plates"]:
        assert plate["tlc_profile_id"] == "reference-publication-unknown"
        assert plate["device_profile_id"] == "publication-scanner-a"
        assert plate["dinov2_embedding_dim"] == 384
        assert len(plate["dinov2_nearest_reference_plates"]) == 5
        assert 0 <= plate["dinov2_reference_anomaly_score_0_1"] <= 1
        assert 0 <= plate["dinov2_lct_fused_reference_score_0_1"] <= 1
        assert plate["clinical_risk"] is None
        assert plate["clinical_claim"] == "NONE"


def test_true_pair_runs_after_registration_and_survives_persistence_and_report(
    isolated_runtime, monkeypatch
):
    client, calls = isolated_runtime
    monkeypatch.setattr(analysis_engine, "detect_circular_plates", lambda image: [])
    metadata = [
        {
            "filename": "left.png",
            "side": "LEFT",
            "position": "UPPER",
            "tlc_profile_id": "client-device-tlc-pending",
            "device_profile_id": "client-device-01",
        },
        {
            "filename": "right.png",
            "side": "RIGHT",
            "position": "UPPER",
            "tlc_profile_id": "client-device-tlc-pending",
            "device_profile_id": "client-device-01",
        },
    ]
    response = client.post(
        "/api/exams/analyze",
        files=[
            ("files", ("left.png", _png_bytes(_image((30, 110, 210))), "image/png")),
            ("files", ("right.png", _png_bytes(_image((35, 120, 200))), "image/png")),
        ],
        data={"exam_id": "true-pair", "metadata_json": json.dumps(metadata)},
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert len(calls) == 3  # two normalized plates plus the registered right plate
    assert result["bilateral_pairs_created"] == 1
    pair = result["bilateral_analysis"][0]
    assert pair["tlc_profile_id"] == "client-device-tlc-pending"
    assert pair["device_profile_id"] == "client-device-01"
    assert 0 <= pair["dinov2_cosine_distance"] <= 2
    assert 0 <= pair["dinov2_pair_reference_anomaly_score_0_1"] <= 1
    assert pair["clinical_risk"] is None
    assert pair["clinical_claim"] == "NONE"

    saved = client.get("/api/exams/true-pair")
    assert saved.status_code == 200
    saved_result = saved.json()["result"]
    assert saved_result["tlc_profile_ids"] == ["client-device-tlc-pending"]
    assert saved_result["device_profile_ids"] == ["client-device-01"]
    assert saved_result["bilateral_analysis"][0][
        "dinov2_pair_reference_anomaly_score_0_1"
    ] == pair["dinov2_pair_reference_anomaly_score_0_1"]

    report = client.get("/reports/true-pair")
    assert report.status_code == 200
    assert "client-device-tlc-pending" in report.text
    assert "client-device-01" in report.text
    assert "DINOv2 pair reference unusualness score" in report.text
    assert "Clinical claim:</b> NONE" in report.text


def test_pairing_rejects_cross_profile_domain_mix(isolated_runtime, monkeypatch):
    client, calls = isolated_runtime
    monkeypatch.setattr(analysis_engine, "detect_circular_plates", lambda image: [])
    metadata = [
        {
            "filename": "left.png",
            "side": "LEFT",
            "position": "P1",
            "tlc_profile_id": "reference-publication-unknown",
        },
        {
            "filename": "right.png",
            "side": "RIGHT",
            "position": "P1",
            "tlc_profile_id": "client-device-tlc-pending",
        },
    ]
    payload = _png_bytes(_image((20, 100, 180)))
    response = client.post(
        "/api/exams/analyze",
        files=[
            ("files", ("left.png", payload, "image/png")),
            ("files", ("right.png", payload, "image/png")),
        ],
        data={"exam_id": "domain-mismatch", "metadata_json": json.dumps(metadata)},
    )
    assert response.status_code == 200
    assert response.json()["bilateral_pairs_created"] == 0
    assert len(calls) == 2


def test_profile_controls_active_response_thresholds():
    base = resolve_tlc_profile("client-device-tlc-pending")
    strict = TLCProfile(
        id="strict-test-profile",
        label="Strict test",
        calibration_status="TEST",
        saturation_min=255,
        value_min=255,
        min_component_area_px=base.min_component_area_px,
        normalization_mode="identity",
        qc_dark_clip_value=base.qc_dark_clip_value,
        qc_bright_clip_value=base.qc_bright_clip_value,
        qc_saturation_clip_value=base.qc_saturation_clip_value,
    )
    permissive_features, _, _ = analysis_engine.signal_features(
        _image((20, 100, 180)), base
    )
    strict_features, _, _ = analysis_engine.signal_features(
        _image((20, 100, 180)), strict
    )
    assert permissive_features["response_area_fraction"] > 0
    assert strict_features["response_area_fraction"] == 0


def test_unknown_profile_is_a_bad_request(isolated_runtime, monkeypatch):
    client, calls = isolated_runtime
    monkeypatch.setattr(analysis_engine, "detect_circular_plates", lambda image: [])
    metadata = [
        {"filename": "plate.png", "tlc_profile_id": "not-a-configured-profile"}
    ]
    response = client.post(
        "/api/exams/analyze",
        files=[("files", ("plate.png", _png_bytes(_image((20, 100, 180))), "image/png"))],
        data={"metadata_json": json.dumps(metadata)},
    )
    assert response.status_code == 400
    assert "Unknown tlc_profile_id" in response.json()["detail"]
    assert calls == []
