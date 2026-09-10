from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from app.services import analysis_engine
from app.services.client_device_domain import (
    ClientDeviceSegmentationConfig,
    letterbox_square,
    robust_colour_domain_shift,
    segment_client_response,
    summarize_response,
)


def test_client_selector_rejects_dim_saturated_background_and_keeps_bright_response():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:] = (30, 20, 25)
    # Dim saturated nuisance region: generic S/V thresholding can over-segment this.
    cv2.rectangle(image, (15, 15), (120, 90), (80, 25, 75), -1)
    # Bright chromatic TLC-like response region.
    cv2.ellipse(image, (150, 145), (55, 28), 15, 0, 360, (50, 235, 175), -1)

    mask, components = segment_client_response(image)
    assert mask[145, 150] == 255
    assert mask[45, 45] == 0
    assert len(components) == 1

    features = summarize_response(image, mask)
    assert features["response_area_fraction"] > 0.02
    assert features["component_count"] == 1


def test_client_selector_config_is_profile_specific_not_temperature_calibration():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(image, (128, 128), 45, (50, 220, 160), -1)
    permissive, _ = segment_client_response(image)
    strict, _ = segment_client_response(
        image,
        config=ClientDeviceSegmentationConfig(component_mean_value_min=250.0),
    )
    assert permissive.sum() > 0
    assert strict.sum() == 0


def test_letterbox_returns_explicit_valid_area():
    image = np.full((100, 200, 3), 120, dtype=np.uint8)
    square, valid = letterbox_square(image, 256)
    assert square.shape == (256, 256, 3)
    assert valid.shape == (256, 256)
    assert 0.45 < valid.mean() < 0.55


def test_colour_domain_shift_is_research_only_and_finite():
    reference = pd.DataFrame(
        {
            "hue_mean": [60.0, 70.0, 80.0],
            "hue_std": [10.0, 11.0, 12.0],
            "saturation_mean": [90.0, 100.0, 110.0],
            "value_mean": [70.0, 80.0, 90.0],
            "lab_a_mean": [120.0, 125.0, 130.0],
            "lab_b_mean": [120.0, 125.0, 130.0],
        }
    )
    client = pd.DataFrame(
        {
            "hue_mean": [65.0, 75.0],
            "hue_std": [20.0, 22.0],
            "saturation_mean": [100.0, 105.0],
            "value_mean": [210.0, 220.0],
            "lab_a_mean": [105.0, 110.0],
            "lab_b_mean": [135.0, 140.0],
        }
    )
    result = robust_colour_domain_shift(reference, client)
    assert result["clinical_claim"] == "NONE"
    assert np.isfinite(result["max_abs_shift_iqr"])
    assert result["max_abs_shift_iqr"] > 1.0


def test_normal_app_upload_uses_client_selector_for_real_device_profile(client, monkeypatch):
    monkeypatch.setattr(analysis_engine, "detect_circular_plates", lambda image: [])
    image = np.zeros((220, 160, 3), dtype=np.uint8)
    image[:] = (22, 18, 20)
    # Dim chromatic setup/background nuisance.
    cv2.rectangle(image, (5, 5), (120, 60), (75, 20, 70), -1)
    # Bright response deliberately near the image edge; the client path should
    # analyze the whole valid frame rather than clipping to the publication disk.
    cv2.ellipse(image, (30, 180), (24, 15), 0, 0, 360, (45, 230, 170), -1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    response = client.post(
        "/api/exams/analyze",
        files=[("files", ("client-mouse.png", encoded.tobytes(), "image/png"))],
        data={
            "exam_id": "lane-e-client-selector",
            "tlc_profile_id": "client-device-tlc-pending",
            "device_profile_id": "mouse-device-test",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    plate = body["sources"][0]["plates"][0]
    assert body["clinical_claim"] == "NONE"
    assert body["sources"][0]["extraction_mode"] == "whole_image_fallback"
    assert plate["client_device_segmentation"]["selector_version"] == "client-device-v0.1"
    assert plate["client_device_segmentation"]["clinical_claim"] == "NONE"
    assert plate["signal_features"]["response_area_fraction"] > 0.005
    assert plate["signal_features"]["value_mean"] > 160
    assert plate["qc"]["clinical_claim"] == "NONE"
    assert plate["qc"]["semantics"] == "engineering client-device acquisition QC; not a clinical assessment"
    assert "response_touches_valid_edge" in plate["qc"]["metrics"]


def test_reference_profile_keeps_publication_analysis_path(client, sample_png, monkeypatch):
    monkeypatch.setattr(analysis_engine, "detect_circular_plates", lambda image: [])
    response = client.post(
        "/api/exams/analyze",
        files=[("files", ("reference.png", sample_png, "image/png"))],
        data={"exam_id": "lane-e-reference-control", "tlc_profile_id": "reference-publication-unknown"},
    )
    assert response.status_code == 200, response.text
    plate = response.json()["sources"][0]["plates"][0]
    assert "client_device_segmentation" not in plate
    assert plate["tlc_profile_id"] == "reference-publication-unknown"
    assert plate["clinical_claim"] == "NONE"
