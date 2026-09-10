from __future__ import annotations

import cv2
import numpy as np

from app.services.client_device_domain import segment_client_response
from app.services.client_device_qc import assess_client_device_quality


def test_specular_glare_is_flagged_but_not_selected_as_tlc_response():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:] = (25, 20, 22)
    valid = np.ones((256, 256), dtype=bool)

    # Bright, nearly white setup glare: should trigger QC but fail the saturation selector.
    cv2.rectangle(image, (10, 10), (90, 90), (250, 250, 250), -1)
    # Bright chromatic TLC-like response.
    cv2.ellipse(image, (165, 150), (42, 24), 0, 0, 360, (45, 230, 170), -1)

    mask, _ = segment_client_response(image, valid)
    qc = assess_client_device_quality(image, mask, valid)

    assert mask[40, 40] == 0
    assert mask[150, 165] == 255
    assert "SPECULAR_GLARE_PRESENT" in qc["flags"]
    assert qc["metrics"]["specular_glare_fraction"] > 0.04
    assert qc["clinical_claim"] == "NONE"


def test_response_touching_valid_acquisition_edge_is_review_flag():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:] = (25, 20, 22)
    valid = np.zeros((256, 256), dtype=bool)
    valid[32:224, 48:208] = True
    response = np.zeros((256, 256), dtype=np.uint8)
    response[100:140, 48:80] = 255

    qc = assess_client_device_quality(image, response, valid)
    assert qc["metrics"]["response_touches_valid_edge"] is True
    assert qc["metrics"]["response_edge_fraction"] > 0
    assert "RESPONSE_TOUCHES_FRAME" in qc["flags"]
    assert qc["status"] in {"REVIEW", "REVIEW_REQUIRED"}


def test_clean_client_frame_can_pass_engineering_qc():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    # Textured mid-range frame with no clipping/glare.
    for y in range(256):
        image[y, :, :] = (45 + y % 60, 70 + y % 50, 95 + y % 40)
    valid = np.ones((256, 256), dtype=bool)
    response = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(response, (128, 128), (35, 20), 0, 0, 360, 255, -1)

    qc = assess_client_device_quality(image, response, valid)
    assert "BRIGHT_CLIPPING" not in qc["flags"]
    assert "SPECULAR_GLARE_PRESENT" not in qc["flags"]
    assert "RESPONSE_TOUCHES_FRAME" not in qc["flags"]
    assert qc["clinical_claim"] == "NONE"
