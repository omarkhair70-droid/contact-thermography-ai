from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.services.client_device_domain import analyze_client_image


def test_client_analysis_contract_has_no_diagnostic_or_temperature_claim(tmp_path: Path):
    image = np.zeros((320, 240, 3), dtype=np.uint8)
    cv2.ellipse(image, (120, 160), (65, 35), 10, 0, 360, (45, 230, 170), -1)
    path = tmp_path / "client.jpg"
    assert cv2.imwrite(str(path), image)

    result, normalized, mask = analyze_client_image(path)
    assert normalized.shape == (256, 256, 3)
    assert mask.shape == (256, 256)
    assert result["clinical_claim"] == "NONE"
    assert result["absolute_temperature_interpretation"] == "DISABLED_UNCALIBRATED_TLC"
    assert "tumor_probability" not in result
    assert "diagnosis" not in result
    assert result["response_area_fraction"] > 0
