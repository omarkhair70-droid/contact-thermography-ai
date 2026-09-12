import io

import numpy as np
from PIL import Image

from app.services.mumguard_session_fusion import SessionFrameInput, analyze_bilateral_session


def _png_bytes(rgb: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(out, format="PNG")
    return out.getvalue()


def _base_rgb() -> np.ndarray:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[:, :] = [0, 220, 0]
    return image


def test_session_fusion_runs_without_independent_contact_annotation():
    left = _base_rgb()
    right = _base_rgb()
    result, arrays = analyze_bilateral_session(
        [
            SessionFrameInput(_png_bytes(left), "left.png", "LEFT", 1),
            SessionFrameInput(_png_bytes(right), "right.png", "RIGHT", 1),
        ],
        include_dino=False,
    )

    assert result["architecture"] == "mumguard_session_fusion_v1"
    assert result["target_species"] == "human"
    assert result["contact_annotation_required_for_measurement"] is False
    assert result["clinical_claim"] == "NONE"
    assert result["left"]["observable_fraction"] > 0.5
    assert result["right"]["observable_fraction"] > 0.5
    assert arrays["left_signal"].ndim == 2
    assert arrays["bilateral_asymmetry"].shape == (256, 256)


def test_session_fusion_detects_one_sided_relative_signal_change():
    left = _base_rgb()
    right = _base_rgb()
    left[70:150, 40:120] = [220, 0, 0]

    result, _ = analyze_bilateral_session(
        [
            SessionFrameInput(_png_bytes(left), "left.png", "LEFT", 1),
            SessionFrameInput(_png_bytes(right), "right.png", "RIGHT", 1),
        ],
        include_dino=False,
    )

    scores = result["three_channel_scores"]
    assert scores["bilateral_asymmetry_score"] > 0.0
    assert scores["core_hyperthermia_score"] >= 0.0
    assert scores["abnormal_skin_behavior_score"] >= 0.0
