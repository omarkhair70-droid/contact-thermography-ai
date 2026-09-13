import io

import numpy as np
from PIL import Image

from app.services import mumguard_session_fusion as fusion
from app.services.mumguard_session_fusion import (
    SessionFrameInput,
    analyze_bilateral_session,
    persist_session_evidence,
)


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
    assert result["research_finding"]["decision_origin"] == "MUMGUARD_NATIVE_RESEARCH_V0"
    assert result["native_research_decision"]["clinical_claim"] == "NONE"


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


def test_transfer_ood_does_not_force_primary_finding_inconclusive(monkeypatch):
    monkeypatch.setattr(
        fusion,
        "evaluate_session_source_support",
        lambda *args, **kwargs: {
            "status": "ABSTAIN_OOD",
            "research_decision": "INCONCLUSIVE",
            "research_concern_score": None,
            "source_transfer_score_0_1": None,
            "decision_model_executed": False,
            "clinical_risk": None,
            "clinical_claim": "NONE",
            "reason": "synthetic out-of-domain transfer fixture",
        },
    )
    left = _base_rgb()
    right = _base_rgb()
    result, _ = analyze_bilateral_session(
        [
            SessionFrameInput(_png_bytes(left), "left.png", "LEFT", 1),
            SessionFrameInput(_png_bytes(right), "right.png", "RIGHT", 1),
        ],
        include_dino=False,
    )

    assert result["status"] == "OK"
    assert result["research_finding"]["status"] == "NOT_SUSPICIOUS_RESEARCH"
    assert result["research_finding"]["research_concern_score"] is not None
    assert result["research_finding"]["domain_status"] == "ABSTAIN_OOD"
    assert result["research_finding"]["transfer_used"] is False


def test_persist_session_evidence_writes_viewable_preview_maps(tmp_path):
    left = _base_rgb()
    right = _base_rgb()
    left[80:140, 50:120] = [220, 0, 0]
    result, arrays = analyze_bilateral_session(
        [
            SessionFrameInput(_png_bytes(left), "left.png", "LEFT", 1),
            SessionFrameInput(_png_bytes(right), "right.png", "RIGHT", 1),
        ],
        include_dino=False,
    )

    persisted = persist_session_evidence(result, arrays, tmp_path)
    assert (tmp_path / persisted["evidence_filename"]).exists()
    assert (tmp_path / persisted["maps_filename"]).exists()
    previews = persisted["preview_filenames"]
    assert set(previews) == {
        "left_thermal_evidence",
        "right_thermal_evidence",
        "left_fused_evidence",
        "right_fused_evidence",
        "bilateral_asymmetry",
    }
    assert result["preview_filenames"] == previews
    for filename in previews.values():
        path = tmp_path / filename
        assert path.exists()
        image = Image.open(path)
        assert image.width > 0 and image.height > 0
