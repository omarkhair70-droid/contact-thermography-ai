from pathlib import Path

import numpy as np

from app.services.report import build_report_html


ROOT = Path(__file__).resolve().parents[1]


def test_human_dino_patch_encoder_batches_eight_frames_into_two_forward_passes(monkeypatch):
    import torch

    from app.services import dinov2_service
    from app.services.dinov2_patch_encoder import encode_patches_batch

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def forward_features(self, tensor):
            self.calls += 1
            batch = int(tensor.shape[0])
            tokens = torch.ones((batch, 16 * 16, 384), dtype=tensor.dtype, device=tensor.device)
            return {"x_norm_patchtokens": tokens}

    fake = FakeModel()
    monkeypatch.setattr(dinov2_service.runtime, "model", fake)
    monkeypatch.setattr(dinov2_service.runtime, "device", "cpu")
    monkeypatch.setattr(dinov2_service.runtime, "load_official", lambda *args, **kwargs: dinov2_service.runtime)

    frames = [np.full((224, 224, 3), index, dtype=np.uint8) for index in range(8)]
    outputs = encode_patches_batch(frames, batch_size=4)

    assert len(outputs) == 8
    assert fake.calls == 2
    for tokens, provenance in outputs:
        assert tokens.shape == (16, 16, 384)
        assert np.isfinite(tokens).all()
        assert provenance["batched_inference"] is True
        assert provenance["batch_size"] == 4


def test_human_ui_leads_with_research_finding_and_runtime():
    text = (ROOT / "app" / "static" / "human-exam.html").read_text(encoding="utf-8")
    assert "Primary research finding" in text
    assert "Research Concern Score" in text
    assert "SUSPICIOUS_RESEARCH" in text
    assert "NOT_SUSPICIOUS_RESEARCH" in text
    assert "INCONCLUSIVE" in text
    assert "Analysis runtime" in text
    assert "research_finding" in text
    assert "human_decision" in text  # clinical calibration remains secondary provenance
    assert "DINO frames are batched" in text


def test_root_product_shell_redirects_to_human_workspace():
    text = (ROOT / "app" / "templates" / "index.html").read_text(encoding="utf-8")
    assert "window.location.replace('/human-exam')" in text
    assert "Contact Thermography AI" not in text
    assert "MumGuard" in text


def test_direct_human_report_leads_with_research_result_and_hides_legacy_sections():
    result = {
        "exam_id": "final-product-demo",
        "human_runtime": "DIRECT_SESSION_ONLY",
        "tlc_profile_id": "client-device-tlc-pending",
        "source_images": 8,
        "plates_detected": 0,
        "bilateral_pairs_created": 0,
        "sources": [],
        "bilateral_analysis": [],
        "profile_provenance": {"domain_status": "SEPARATE_TLC_DOMAIN"},
        "mumguard_session_evidence": {
            "architecture": "mumguard_session_fusion_v1",
            "status": "OK",
            "measurement_mode": "relative_tlc_signal",
            "target_species": "human",
            "research_finding": {
                "status": "SUSPICIOUS_RESEARCH",
                "research_concern_score": 82.4,
                "domain_status": "IN_SOURCE_SUPPORT",
                "likely_side": "LEFT",
                "dominant_channels": [
                    {"channel": "focal/core hyperthermia", "evidence": 0.71},
                    {"channel": "bilateral asymmetry", "evidence": 0.62},
                ],
                "reason": "Auxiliary human research-transfer morphology exceeded the research threshold.",
                "score_semantics": "research concern only; not a cancer probability",
            },
            "three_channel_scores": {
                "core_hyperthermia_score": 0.71,
                "bilateral_asymmetry_score": 0.62,
                "abnormal_skin_behavior_score": 0.55,
                "overall_measurement_evidence_score": 0.64,
            },
            "human_decision": {
                "status": "NOT_CALIBRATED",
                "decision_status": "NOT_CALIBRATED",
                "risk_score": None,
                "calibrated": False,
                "reason": "No clinical calibration active.",
            },
            "left": {"observable_fraction": 0.8},
            "right": {"observable_fraction": 0.79},
            "bilateral": {"features": {"joint_fraction": 0.72}},
            "ai_evidence_available": True,
            "performance": {
                "total_analysis_s": 42.3,
                "frame_preparation_s": 2.1,
                "dino_batch_s": 31.0,
                "side_reconstruction_s": 4.5,
            },
            "response_support_semantics": "PROVISIONAL_VISIBLE_TLC_RESPONSE_NOT_CONFIRMED_TISSUE_CONTACT",
        },
    }

    html = build_report_html(result)
    assert "MumGuard Examination Report" in html
    assert "Primary research finding" in html
    assert "SUSPICIOUS_RESEARCH" in html
    assert "82.4/100" in html
    assert "LEFT" in html
    assert "Runtime:" in html
    assert "Human decision layer — clinical calibration (secondary)" in html
    assert "Plate analysis" not in html
    assert "Legacy pairwise bilateral analysis" not in html
    assert "not a cancer probability" in html
