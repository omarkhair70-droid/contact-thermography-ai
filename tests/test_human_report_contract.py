from app.services.report import build_report_html


def test_report_renders_human_decision_and_explainable_session_maps():
    result = {
        "exam_id": "human-demo",
        "tlc_profile_id": "client-device-tlc-pending",
        "source_images": 2,
        "plates_detected": 2,
        "bilateral_pairs_created": 1,
        "sources": [],
        "bilateral_analysis": [],
        "profile_provenance": {"domain_status": "PENDING_CALIBRATION"},
        "mumguard_session_evidence": {
            "architecture": "mumguard_session_fusion_v1",
            "status": "OK",
            "measurement_mode": "relative_tlc_signal",
            "target_species": "human",
            "three_channel_scores": {
                "core_hyperthermia_score": 0.5,
                "bilateral_asymmetry_score": 0.4,
                "abnormal_skin_behavior_score": 0.3,
                "overall_measurement_evidence_score": 0.42,
            },
            "human_decision": {
                "decision_status": "NOT_CALIBRATED",
                "status": "NOT_CALIBRATED",
                "indication": None,
                "risk_score": None,
                "calibrated": False,
                "reason": "No human calibration active.",
            },
            "left": {"observable_fraction": 0.8},
            "right": {"observable_fraction": 0.81},
            "bilateral": {"features": {"joint_fraction": 0.7}},
            "ai_evidence_available": True,
            "response_support_semantics": "PROVISIONAL_VISIBLE_TLC_RESPONSE_NOT_CONFIRMED_TISSUE_CONTACT",
            "evidence_url": "/static/generated/human-demo/mumguard-session/session_evidence.json",
            "maps_url": "/static/generated/human-demo/mumguard-session/session_maps.npz",
            "preview_filenames": {
                "left_thermal_evidence": "left_thermal_evidence.png",
                "right_thermal_evidence": "right_thermal_evidence.png",
                "bilateral_asymmetry": "bilateral_asymmetry.png",
            },
        },
    }
    html = build_report_html(result)
    assert "Human decision layer" in html
    assert "NOT_CALIBRATED" in html
    assert "Risk score:</b> not available" in html
    assert "Explainable session maps" in html
    assert "left_thermal_evidence.png" in html
    assert "bilateral_asymmetry.png" in html
    assert "not cancer probabilities" in html
