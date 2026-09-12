from scripts.run_mumguard_signal_locked_eval import _rank_summary


def test_locked_rank_summary_uses_no_fit_and_compares_to_single_negative():
    rows = [
        {
            "subject_id": "neg",
            "core_hyperthermia_score": 0.10,
            "abnormal_skin_behavior_score": 0.20,
            "static_signal_evidence_score": 0.15,
            "global_anomaly_score": 0.25,
        },
        {
            "subject_id": "p1",
            "core_hyperthermia_score": 0.30,
            "abnormal_skin_behavior_score": 0.40,
            "static_signal_evidence_score": 0.35,
            "global_anomaly_score": 0.45,
        },
        {
            "subject_id": "p2",
            "core_hyperthermia_score": 0.05,
            "abnormal_skin_behavior_score": 0.50,
            "static_signal_evidence_score": 0.25,
            "global_anomaly_score": 0.35,
        },
    ]
    labels = {"neg": "absent", "p1": "present", "p2": "present"}
    result = _rank_summary(rows, labels)
    assert result["negative_subjects"] == 1
    assert result["positive_subjects"] == 2
    assert result["feature_results"]["core_hyperthermia_score"]["positives_above_negative"] == 1
    assert result["feature_results"]["static_signal_evidence_score"]["positives_above_negative"] == 2
    assert result["clinical_claim"] == "NONE"
