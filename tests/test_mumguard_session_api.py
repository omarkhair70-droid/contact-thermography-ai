import json


def _metadata():
    return [
        {
            "filename": "left.png",
            "side": "LEFT",
            "position": "LEFT-SWEEP",
            "sequence_index": 1,
            "tlc_profile_id": "client-device-tlc-pending",
            "species": "human",
            "acquisition_type": "contact-LCT",
        },
        {
            "filename": "right.png",
            "side": "RIGHT",
            "position": "RIGHT-SWEEP",
            "sequence_index": 1,
            "tlc_profile_id": "client-device-tlc-pending",
            "species": "human",
            "acquisition_type": "contact-LCT",
        },
    ]


def test_bilateral_upload_builds_unified_mumguard_session(client, sample_png):
    response = client.post(
        "/api/exams/analyze",
        files=[
            ("files", ("left.png", sample_png, "image/png")),
            ("files", ("right.png", sample_png, "image/png")),
        ],
        data={
            "exam_id": "mumguard-session-api",
            "metadata_json": json.dumps(_metadata()),
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    session = payload["mumguard_session_evidence"]
    assert session["architecture"] == "mumguard_session_fusion_v1"
    assert session["target_species"] == "human"
    assert session["measurement_mode"] == "relative_tlc_signal"
    assert session["contact_annotation_required_for_measurement"] is False
    assert session["clinical_claim"] == "NONE"
    assert session["evidence_url"].endswith("/mumguard-session/session_evidence.json")
    assert session["maps_url"].endswith("/mumguard-session/session_maps.npz")
    assert set(session["three_channel_scores"]).issuperset({
        "core_hyperthermia_score",
        "bilateral_asymmetry_score",
        "abnormal_skin_behavior_score",
        "overall_measurement_evidence_score",
    })
    assert payload["model_status"] == "MUMGUARD_SESSION_EVIDENCE_RESEARCH"

    saved = client.get("/api/exams/mumguard-session-api")
    assert saved.status_code == 200
    assert saved.json()["result"]["mumguard_session_evidence"]["architecture"] == "mumguard_session_fusion_v1"

    report = client.get("/reports/mumguard-session-api")
    assert report.status_code == 200
    assert "MumGuard session evidence" in report.text
    assert "Core hyperthermia evidence" in report.text
    assert "Bilateral asymmetry evidence" in report.text
    assert "Abnormal skin behaviour" in report.text


def test_single_side_keeps_exam_but_marks_session_incomplete(client, sample_png):
    metadata = [_metadata()[0]]
    response = client.post(
        "/api/exams/analyze",
        files=[("files", ("left.png", sample_png, "image/png"))],
        data={
            "exam_id": "mumguard-session-incomplete",
            "metadata_json": json.dumps(metadata),
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["mumguard_session_evidence"] is None
    assert any(
        item["reason"] == "MUMGUARD_BILATERAL_SESSION_INCOMPLETE"
        for item in payload["bilateral_pairing_warnings"]
    )
