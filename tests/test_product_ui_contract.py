from __future__ import annotations

import json


def test_product_ui_exam_level_profile_controls_reach_runtime(client, sample_png):
    metadata = [{
        "filename": "plate.png",
        "side": "LEFT",
        "position": "P1",
        "sequence_index": 1,
    }]
    response = client.post(
        "/api/exams/analyze",
        files=[("files", ("plate.png", sample_png, "image/png"))],
        data={
            "exam_id": "ui-profile-contract",
            "tlc_profile_id": "reference-publication-unknown",
            "device_profile_id": "ui-device-a",
            "metadata_json": json.dumps(metadata),
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["tlc_profile_id"] == "reference-publication-unknown"
    assert body["device_profile_ids"] == ["ui-device-a"]
    assert body["profile_provenance"]["matches_reference_colour_domain"] is True
    plate = body["sources"][0]["plates"][0]
    assert plate["tlc_profile_id"] == "reference-publication-unknown"
    assert plate["device_profile_id"] == "ui-device-a"
    assert plate["clinical_claim"] == "NONE"
