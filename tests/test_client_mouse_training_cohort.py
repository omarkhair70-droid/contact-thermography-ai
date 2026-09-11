from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_client_mouse_training_cohort import assess_mouse_training_cohort


ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "subject_id", "image_path", "source_id", "modality", "species", "label",
    "label_provenance", "split_group", "license_tag", "tlc_profile_id",
    "device_profile_id", "acquisition_profile_id", "data_use_status",
    "redistribution_status", "use_role", "train_eligible", "strain_id", "sex",
    "weight_g", "experimental_group", "capture_session_id", "tlc_batch_id",
    "camera_settings_id", "illumination_profile_id", "view_id", "capture_order", "notes",
]


def _row(index: int, label: str, **overrides) -> dict:
    subject_id = f"NEW-MOUSE-{index:03d}"
    positive = label == "TUMOR_BEARING"
    row = {
        "subject_id": subject_id,
        "image_path": f"client-private://new-mice/{subject_id}.jpg",
        "source_id": "client-mice-native-training-2026",
        "modality": "contact-LCT",
        "species": "mouse",
        "label": label,
        "label_provenance": (
            "experimentally tumor-bearing under existing approved study protocol"
            if positive else
            "documented untreated control under existing approved study protocol"
        ),
        "split_group": subject_id,
        "license_tag": "PRIVATE-CLIENT-DATA-AGREEMENT",
        "tlc_profile_id": "client-device-tlc-pending",
        "device_profile_id": "client-device-v1",
        "acquisition_profile_id": "client-mouse-acq-v1",
        "data_use_status": "MODEL_RESEARCH_ALLOWED",
        "redistribution_status": "NO_REDISTRIBUTION",
        "use_role": "TRAIN_CANDIDATE",
        "train_eligible": True,
        "strain_id": "client-mouse-strain-v1",
        "sex": "F",
        "weight_g": 24.0 + (index % 3),
        "experimental_group": (
            "TUMOR_BEARING_EXISTING_PROTOCOL" if positive else "UNTREATED_CONTROL_EXISTING_PROTOCOL"
        ),
        "capture_session_id": f"SESSION-{1 + ((index - 1) // 4)}",
        "tlc_batch_id": "TLC-BATCH-A",
        "camera_settings_id": "CAMERA-LOCKED-A",
        "illumination_profile_id": "LIGHT-A",
        "view_id": "CANONICAL-V1",
        "capture_order": index,
        "notes": "synthetic test metadata only",
    }
    row.update(overrides)
    return row


def _balanced_rows() -> list[dict]:
    rows = []
    for pair in range(5):
        rows.append(_row(pair * 2 + 1, "TUMOR_BEARING", capture_session_id=f"SESSION-{pair // 2 + 1}"))
        rows.append(_row(pair * 2 + 2, "HEALTHY", capture_session_id=f"SESSION-{pair // 2 + 1}"))
    return rows


def _write(tmp_path: Path, rows: list[dict], name: str = "mouse.csv") -> Path:
    path = tmp_path / name
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


def test_committed_mouse_template_is_empty_and_blocked() -> None:
    result, native = assess_mouse_training_cohort(
        ROOT / "data" / "client_mouse_training_cohort_template.csv",
        allow_empty=True,
    )
    assert result["status"] == "EMPTY_TEMPLATE"
    assert result["binary_training_ready"] is False
    assert result["target_tlc_profile_id"] == "client-device-tlc-pending"
    assert native is None
    assert result["clinical_claim"] == "NONE"


def test_balanced_new_same_domain_cohort_opens_technical_gate(tmp_path: Path) -> None:
    result, native = assess_mouse_training_cohort(_write(tmp_path, _balanced_rows()))
    assert result["status"] == "GREEN"
    assert result["binary_training_ready"] is True
    assert result["positive_subjects"] == 5
    assert result["negative_subjects"] == 5
    assert result["target_tlc_profile_id"] == "client-device-tlc-pending"
    assert native is not None
    assert len(native) == 10
    assert native["subject_id"].is_unique
    assert set(native["use_role"]) == {"TRAIN_CANDIDATE"}


def test_controls_alone_do_not_open_gate_because_frozen_positives_stay_eval(tmp_path: Path) -> None:
    rows = [_row(i, "HEALTHY", capture_session_id="SESSION-1") for i in range(1, 6)]
    result, native = assess_mouse_training_cohort(_write(tmp_path, rows))
    assert result["binary_training_ready"] is False
    assert result["positive_subjects"] == 0
    assert any("NEW tumor-bearing" in reason for reason in result["reasons"])
    assert native is None


def test_frozen_client_subject_cannot_be_reused_for_training(tmp_path: Path) -> None:
    rows = _balanced_rows()
    rows[0]["subject_id"] = "CLIENT-MOUSE-0030"
    rows[0]["split_group"] = "CLIENT-MOUSE-0030"
    with pytest.raises(ValueError, match="frozen client evaluation subjects"):
        assess_mouse_training_cohort(_write(tmp_path, rows))


def test_non_target_tlc_profile_is_blocked_by_417_feature_contract(tmp_path: Path) -> None:
    rows = _balanced_rows()
    for row in rows:
        row["tlc_profile_id"] = "reference-publication-unknown"
    result, native = assess_mouse_training_cohort(_write(tmp_path, rows))
    assert result["binary_training_ready"] is False
    assert any("lct-target-v1 is currently profile-locked" in reason for reason in result["reasons"])
    assert native is None


def test_mixed_strain_blocks_same_domain_training(tmp_path: Path) -> None:
    rows = _balanced_rows()
    rows[-1]["strain_id"] = "OTHER-STRAIN"
    result, native = assess_mouse_training_cohort(_write(tmp_path, rows))
    assert result["binary_training_ready"] is False
    assert any("one mouse strain" in reason for reason in result["reasons"])
    assert native is None


def test_mixed_sex_blocks_strict_v1_training_pool(tmp_path: Path) -> None:
    rows = _balanced_rows()
    rows[-1]["sex"] = "M"
    result, native = assess_mouse_training_cohort(_write(tmp_path, rows))
    assert result["binary_training_ready"] is False
    assert any("one mouse sex" in reason for reason in result["reasons"])
    assert native is None


def test_class_separated_capture_sessions_are_flagged_as_confound(tmp_path: Path) -> None:
    rows = _balanced_rows()
    for row in rows:
        row["capture_session_id"] = "POSITIVE-DAY" if row["label"] == "TUMOR_BEARING" else "CONTROL-DAY"
    result, native = assess_mouse_training_cohort(_write(tmp_path, rows))
    assert result["binary_training_ready"] is False
    assert any("disjoint capture sessions" in reason for reason in result["reasons"])
    assert native is None
