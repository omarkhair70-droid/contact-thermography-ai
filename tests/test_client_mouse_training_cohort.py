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
    "camera_settings_id", "illumination_profile_id", "view_id", "capture_order",
    "matches_client_positive_domain", "notes",
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
        "label_provenance": "test fixture",
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
        "experimental_group": "TUMOR" if positive else "CONTROL",
        "capture_session_id": f"SESSION-{1 + ((index - 1) // 4)}",
        "tlc_batch_id": "TLC-BATCH-A",
        "camera_settings_id": "CAMERA-LOCKED-A",
        "illumination_profile_id": "LIGHT-A",
        "view_id": "CANONICAL-V1",
        "capture_order": index,
        "matches_client_positive_domain": True,
        "notes": "synthetic test metadata only",
    }
    row.update(overrides)
    return row


def _write(tmp_path: Path, rows: list[dict], name: str = "mouse.csv") -> Path:
    path = tmp_path / name
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


def _class_map(tmp_path: Path, labels: list[str]) -> Path:
    assert len(labels) == 9
    rows = []
    for offset, label in enumerate(labels, start=30):
        rows.append({
            "subject_id": f"CLIENT-MOUSE-{offset:04d}",
            "image_id": f"IMG-20260910-WA{offset:04d}.jpg",
            "tumor_status": label,
            "tumor_size_value": "",
            "tumor_size_unit": "",
            "tumor_volume_value": "",
            "tumor_volume_unit": "",
            "measurement_method": "",
            "measurement_timepoint": "",
            "control_or_baseline": "",
            "notes": "test class map",
        })
    path = tmp_path / "class_map.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_committed_client_map_is_pending_not_positive_only() -> None:
    result, native = assess_mouse_training_cohort(
        ROOT / "data" / "client_mouse_training_cohort_template.csv",
        allow_empty=True,
    )
    assert result["status"] == "CLASS_MAP_PENDING"
    assert result["binary_training_ready"] is False
    assert len(result["pending_subjects"]) == 9
    assert result["mapped_positive_subjects"] == 0
    assert result["mapped_negative_subjects"] == 0
    assert any("contains both tumor-bearing and normal/no-tumor" in reason for reason in result["reasons"])
    assert native is None


def test_resolved_five_four_client_map_opens_internal_gate_without_new_images(tmp_path: Path) -> None:
    labels = ["TUMOR_BEARING"] * 5 + ["HEALTHY"] * 4
    result, native = assess_mouse_training_cohort(
        ROOT / "data" / "client_mouse_training_cohort_template.csv",
        allow_empty=True,
        class_map_path=_class_map(tmp_path, labels),
    )
    assert result["status"] == "GREEN"
    assert result["binary_training_ready"] is True
    assert result["positive_subjects"] == 5
    assert result["negative_subjects"] == 4
    assert result["existing_client_subjects"] == 9
    assert result["supplemental_trainable_subjects"] == 0
    assert native is not None
    assert len(native) == 9
    assert set(native["label"]) == {"TUMOR_BEARING", "HEALTHY"}
    assert set(native["use_role"]) == {"TRAIN_CANDIDATE"}


def test_one_unmapped_client_subject_keeps_gate_closed(tmp_path: Path) -> None:
    labels = ["TUMOR_BEARING"] * 4 + ["HEALTHY"] * 4 + ["PENDING_CLASS_MAP"]
    result, native = assess_mouse_training_cohort(
        ROOT / "data" / "client_mouse_training_cohort_template.csv",
        allow_empty=True,
        class_map_path=_class_map(tmp_path, labels),
    )
    assert result["status"] == "CLASS_MAP_PENDING"
    assert result["pending_subjects"] == ["CLIENT-MOUSE-0038"]
    assert native is None


def test_client_subject_cannot_be_duplicated_in_supplemental_manifest(tmp_path: Path) -> None:
    labels = ["TUMOR_BEARING"] * 5 + ["HEALTHY"] * 4
    rows = [_row(1, "HEALTHY")]
    rows[0]["subject_id"] = "CLIENT-MOUSE-0030"
    rows[0]["split_group"] = "CLIENT-MOUSE-0030"
    with pytest.raises(ValueError, match="must not be duplicated"):
        assess_mouse_training_cohort(
            _write(tmp_path, rows),
            class_map_path=_class_map(tmp_path, labels),
        )


def test_non_target_tlc_profile_in_supplemental_rows_is_blocked(tmp_path: Path) -> None:
    labels = ["TUMOR_BEARING"] * 5 + ["HEALTHY"] * 4
    rows = [_row(1, "HEALTHY", tlc_profile_id="reference-publication-unknown")]
    result, native = assess_mouse_training_cohort(
        _write(tmp_path, rows),
        class_map_path=_class_map(tmp_path, labels),
    )
    assert result["binary_training_ready"] is False
    assert any("profile-locked" in reason for reason in result["reasons"])
    assert native is None


def test_unconfirmed_domain_match_blocks_supplemental_rows(tmp_path: Path) -> None:
    labels = ["TUMOR_BEARING"] * 5 + ["HEALTHY"] * 4
    rows = [_row(1, "HEALTHY", matches_client_positive_domain=False)]
    result, native = assess_mouse_training_cohort(
        _write(tmp_path, rows),
        class_map_path=_class_map(tmp_path, labels),
    )
    assert result["binary_training_ready"] is False
    assert any("matches_client_positive_domain=true" in reason for reason in result["reasons"])
    assert native is None
