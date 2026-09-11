from pathlib import Path

import pandas as pd
import pytest

from scripts.validate_external_lct_intake import validate_external_lct_intake


ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "subject_id", "image_path", "source_id", "modality", "species", "label",
    "label_provenance", "split_group", "license_tag", "tlc_profile_id",
    "device_profile_id", "acquisition_profile_id", "data_use_status",
    "redistribution_status", "use_role", "train_eligible", "notes",
]


def _row(**overrides):
    row = {
        "subject_id": "S-001",
        "image_path": "incoming://S-001/view-1.png",
        "source_id": "external-test",
        "modality": "contact-LCT",
        "species": "mouse",
        "label": "HEALTHY",
        "label_provenance": "experimentally confirmed untreated control mouse",
        "split_group": "S-001",
        "license_tag": "PRIVATE-DATA-AGREEMENT",
        "tlc_profile_id": "client-device-tlc-v1",
        "device_profile_id": "client-device-v1",
        "acquisition_profile_id": "client-acq-v1",
        "data_use_status": "MODEL_RESEARCH_ALLOWED",
        "redistribution_status": "NO_REDISTRIBUTION",
        "use_role": "REFERENCE_ONLY",
        "train_eligible": False,
        "notes": "synthetic test row; not project data",
    }
    row.update(overrides)
    return row


def _write(tmp_path: Path, rows: list[dict], name: str = "intake.csv") -> Path:
    path = tmp_path / name
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


def test_committed_empty_template_has_complete_schema() -> None:
    result = validate_external_lct_intake(
        ROOT / "data" / "external_lct_intake_template.csv", allow_empty=True
    )
    assert result["status"] == "EMPTY_TEMPLATE"
    assert result["train_eligible_rows"] == 0
    assert result["clinical_claim"] == "NONE"


def test_safe_non_trainable_external_row_is_accepted(tmp_path: Path) -> None:
    result = validate_external_lct_intake(_write(tmp_path, [_row()]))
    assert result["status"] == "GREEN"
    assert result["subjects"] == 1
    assert result["train_eligible_rows"] == 0


def test_training_candidate_requires_explicit_model_use_permission(tmp_path: Path) -> None:
    path = _write(tmp_path, [_row(
        use_role="TRAIN_CANDIDATE",
        train_eligible=True,
        data_use_status="REFERENCE_USE_ONLY",
    )])
    with pytest.raises(ValueError, match="model-research data-use permission"):
        validate_external_lct_intake(path)


def test_training_candidate_requires_known_device_and_acquisition_profiles(tmp_path: Path) -> None:
    path = _write(tmp_path, [_row(
        use_role="TRAIN_CANDIDATE",
        train_eligible=True,
        device_profile_id="unknown",
        acquisition_profile_id="pending",
    )])
    with pytest.raises(ValueError, match="device profile must be known"):
        validate_external_lct_intake(path)


def test_same_subject_cannot_change_ground_truth_or_domain(tmp_path: Path) -> None:
    rows = [
        _row(image_path="incoming://S-001/view-1.png"),
        _row(image_path="incoming://S-001/view-2.png", label="MALIGNANT"),
    ]
    with pytest.raises(ValueError, match="inconsistent label"):
        validate_external_lct_intake(_write(tmp_path, rows))


def test_direct_identifier_columns_are_rejected(tmp_path: Path) -> None:
    frame = pd.DataFrame([_row()])
    frame["patient_name"] = "Do Not Store"
    path = tmp_path / "identifiable.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="direct identifier columns are not allowed"):
        validate_external_lct_intake(path)
