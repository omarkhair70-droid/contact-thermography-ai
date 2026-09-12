from pathlib import Path

import pandas as pd
import pytest

from scripts.validate_lct_target_dataset import validate


def _base_rows() -> list[dict]:
    return [
        {
            "subject_id": "P1",
            "image_path": "external://p1",
            "source_id": "source-a",
            "modality": "contact-LCT",
            "species": "human",
            "label": "MALIGNANT",
            "label_provenance": "biopsy confirmed",
            "split_group": "REFERENCE_EXEMPLAR",
            "license_tag": "CC-BY-4.0",
            "tlc_profile_id": "profile-a",
            "device_profile_id": "device-a",
            "use_role": "REFERENCE_ONLY",
            "train_eligible": False,
        },
        {
            "subject_id": "P2",
            "image_path": "external://p2",
            "source_id": "source-a",
            "modality": "contact-LCT",
            "species": "human",
            "label": "BENIGN",
            "label_provenance": "biopsy confirmed",
            "split_group": "REFERENCE_EXEMPLAR",
            "license_tag": "CC-BY-4.0",
            "tlc_profile_id": "profile-a",
            "device_profile_id": "device-a",
            "use_role": "REFERENCE_ONLY",
            "train_eligible": False,
        },
    ]


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_reference_only_manifest_is_valid_but_not_training_ready(tmp_path: Path) -> None:
    result = validate(_write(tmp_path, _base_rows()))
    assert result["valid"] is True
    assert result["binary_training_ready"] is False
    assert result["clinical_claim"] == "NONE"


def test_client_mouse_healthy_label_is_allowed_after_client_clarification(tmp_path: Path) -> None:
    rows = _base_rows()
    row = rows[0].copy()
    row.update(
        {
            "subject_id": "CLIENT-MOUSE-1",
            "source_id": "client-mice-2026",
            "species": "mouse",
            "label": "HEALTHY",
            "use_role": "FROZEN_TARGET_EVAL",
            "split_group": "CLIENT-MOUSE-1",
            "train_eligible": False,
        }
    )
    rows.append(row)
    result = validate(_write(tmp_path, rows))
    assert result["valid"] is True


def test_pending_client_class_map_is_valid_but_warned(tmp_path: Path) -> None:
    rows = _base_rows()
    row = rows[0].copy()
    row.update(
        {
            "subject_id": "CLIENT-MOUSE-1",
            "source_id": "client-mice-2026",
            "species": "mouse",
            "label": "PENDING_CLIENT_CLASS_MAP",
            "use_role": "FROZEN_TARGET_EVAL",
            "split_group": "CLIENT-MOUSE-1",
            "train_eligible": False,
        }
    )
    rows.append(row)
    result = validate(_write(tmp_path, rows))
    assert result["valid"] is True
    assert any("pending per-image class labels" in warning for warning in result["warnings"])


def test_reference_only_row_cannot_be_train_eligible(tmp_path: Path) -> None:
    rows = _base_rows()
    rows[0]["train_eligible"] = True
    with pytest.raises(ValueError, match="only TRAIN_CANDIDATE"):
        validate(_write(tmp_path, rows))


def test_benign_train_candidate_requires_explicit_presence_endpoint(tmp_path: Path) -> None:
    rows = _base_rows()
    rows[1]["use_role"] = "TRAIN_CANDIDATE"
    rows[1]["train_eligible"] = True
    with pytest.raises(ValueError, match="BENIGN cannot be mapped"):
        validate(_write(tmp_path, rows))
