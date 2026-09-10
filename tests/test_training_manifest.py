from pathlib import Path

import pandas as pd
import pytest

from scripts.validate_training_manifest import validate_manifest


BASE = {
    "subject_id": "S1",
    "image_path": "/private/s1.jpg",
    "source_id": "fixture",
    "modality": "infrared",
    "species": "human",
    "label": "HEALTHY",
    "label_provenance": "fixture",
    "split_group": "S1",
    "license_tag": "RESEARCH-ONLY",
    "tlc_profile_id": "",
    "device_profile_id": "",
}


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_manifest_accepts_multiple_views_with_same_subject_group(tmp_path):
    a = dict(BASE)
    b = dict(BASE, image_path="/private/s1-view2.jpg")
    result = validate_manifest(_write(tmp_path, [a, b]))
    assert result["valid"] is True
    assert result["subjects"] == 1
    assert result["clinical_claim"] == "NONE"


def test_manifest_rejects_subject_split_leakage(tmp_path):
    a = dict(BASE)
    b = dict(BASE, image_path="/private/s1-view2.jpg", split_group="OTHER")
    with pytest.raises(ValueError, match="multiple split groups"):
        validate_manifest(_write(tmp_path, [a, b]))


def test_contact_lct_requires_profile_provenance(tmp_path):
    row = dict(BASE, modality="contact-LCT", label="TUMOR_BEARING")
    with pytest.raises(ValueError, match="tlc_profile_id"):
        validate_manifest(_write(tmp_path, [row]))


def test_noncommercial_or_unknown_license_is_visible_warning(tmp_path):
    row = dict(BASE, license_tag="CC-BY-NC-3.0")
    result = validate_manifest(_write(tmp_path, [row]))
    assert result["warnings"]
