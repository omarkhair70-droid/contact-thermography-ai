from pathlib import Path

import pytest

from scripts.fetch_lct_open_reference_assets import load_asset_manifest, validate_asset_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "lct_open_reference_assets_v1.csv"


def test_committed_open_reference_asset_manifest_is_safe() -> None:
    rows = load_asset_manifest(MANIFEST)
    result = validate_asset_manifest(rows)
    assert result["status"] == "GREEN"
    assert result["assets"] == 3
    assert result["malignant_assets"] == 2
    assert result["benign_or_healthy_assets"] == 1
    assert result["train_eligible_assets"] == 0
    assert result["clinical_claim"] == "NONE"


def test_pack_contains_only_open_reference_rows() -> None:
    rows = load_asset_manifest(MANIFEST)
    assert all(row["use_role"] == "OPEN_LICENSE_REFERENCE" for row in rows)
    assert all(row["license_status"].startswith("CC-BY") for row in rows)
    assert all("braster.eu" not in row["asset_url"] for row in rows)


def test_reference_assets_cannot_become_training_rows() -> None:
    rows = load_asset_manifest(MANIFEST)
    rows[0]["train_eligible"] = "true"
    with pytest.raises(ValueError, match="train_eligible=false"):
        validate_asset_manifest(rows)


def test_reference_asset_requires_explicit_open_license() -> None:
    rows = load_asset_manifest(MANIFEST)
    rows[0]["license_status"] = "COPYRIGHT-UNKNOWN"
    with pytest.raises(ValueError, match="explicit CC-BY"):
        validate_asset_manifest(rows)
