from pathlib import Path

import pandas as pd
import pytest

from scripts.validate_lct_reference_cases import validate_reference_cases


ROOT = Path(__file__).resolve().parents[1]


def test_committed_lct_reference_registry_is_safe() -> None:
    result = validate_reference_cases(ROOT / "data" / "lct_reference_cases_v1.csv")
    assert result["status"] == "GREEN"
    assert result["cases"] >= 8
    assert result["open_license_reference_cases"] >= 3
    assert result["benign_or_healthy_cases"] >= 1
    assert result["train_eligible_cases"] == 0
    assert result["clinical_claim"] == "NONE"


def test_reference_registry_cannot_enable_training(tmp_path: Path) -> None:
    original = pd.read_csv(ROOT / "data" / "lct_reference_cases_v1.csv", keep_default_na=False)
    original.loc[0, "train_eligible"] = True
    candidate = tmp_path / "unsafe.csv"
    original.to_csv(candidate, index=False)
    with pytest.raises(ValueError, match="must not silently enable native training"):
        validate_reference_cases(candidate)


def test_open_reference_requires_explicit_cc_by(tmp_path: Path) -> None:
    original = pd.read_csv(ROOT / "data" / "lct_reference_cases_v1.csv", keep_default_na=False)
    row = original[original["use_role"] == "OPEN_LICENSE_REFERENCE"].index[0]
    original.loc[row, "license_status"] = "UNKNOWN"
    candidate = tmp_path / "bad-license.csv"
    original.to_csv(candidate, index=False)
    with pytest.raises(ValueError, match="explicit CC-BY"):
        validate_reference_cases(candidate)
