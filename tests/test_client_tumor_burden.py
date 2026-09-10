from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.target_feature_builder import FEATURE_CONTRACT_VERSION, feature_columns
from scripts.evaluate_client_tumor_burden import evaluate


def _features(rows: int = 6) -> pd.DataFrame:
    columns = feature_columns()
    records = []
    for index in range(rows):
        record = {
            "subject_id": f"CLIENT-MOUSE-{index:04d}",
            "label": "TUMOR_BEARING",
            "use_role": "FROZEN_TARGET_EVAL",
            "train_eligible": False,
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
        }
        values = np.zeros(len(columns), dtype=float)
        values[0] = float(index + 1)
        values[2] = float((index + 1) * 2)
        record.update({name: float(value) for name, value in zip(columns, values)})
        records.append(record)
    return pd.DataFrame(records)


def _truth(rows: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": f"CLIENT-MOUSE-{index:04d}",
                "tumor_status": "TUMOR_BEARING",
                "tumor_size_value": float(index + 1),
                "tumor_size_unit": "mm",
                "tumor_volume_value": "",
                "tumor_volume_unit": "",
                "measurement_method": "caliper",
                "measurement_timepoint": "same-day",
            }
            for index in range(rows)
        ]
    )


def _write(tmp_path: Path, name: str, frame: pd.DataFrame) -> Path:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def test_burden_evaluator_uses_prespecified_signal_features(tmp_path: Path) -> None:
    result = evaluate(
        _write(tmp_path, "features.csv", _features()),
        _write(tmp_path, "truth.csv", _truth()),
    )
    assert result["subjects_with_burden"] == 6
    assert result["burden_field"] == "tumor_size_value"
    by_name = {row["feature"]: row["spearman_rho"] for row in result["signal_feature_correlations"]}
    assert by_name["response_area_fraction"] == pytest.approx(1.0)
    assert result["clinical_claim"] == "NONE"


def test_burden_evaluator_requires_real_measurements(tmp_path: Path) -> None:
    truth = _truth()
    truth["tumor_size_value"] = ""
    with pytest.raises(ValueError, match="at least 5 numeric"):
        evaluate(
            _write(tmp_path, "features.csv", _features()),
            _write(tmp_path, "truth.csv", truth),
        )


def test_burden_evaluator_rejects_non_frozen_or_non_tumor_rows(tmp_path: Path) -> None:
    features = _features()
    features.loc[0, "use_role"] = "TRAIN_CANDIDATE"
    with pytest.raises(ValueError, match="FROZEN_TARGET_EVAL"):
        evaluate(
            _write(tmp_path, "features.csv", features),
            _write(tmp_path, "truth.csv", _truth()),
        )
