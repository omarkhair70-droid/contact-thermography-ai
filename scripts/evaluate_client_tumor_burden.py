from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.target_feature_builder import SIGNAL_FEATURES, feature_schema


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.std(xr) <= 1e-12 or np.std(yr) <= 1e-12:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


def evaluate(features_csv: Path, ground_truth_csv: Path) -> dict:
    features = pd.read_csv(features_csv, keep_default_na=False)
    truth = pd.read_csv(ground_truth_csv, keep_default_na=False)
    required_features = {"subject_id", "label", "use_role", "train_eligible", "feature_contract_version"}
    required_truth = {
        "subject_id", "tumor_status", "tumor_size_value", "tumor_size_unit",
        "tumor_volume_value", "tumor_volume_unit", "measurement_method", "measurement_timepoint",
    }
    missing_f = sorted(required_features - set(features.columns))
    missing_t = sorted(required_truth - set(truth.columns))
    if missing_f or missing_t:
        raise ValueError(f"missing columns: features={missing_f}, ground_truth={missing_t}")

    if not (features["label"] == "TUMOR_BEARING").all():
        raise ValueError("client burden evaluator accepts the frozen tumor-bearing cohort only")
    if not (features["use_role"] == "FROZEN_TARGET_EVAL").all():
        raise ValueError("client burden evaluator requires FROZEN_TARGET_EVAL features")
    if features["train_eligible"].astype(str).str.lower().isin({"true", "1", "yes"}).any():
        raise ValueError("frozen target features must not be train_eligible")
    if not (truth["tumor_status"] == "TUMOR_BEARING").all():
        raise ValueError("ground truth must preserve tumor-bearing status")

    size = pd.to_numeric(truth["tumor_size_value"], errors="coerce")
    volume = pd.to_numeric(truth["tumor_volume_value"], errors="coerce")
    if volume.notna().sum() >= 5:
        truth = truth.assign(_burden=volume)
        burden_field = "tumor_volume_value"
        unit_field = "tumor_volume_unit"
    elif size.notna().sum() >= 5:
        truth = truth.assign(_burden=size)
        burden_field = "tumor_size_value"
        unit_field = "tumor_size_unit"
    else:
        raise ValueError("need at least 5 numeric tumor size or volume measurements")

    selected_truth = truth[truth["_burden"].notna()].copy()
    joined = features.merge(
        selected_truth[["subject_id", "_burden", unit_field, "measurement_method", "measurement_timepoint"]],
        on="subject_id",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) < 5:
        raise ValueError("fewer than 5 subjects remain after joining features to tumor burden")

    units = sorted(value for value in joined[unit_field].astype(str).unique() if value.strip())
    methods = sorted(value for value in joined["measurement_method"].astype(str).unique() if value.strip())
    timepoints = sorted(value for value in joined["measurement_timepoint"].astype(str).unique() if value.strip())
    warnings = []
    if len(units) != 1:
        warnings.append("tumor burden units are missing or mixed; correlation is exploratory only")
    if len(methods) > 1:
        warnings.append("measurement methods are mixed")
    if len(timepoints) > 1:
        warnings.append("measurement timepoints are mixed")

    schema = feature_schema()
    signal_start = int(schema["groups"]["signal_features"]["start"])
    correlations = []
    y = joined["_burden"].to_numpy(dtype=float)
    for offset, feature_name in enumerate(SIGNAL_FEATURES):
        column = f"feature_{signal_start + offset:03d}"
        if column not in joined.columns:
            raise ValueError(f"missing pre-specified signal feature column: {column}")
        x = pd.to_numeric(joined[column], errors="raise").to_numpy(dtype=float)
        correlations.append(
            {
                "feature": feature_name,
                "feature_column": column,
                "spearman_rho": _spearman(x, y),
            }
        )
    correlations.sort(key=lambda row: abs(row["spearman_rho"]), reverse=True)

    return {
        "analysis": "pre-specified frozen client tumor-burden rank association",
        "subjects_with_burden": int(len(joined)),
        "burden_field": burden_field,
        "units": units,
        "measurement_methods": methods,
        "measurement_timepoints": timepoints,
        "signal_feature_correlations": correlations,
        "warnings": warnings,
        "semantics": (
            "exploratory association inside an all-tumor-bearing mouse cohort; "
            "not a tumor/no-tumor classifier and not clinical validation"
        ),
        "clinical_claim": "NONE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features_csv", type=Path)
    parser.add_argument("ground_truth_csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    result = evaluate(args.features_csv, args.ground_truth_csv)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
