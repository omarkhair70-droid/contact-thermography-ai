from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ALLOWED_ROLES = {"OPEN_LICENSE_REFERENCE", "COPYRIGHT_RESTRICTED_REFERENCE"}
ALLOWED_LABELS = {"MALIGNANT", "BENIGN", "HEALTHY", "TUMOR_BEARING"}


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def validate_reference_cases(path: Path) -> dict:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "case_id",
        "source_id",
        "year",
        "species",
        "modality",
        "ground_truth_label",
        "ground_truth_provenance",
        "lct_pattern",
        "image_access",
        "license_status",
        "use_role",
        "train_eligible",
        "source_url",
        "notes",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing reference-case columns: {missing}")
    if frame.empty:
        raise ValueError("reference-case registry is empty")
    if frame["case_id"].duplicated().any():
        dup = frame.loc[frame["case_id"].duplicated(), "case_id"].tolist()
        raise ValueError(f"duplicate case IDs: {dup[:20]}")
    if set(frame["modality"]) != {"contact-LCT"}:
        raise ValueError("every registered case must be contact-LCT")
    bad_roles = sorted(set(frame["use_role"]) - ALLOWED_ROLES)
    if bad_roles:
        raise ValueError(f"unsupported reference roles: {bad_roles}")
    bad_labels = sorted(set(frame["ground_truth_label"]) - ALLOWED_LABELS)
    if bad_labels:
        raise ValueError(f"unsupported ground-truth labels: {bad_labels}")
    if frame["ground_truth_provenance"].str.strip().eq("").any():
        raise ValueError("every case requires explicit ground-truth provenance")
    if frame["source_url"].str.strip().eq("").any():
        raise ValueError("every case requires a source URL")
    if frame["train_eligible"].map(_as_bool).any():
        raise ValueError("reference-case registry must not silently enable native training")

    open_rows = frame[frame["use_role"] == "OPEN_LICENSE_REFERENCE"]
    if not open_rows.empty and not open_rows["license_status"].str.startswith("CC-BY").all():
        raise ValueError("OPEN_LICENSE_REFERENCE rows require an explicit CC-BY license status")

    restricted = frame[frame["use_role"] == "COPYRIGHT_RESTRICTED_REFERENCE"]
    if not restricted.empty and restricted["license_status"].str.startswith("CC-BY").any():
        raise ValueError("CC-BY rows should not be marked copyright-restricted")

    return {
        "status": "GREEN",
        "cases": int(len(frame)),
        "open_license_reference_cases": int(len(open_rows)),
        "copyright_restricted_reference_cases": int(len(restricted)),
        "malignant_cases": int((frame["ground_truth_label"] == "MALIGNANT").sum()),
        "benign_or_healthy_cases": int(frame["ground_truth_label"].isin({"BENIGN", "HEALTHY"}).sum()),
        "train_eligible_cases": 0,
        "clinical_claim": "NONE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_reference_cases(args.registry), indent=2))


if __name__ == "__main__":
    main()
