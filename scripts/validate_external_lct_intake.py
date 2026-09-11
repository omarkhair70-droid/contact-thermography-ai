from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "subject_id",
    "image_path",
    "source_id",
    "modality",
    "species",
    "label",
    "label_provenance",
    "split_group",
    "license_tag",
    "tlc_profile_id",
    "device_profile_id",
    "acquisition_profile_id",
    "data_use_status",
    "redistribution_status",
    "use_role",
    "train_eligible",
    "notes",
}

SUPPORTED_LABELS = {"CANCER", "MALIGNANT", "TUMOR_BEARING", "HEALTHY", "BENIGN"}
TRAIN_USE_STATUSES = {"MODEL_RESEARCH_ALLOWED", "MODEL_RESEARCH_COMMERCIAL_ALLOWED"}
FALSE_VALUES = {"", "0", "false", "no", "n"}
UNKNOWN_TOKENS = {"", "unknown", "unspecified", "pending", "n/a", "na", "none"}
DIRECT_IDENTIFIER_COLUMNS = {
    "name", "full_name", "patient_name", "email", "phone", "telephone",
    "date_of_birth", "dob", "medical_record_number", "mrn", "national_id",
}


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() not in FALSE_VALUES


def _known(value: object) -> bool:
    return str(value).strip().lower() not in UNKNOWN_TOKENS


def _require_consistent_subject_metadata(frame: pd.DataFrame) -> None:
    fields = [
        "species", "label", "label_provenance", "split_group", "tlc_profile_id",
        "device_profile_id", "acquisition_profile_id", "data_use_status",
    ]
    for subject_id, group in frame.groupby("subject_id", sort=False):
        for field in fields:
            if group[field].astype(str).nunique(dropna=False) > 1:
                raise ValueError(f"subject {subject_id} has inconsistent {field}")


def validate_external_lct_intake(path: Path, *, allow_empty: bool = False) -> dict:
    frame = pd.read_csv(path, keep_default_na=False)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"missing intake columns: {missing}")

    identifiers = sorted(DIRECT_IDENTIFIER_COLUMNS.intersection({c.lower() for c in frame.columns}))
    if identifiers:
        raise ValueError(f"direct identifier columns are not allowed: {identifiers}")

    if frame.empty:
        if not allow_empty:
            raise ValueError("external intake manifest is empty")
        return {
            "status": "EMPTY_TEMPLATE",
            "rows": 0,
            "subjects": 0,
            "train_eligible_rows": 0,
            "clinical_claim": "NONE",
        }

    for field in ["subject_id", "image_path", "source_id", "modality", "species", "label_provenance", "split_group"]:
        if frame[field].astype(str).str.strip().eq("").any():
            raise ValueError(f"{field} must be populated for every row")

    if set(frame["modality"].astype(str)) != {"contact-LCT"}:
        raise ValueError("external native intake accepts contact-LCT rows only")

    bad_labels = sorted(set(frame["label"].astype(str)) - SUPPORTED_LABELS)
    if bad_labels:
        raise ValueError(f"unsupported labels: {bad_labels}")

    if frame["image_path"].duplicated().any():
        dup = frame.loc[frame["image_path"].duplicated(), "image_path"].tolist()
        raise ValueError(f"duplicate image paths: {dup[:20]}")

    _require_consistent_subject_metadata(frame)

    train_mask = frame["train_eligible"].map(_as_bool)
    train = frame[train_mask].copy()
    reasons: list[str] = []

    for idx, row in train.iterrows():
        prefix = f"row {idx} / subject {row['subject_id']}"
        if str(row["use_role"]).strip() != "TRAIN_CANDIDATE":
            reasons.append(f"{prefix}: train-eligible row must use TRAIN_CANDIDATE")
        if str(row["data_use_status"]).strip() not in TRAIN_USE_STATUSES:
            reasons.append(f"{prefix}: explicit model-research data-use permission required")
        if not _known(row["tlc_profile_id"]):
            reasons.append(f"{prefix}: TLC profile must be known before training")
        if not _known(row["device_profile_id"]):
            reasons.append(f"{prefix}: device profile must be known before training")
        if not _known(row["acquisition_profile_id"]):
            reasons.append(f"{prefix}: acquisition profile must be known before training")
        if not _known(row["license_tag"]):
            reasons.append(f"{prefix}: rights/license tag must be explicit")
        if len(str(row["label_provenance"]).strip()) < 8:
            reasons.append(f"{prefix}: label provenance is insufficient")

    if reasons:
        raise ValueError("unsafe train-eligible intake: " + "; ".join(reasons))

    return {
        "status": "GREEN",
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "train_eligible_rows": int(train_mask.sum()),
        "species": sorted(frame["species"].astype(str).unique().tolist()),
        "tlc_profiles": sorted(frame["tlc_profile_id"].astype(str).unique().tolist()),
        "device_profiles": sorted(frame["device_profile_id"].astype(str).unique().tolist()),
        "data_use_statuses": sorted(frame["data_use_status"].astype(str).unique().tolist()),
        "clinical_claim": "NONE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate_external_lct_intake(args.manifest, allow_empty=args.allow_empty), indent=2))


if __name__ == "__main__":
    main()
