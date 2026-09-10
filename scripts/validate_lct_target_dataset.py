from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED = {
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
    "use_role",
    "train_eligible",
}

ALLOWED_ROLES = {"REFERENCE_ONLY", "FROZEN_TARGET_EVAL", "TRAIN_CANDIDATE", "BLOCKED_RIGHTS"}
POSITIVE_LABELS = {"CANCER", "MALIGNANT", "TUMOR_BEARING"}
NEGATIVE_LABELS = {"HEALTHY", "BENIGN"}


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def validate(path: Path) -> dict:
    frame = pd.read_csv(path, keep_default_na=False)
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if frame.empty:
        raise ValueError("target dataset manifest is empty")

    errors: list[str] = []
    warnings: list[str] = []

    if set(frame["modality"]) != {"contact-LCT"}:
        errors.append("target dataset v1 may contain contact-LCT rows only")

    bad_roles = sorted(set(frame["use_role"]) - ALLOWED_ROLES)
    if bad_roles:
        errors.append(f"unsupported use_role values: {bad_roles}")

    duplicated = frame[frame["subject_id"].duplicated()]["subject_id"].tolist()
    if duplicated:
        errors.append(f"duplicate target subject IDs: {duplicated[:20]}")

    if (frame["tlc_profile_id"].astype(str).str.strip() == "").any():
        errors.append("every contact-LCT row requires tlc_profile_id")

    frame = frame.copy()
    frame["_train"] = frame["train_eligible"].map(_as_bool)

    illegal_reference_train = frame[(frame["use_role"] != "TRAIN_CANDIDATE") & frame["_train"]]
    if not illegal_reference_train.empty:
        errors.append("only TRAIN_CANDIDATE rows may set train_eligible=true")

    private_negatives = frame[
        (frame["source_id"] == "client-mice-2026") & frame["label"].isin(NEGATIVE_LABELS)
    ]
    if not private_negatives.empty:
        errors.append("client mouse cohort is known positive-only; negative labels are forbidden")

    train = frame[frame["_train"]]
    train_pos = int(train["label"].isin(POSITIVE_LABELS).sum())
    train_neg = int(train["label"].isin(NEGATIVE_LABELS).sum())
    binary_training_ready = train_pos >= 5 and train_neg >= 5 and train["species"].nunique() == 1

    if not binary_training_ready:
        warnings.append(
            "target-domain binary training is NOT ready: require at least 5 eligible positives and 5 eligible negatives from one species before fitting even a provisional classifier"
        )

    frozen = frame[frame["use_role"] == "FROZEN_TARGET_EVAL"]
    if not frozen.empty and frozen["split_group"].nunique() != 1:
        errors.append("frozen target cohort must remain in one split_group")

    summary = {
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "species": frame["species"].value_counts().to_dict(),
        "labels": frame["label"].value_counts().to_dict(),
        "roles": frame["use_role"].value_counts().to_dict(),
        "train_eligible_rows": int(train.shape[0]),
        "train_positive_rows": train_pos,
        "train_negative_rows": train_neg,
        "binary_training_ready": bool(binary_training_ready),
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
        "clinical_claim": "NONE",
    }
    if errors:
        raise ValueError(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    result = validate(args.manifest)
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
