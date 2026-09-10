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
}

ALLOWED_MODALITIES = {"contact-LCT", "infrared"}
ALLOWED_LABELS = {"HEALTHY", "CANCER", "BENIGN", "MALIGNANT", "TUMOR_BEARING", "UNKNOWN"}


def validate_manifest(path: Path) -> dict:
    frame = pd.read_csv(path, keep_default_na=False)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Manifest is empty")

    errors: list[str] = []
    warnings: list[str] = []

    for column in ("subject_id", "image_path", "source_id", "modality", "species", "label", "label_provenance", "split_group", "license_tag"):
        if (frame[column].astype(str).str.strip() == "").any():
            errors.append(f"blank required value in {column}")

    bad_modality = sorted(set(frame["modality"]) - ALLOWED_MODALITIES)
    if bad_modality:
        errors.append(f"unsupported modality values: {bad_modality}")

    bad_labels = sorted(set(frame["label"]) - ALLOWED_LABELS)
    if bad_labels:
        errors.append(f"unsupported label values: {bad_labels}")

    # Every subject must stay in one split group. Multiple views/sequences of a
    # subject are allowed, but leakage across groups is not.
    subject_groups = frame.groupby("subject_id")["split_group"].nunique()
    leaked_subjects = subject_groups[subject_groups > 1].index.tolist()
    if leaked_subjects:
        errors.append(f"subjects assigned to multiple split groups: {leaked_subjects[:20]}")

    # Subject labels must be internally consistent within one source.
    label_counts = frame.groupby(["source_id", "subject_id"])["label"].nunique()
    inconsistent = label_counts[label_counts > 1].index.tolist()
    if inconsistent:
        errors.append(f"subjects with inconsistent labels: {inconsistent[:20]}")

    # Contact-LCT rows must identify their TLC profile; this prevents silently
    # mixing formulations/devices into one target-domain pool.
    lct = frame[frame["modality"] == "contact-LCT"]
    if not lct.empty and (lct["tlc_profile_id"].astype(str).str.strip() == "").any():
        errors.append("contact-LCT rows require tlc_profile_id")

    # Warn, rather than fail, when a license needs a human decision.
    risky = frame[frame["license_tag"].str.contains("NC|UNKNOWN|REVIEW", case=False, regex=True)]
    if not risky.empty:
        warnings.append(
            f"{len(risky)} rows have noncommercial/unknown/review-required license tags; do not use them in a commercial model without review"
        )

    summary = {
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "sources": sorted(frame["source_id"].unique().tolist()),
        "modalities": frame["modality"].value_counts().to_dict(),
        "labels": frame["label"].value_counts().to_dict(),
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

    summary = validate_manifest(args.manifest)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
