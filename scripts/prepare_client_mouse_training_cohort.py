from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.validate_external_lct_intake import validate_external_lct_intake

POSITIVE_LABEL = "TUMOR_BEARING"
NEGATIVE_LABEL = "HEALTHY"
TARGET_TLC_PROFILE_ID = "client-device-tlc-pending"
CLIENT_COHORT_SUBJECTS = {f"CLIENT-MOUSE-{number:04d}" for number in range(30, 39)}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIENT_MANIFEST = ROOT / "data" / "lct_target_dataset_v1_manifest.csv"
DEFAULT_CLASS_MAP = ROOT / "data" / "client_mouse_ground_truth_template.csv"
MOUSE_REQUIRED_COLUMNS = {
    "strain_id",
    "sex",
    "weight_g",
    "experimental_group",
    "capture_session_id",
    "tlc_batch_id",
    "camera_settings_id",
    "illumination_profile_id",
    "view_id",
    "capture_order",
    "matches_client_positive_domain",
}
DECLARED_MOUSE_FIELDS = [
    "strain_id",
    "sex",
    "experimental_group",
    "capture_session_id",
    "tlc_batch_id",
    "camera_settings_id",
    "illumination_profile_id",
    "view_id",
    "capture_order",
    "matches_client_positive_domain",
]


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _declared(frame: pd.DataFrame, field: str) -> bool:
    return not frame[field].astype(str).str.strip().eq("").any()


def _native_subject_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
        "notes",
    ]
    return frame[columns].copy().sort_values("subject_id").reset_index(drop=True)


def _load_client_mixed_cohort(manifest_path: Path, class_map_path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    required = {
        "subject_id", "image_path", "source_id", "modality", "species", "label",
        "label_provenance", "split_group", "license_tag", "tlc_profile_id",
        "device_profile_id", "use_role", "train_eligible", "notes",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"client manifest missing columns: {missing}")

    cohort = frame[frame["subject_id"].isin(CLIENT_COHORT_SUBJECTS)].copy()
    found = set(cohort["subject_id"].astype(str))
    if found != CLIENT_COHORT_SUBJECTS:
        raise ValueError(
            f"client cohort mismatch; missing={sorted(CLIENT_COHORT_SUBJECTS - found)}, "
            f"extra={sorted(found - CLIENT_COHORT_SUBJECTS)}"
        )
    if len(cohort) != len(CLIENT_COHORT_SUBJECTS) or cohort["subject_id"].duplicated().any():
        raise ValueError("client cohort must contain exactly one row for each of the nine subjects")
    if set(cohort["modality"].astype(str)) != {"contact-LCT"}:
        raise ValueError("client cohort must be contact-LCT")
    if set(cohort["species"].astype(str)) != {"mouse"}:
        raise ValueError("client cohort must be mouse")
    if set(cohort["tlc_profile_id"].astype(str)) != {TARGET_TLC_PROFILE_ID}:
        raise ValueError(f"client cohort must use {TARGET_TLC_PROFILE_ID}")
    if (cohort["split_group"].astype(str) != cohort["subject_id"].astype(str)).any():
        raise ValueError("client cohort split_group must equal subject_id")

    mapping = pd.read_csv(class_map_path, keep_default_na=False)
    required_map = {"subject_id", "image_id", "tumor_status"}
    missing_map = sorted(required_map - set(mapping.columns))
    if missing_map:
        raise ValueError(f"client class map missing columns: {missing_map}")
    mapping = mapping[mapping["subject_id"].isin(CLIENT_COHORT_SUBJECTS)].copy()
    if set(mapping["subject_id"].astype(str)) != CLIENT_COHORT_SUBJECTS or mapping["subject_id"].duplicated().any():
        raise ValueError("client class map must contain exactly one row for every client mouse subject")

    mapping["tumor_status"] = mapping["tumor_status"].astype(str).str.strip().str.upper()
    valid = {POSITIVE_LABEL, NEGATIVE_LABEL}
    pending = sorted(mapping.loc[~mapping["tumor_status"].isin(valid), "subject_id"].astype(str).tolist())

    cohort = cohort.drop(columns=["label"], errors="ignore").merge(
        mapping[["subject_id", "tumor_status"]], on="subject_id", how="left", validate="one_to_one"
    )
    cohort = cohort.rename(columns={"tumor_status": "label"})
    cohort["label_provenance"] = cohort.apply(
        lambda row: (
            "Client-supplied per-image mouse class map: experimentally tumor-bearing"
            if row["label"] == POSITIVE_LABEL else
            "Client-supplied per-image mouse class map: normal/no-tumor control"
            if row["label"] == NEGATIVE_LABEL else
            "Client states supplied cohort contains tumor-bearing and normal/no-tumor mice; exact per-image class pending"
        ), axis=1
    )
    cohort["use_role"] = "TRAIN_CANDIDATE" if not pending else "FROZEN_TARGET_EVAL"
    cohort["train_eligible"] = not pending
    cohort["notes"] = cohort["notes"].astype(str).map(
        lambda value: (
            value
            + "; internal research cross-validation only; not independent external validation"
        ).strip("; ")
    )
    return cohort, pending


def assess_mouse_training_cohort(
    manifest_path: Path,
    *,
    min_per_class: int = 2,
    allow_empty: bool = False,
    client_manifest_path: Path = DEFAULT_CLIENT_MANIFEST,
    class_map_path: Path = DEFAULT_CLASS_MAP,
) -> tuple[dict, pd.DataFrame | None]:
    intake = validate_external_lct_intake(manifest_path, allow_empty=allow_empty)
    supplemental = pd.read_csv(manifest_path, keep_default_na=False)
    client, pending = _load_client_mixed_cohort(client_manifest_path, class_map_path)

    mapped_positive = int((client["label"] == POSITIVE_LABEL).sum())
    mapped_negative = int((client["label"] == NEGATIVE_LABEL).sum())
    if pending:
        return ({
            "status": "CLASS_MAP_PENDING",
            "binary_training_ready": False,
            "client_subjects": int(len(client)),
            "mapped_positive_subjects": mapped_positive,
            "mapped_negative_subjects": mapped_negative,
            "pending_subjects": pending,
            "target_tlc_profile_id": TARGET_TLC_PROFILE_ID,
            "validation_scope": "INTERNAL_RESEARCH_CROSS_VALIDATION",
            "independent_external_validation": False,
            "reasons": [
                "client clarified that the supplied nine-image cohort contains both tumor-bearing and normal/no-tumor mice; exact per-image class mapping is required before training"
            ],
            "warnings": [
                "do not request replacement controls solely because the cohort was previously misclassified as positive-only",
                "the original Kaggle DINO run intentionally used labels_seen=false and therefore does not contain the missing class map",
            ],
            "clinical_claim": "NONE",
        }, None)

    if supplemental.empty:
        train = supplemental.copy()
    else:
        missing = sorted(MOUSE_REQUIRED_COLUMNS - set(supplemental.columns))
        if missing:
            raise ValueError(f"missing mouse cohort columns: {missing}")
        for field in DECLARED_MOUSE_FIELDS:
            if not _declared(supplemental, field):
                raise ValueError(f"{field} must be populated for every supplemental mouse row")
        if set(supplemental["species"].astype(str)) != {"mouse"}:
            raise ValueError("supplemental client cohort must contain mouse rows only")
        if set(supplemental["label"].astype(str)) - {POSITIVE_LABEL, NEGATIVE_LABEL}:
            raise ValueError("mouse native cohort labels must be TUMOR_BEARING or HEALTHY")
        if supplemental["subject_id"].duplicated().any():
            dup = supplemental.loc[supplemental["subject_id"].duplicated(), "subject_id"].tolist()
            raise ValueError(f"current lct-target-v1 mouse cohort requires one canonical image per animal; duplicates: {dup[:20]}")
        overlap = set(supplemental["subject_id"]).intersection(CLIENT_COHORT_SUBJECTS)
        if overlap:
            raise ValueError(f"client cohort subjects must not be duplicated in supplemental rows: {sorted(overlap)}")
        if (supplemental["split_group"].astype(str) != supplemental["subject_id"].astype(str)).any():
            raise ValueError("mouse split_group must equal subject_id for animal-level isolation")

        weights = pd.to_numeric(supplemental["weight_g"], errors="coerce")
        if weights.isna().any() or (weights <= 0).any():
            raise ValueError("weight_g must be a positive numeric value for every supplemental mouse")
        capture_order = pd.to_numeric(supplemental["capture_order"], errors="coerce")
        if capture_order.isna().any() or (capture_order < 1).any():
            raise ValueError("capture_order must be a positive integer-like value")
        if capture_order.duplicated().any():
            raise ValueError("capture_order must be unique within the supplied supplemental cohort")
        train = supplemental[supplemental["train_eligible"].map(_as_bool)].copy()

    reasons: list[str] = []
    warnings = [
        "the nine client images are an internal development cohort, not an independent external validation set",
        "historical strain/sex/weight/session metadata are incomplete; report this limitation with the first model",
    ]

    if not train.empty:
        if not (train["use_role"] == "TRAIN_CANDIDATE").all():
            reasons.append("all trainable supplemental mouse rows must use TRAIN_CANDIDATE")
        if not train["matches_client_positive_domain"].map(_as_bool).all():
            reasons.append("every trainable supplemental row must explicitly confirm matches_client_positive_domain=true")
        if set(train["tlc_profile_id"].astype(str)) != {TARGET_TLC_PROFILE_ID}:
            reasons.append(
                f"lct-target-v1 is currently profile-locked to {TARGET_TLC_PROFILE_ID}; "
                f"found {sorted(train['tlc_profile_id'].astype(str).unique().tolist())}"
            )
        for field, description in [
            ("tlc_profile_id", "TLC profile"),
            ("device_profile_id", "device profile"),
            ("acquisition_profile_id", "acquisition profile"),
            ("strain_id", "mouse strain"),
            ("sex", "mouse sex"),
            ("tlc_batch_id", "TLC batch"),
            ("camera_settings_id", "camera settings"),
            ("illumination_profile_id", "illumination profile"),
            ("view_id", "canonical view"),
        ]:
            if train[field].nunique() != 1:
                reasons.append(
                    f"trainable supplemental pool must use one {description}; "
                    f"found {sorted(train[field].astype(str).unique().tolist())}"
                )

    combined = pd.concat([client, train], ignore_index=True, sort=False)
    positive_subjects = int(combined.loc[combined["label"] == POSITIVE_LABEL, "subject_id"].nunique())
    negative_subjects = int(combined.loc[combined["label"] == NEGATIVE_LABEL, "subject_id"].nunique())
    if positive_subjects < min_per_class:
        reasons.append(f"need at least {min_per_class} mapped tumor-bearing subjects; found {positive_subjects}")
    if negative_subjects < min_per_class:
        reasons.append(f"need at least {min_per_class} mapped normal/no-tumor subjects; found {negative_subjects}")

    if min(positive_subjects, negative_subjects) < 5:
        warnings.append(
            "class counts are very small; treat metrics as exploratory internal research only and prefer repeated/leave-subject-out sensitivity analysis"
        )

    promoted_device_profile = None
    if not train.empty:
        promoted_device_profile = str(train["device_profile_id"].iloc[0])
        client = client.copy()
        client["device_profile_id"] = promoted_device_profile
        combined = pd.concat([client, train], ignore_index=True, sort=False)

    ready = not reasons
    native_manifest = _native_subject_manifest(combined) if ready else None
    result = {
        "status": "GREEN" if ready else "BLOCKED",
        "binary_training_ready": ready,
        "intake_status": intake["status"],
        "trainable_subjects": int(positive_subjects + negative_subjects),
        "positive_subjects": positive_subjects,
        "negative_subjects": negative_subjects,
        "existing_client_subjects": int(len(client)),
        "supplemental_trainable_subjects": int(len(train)),
        "min_per_class": int(min_per_class),
        "target_tlc_profile_id": TARGET_TLC_PROFILE_ID,
        "promoted_device_profile_id": promoted_device_profile,
        "validation_scope": "INTERNAL_RESEARCH_CROSS_VALIDATION",
        "independent_external_validation": False,
        "reasons": reasons,
        "warnings": warnings,
        "clinical_claim": "NONE",
    }
    return result, native_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--min-per-class", type=int, default=2)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--client-manifest", type=Path, default=DEFAULT_CLIENT_MANIFEST)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--output-native-manifest", type=Path)
    args = parser.parse_args()

    result, native_manifest = assess_mouse_training_cohort(
        args.manifest,
        min_per_class=args.min_per_class,
        allow_empty=args.allow_empty,
        client_manifest_path=args.client_manifest,
        class_map_path=args.class_map,
    )
    if args.output_native_manifest is not None:
        if native_manifest is None:
            raise ValueError("mouse native training gate is closed; refusing to write a trainable manifest")
        args.output_native_manifest.parent.mkdir(parents=True, exist_ok=True)
        native_manifest.to_csv(args.output_native_manifest, index=False)
        result["output_native_manifest"] = str(args.output_native_manifest)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
