from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.validate_external_lct_intake import validate_external_lct_intake

POSITIVE_LABEL = "TUMOR_BEARING"
NEGATIVE_LABEL = "HEALTHY"
TARGET_TLC_PROFILE_ID = "client-device-tlc-pending"
CLIENT_DEVELOPMENT_POSITIVES = {f"CLIENT-MOUSE-{number:04d}" for number in range(30, 39)}
DEFAULT_POSITIVE_MANIFEST = Path(__file__).resolve().parents[1] / "data" / "lct_target_dataset_v1_manifest.csv"
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


def _load_client_development_positives(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
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
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"development-positive manifest missing columns: {missing}")

    positive = frame[frame["subject_id"].isin(CLIENT_DEVELOPMENT_POSITIVES)].copy()
    found = set(positive["subject_id"].astype(str))
    if found != CLIENT_DEVELOPMENT_POSITIVES:
        missing_subjects = sorted(CLIENT_DEVELOPMENT_POSITIVES - found)
        extra_subjects = sorted(found - CLIENT_DEVELOPMENT_POSITIVES)
        raise ValueError(
            f"client development-positive manifest mismatch; missing={missing_subjects}, extra={extra_subjects}"
        )
    if len(positive) != len(CLIENT_DEVELOPMENT_POSITIVES) or positive["subject_id"].duplicated().any():
        raise ValueError("client development-positive manifest must contain exactly one row for each of the nine subjects")
    if set(positive["modality"].astype(str)) != {"contact-LCT"}:
        raise ValueError("client development positives must be contact-LCT")
    if set(positive["species"].astype(str)) != {"mouse"}:
        raise ValueError("client development positives must be mouse")
    if set(positive["label"].astype(str)) != {POSITIVE_LABEL}:
        raise ValueError("client development positives must remain TUMOR_BEARING")
    if set(positive["tlc_profile_id"].astype(str)) != {TARGET_TLC_PROFILE_ID}:
        raise ValueError(f"client development positives must use {TARGET_TLC_PROFILE_ID}")
    if (positive["split_group"].astype(str) != positive["subject_id"].astype(str)).any():
        raise ValueError("client development-positive split_group must equal subject_id")

    # The canonical manifest remains immutable/non-trainable. Promotion happens only
    # in the generated internal-development manifest after genuine controls pass.
    positive["use_role"] = "TRAIN_CANDIDATE"
    positive["train_eligible"] = True
    positive["notes"] = positive["notes"].astype(str).map(
        lambda value: (
            value
            + "; promoted only for internal research cross-validation; not independent external validation"
        ).strip("; ")
    )
    return positive


def assess_mouse_training_cohort(
    manifest_path: Path,
    *,
    min_per_class: int = 5,
    allow_empty: bool = False,
    positive_manifest_path: Path = DEFAULT_POSITIVE_MANIFEST,
) -> tuple[dict, pd.DataFrame | None]:
    intake = validate_external_lct_intake(manifest_path, allow_empty=allow_empty)
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    development_positive = _load_client_development_positives(positive_manifest_path)

    if frame.empty:
        return ({
            "status": "EMPTY_TEMPLATE",
            "binary_training_ready": False,
            "positive_subjects": int(len(development_positive)),
            "negative_subjects": 0,
            "min_control_subjects": int(min_per_class),
            "target_tlc_profile_id": TARGET_TLC_PROFILE_ID,
            "development_positive_subjects": sorted(CLIENT_DEVELOPMENT_POSITIVES),
            "validation_scope": "INTERNAL_RESEARCH_CROSS_VALIDATION",
            "independent_external_validation": False,
            "reasons": [f"need at least {min_per_class} genuine same-domain control subjects; found 0"],
            "warnings": [
                "the nine client tumor-bearing mice are development positives, not an independent test set",
                "historical positive strain/sex/weight/session metadata are incomplete; document matching controls carefully",
            ],
            "clinical_claim": "NONE",
        }, None)

    missing = sorted(MOUSE_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"missing mouse cohort columns: {missing}")

    for field in DECLARED_MOUSE_FIELDS:
        if not _declared(frame, field):
            raise ValueError(f"{field} must be populated for every mouse row")

    if set(frame["species"].astype(str)) != {"mouse"}:
        raise ValueError("client mouse supplemental cohort must contain mouse rows only")
    if set(frame["label"].astype(str)) - {POSITIVE_LABEL, NEGATIVE_LABEL}:
        raise ValueError("mouse native cohort labels must be TUMOR_BEARING or HEALTHY")
    if frame["subject_id"].duplicated().any():
        dup = frame.loc[frame["subject_id"].duplicated(), "subject_id"].tolist()
        raise ValueError(f"current lct-target-v1 mouse cohort requires one canonical image per animal; duplicates: {dup[:20]}")
    if set(frame["subject_id"]).intersection(CLIENT_DEVELOPMENT_POSITIVES):
        duplicate = sorted(set(frame["subject_id"]).intersection(CLIENT_DEVELOPMENT_POSITIVES))
        raise ValueError(
            "the nine canonical client development positives are injected automatically; "
            f"do not duplicate them in the supplemental manifest: {duplicate}"
        )
    if (frame["split_group"].astype(str) != frame["subject_id"].astype(str)).any():
        raise ValueError("mouse split_group must equal subject_id for animal-level isolation")

    weights = pd.to_numeric(frame["weight_g"], errors="coerce")
    if weights.isna().any() or (weights <= 0).any():
        raise ValueError("weight_g must be a positive numeric value for every supplemental mouse")
    capture_order = pd.to_numeric(frame["capture_order"], errors="coerce")
    if capture_order.isna().any() or (capture_order < 1).any():
        raise ValueError("capture_order must be a positive integer-like value")
    if capture_order.duplicated().any():
        raise ValueError("capture_order must be unique within the supplied supplemental cohort")

    train = frame[frame["train_eligible"].map(_as_bool)].copy()
    reasons: list[str] = []
    warnings = [
        "the nine client tumor-bearing mice are reused for internal development cross-validation and are not an independent test set",
        "historical positive strain/sex/weight/session metadata are incomplete; same-cohort matching cannot be fully audited from the original nine alone",
    ]
    if train.empty:
        reasons.append("no train_eligible supplemental mouse subjects")
    if not train.empty and not (train["use_role"] == "TRAIN_CANDIDATE").all():
        reasons.append("all trainable supplemental mouse rows must use TRAIN_CANDIDATE")
    if not train.empty and not train["matches_client_positive_domain"].map(_as_bool).all():
        reasons.append("every trainable supplemental row must explicitly confirm matches_client_positive_domain=true")
    if not train.empty and set(train["tlc_profile_id"].astype(str)) != {TARGET_TLC_PROFILE_ID}:
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
        if not train.empty and train[field].nunique() != 1:
            reasons.append(
                f"trainable supplemental pool must use one {description}; "
                f"found {sorted(train[field].astype(str).unique().tolist())}"
            )

    incoming_positive_subjects = int(train.loc[train["label"] == POSITIVE_LABEL, "subject_id"].nunique())
    negative_subjects = int(train.loc[train["label"] == NEGATIVE_LABEL, "subject_id"].nunique())
    positive_subjects = int(len(development_positive) + incoming_positive_subjects)
    if negative_subjects < min_per_class:
        reasons.append(f"need at least {min_per_class} genuine same-domain control subjects; found {negative_subjects}")

    sessions_by_label = {
        label: sorted(group["capture_session_id"].astype(str).unique().tolist())
        for label, group in train.groupby("label")
    }
    if incoming_positive_subjects and negative_subjects:
        positive_sessions = set(sessions_by_label.get(POSITIVE_LABEL, []))
        negative_sessions = set(sessions_by_label.get(NEGATIVE_LABEL, []))
        if positive_sessions.isdisjoint(negative_sessions):
            reasons.append(
                "new positive and control subjects have disjoint capture sessions; "
                "interleave classes across sessions to reduce session confounding"
            )
    elif negative_subjects:
        warnings.append(
            "legacy positive capture-session IDs are not available, so session confounding against the nine positives "
            "must be documented as a limitation"
        )

    ready = not reasons
    native_manifest = None
    promoted_device_profile = None
    if ready:
        promoted_device_profile = str(train["device_profile_id"].iloc[0])
        promoted = development_positive.copy()
        promoted["device_profile_id"] = promoted_device_profile
        native_manifest = _native_subject_manifest(pd.concat([promoted, train], ignore_index=True, sort=False))

    result = {
        "status": "GREEN" if ready else "BLOCKED",
        "binary_training_ready": ready,
        "intake_status": intake["status"],
        "trainable_subjects": int(positive_subjects + negative_subjects),
        "positive_subjects": positive_subjects,
        "existing_development_positive_subjects": int(len(development_positive)),
        "incoming_positive_subjects": incoming_positive_subjects,
        "negative_subjects": negative_subjects,
        "min_control_subjects": int(min_per_class),
        "target_tlc_profile_id": TARGET_TLC_PROFILE_ID,
        "development_positive_subjects": sorted(CLIENT_DEVELOPMENT_POSITIVES),
        "promoted_device_profile_id": promoted_device_profile,
        "sessions_by_label": sessions_by_label,
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
    parser.add_argument("--min-per-class", type=int, default=5)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--positive-manifest", type=Path, default=DEFAULT_POSITIVE_MANIFEST)
    parser.add_argument("--output-native-manifest", type=Path)
    args = parser.parse_args()

    result, native_manifest = assess_mouse_training_cohort(
        args.manifest,
        min_per_class=args.min_per_class,
        allow_empty=args.allow_empty,
        positive_manifest_path=args.positive_manifest,
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
