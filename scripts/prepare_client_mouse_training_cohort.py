from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.validate_external_lct_intake import validate_external_lct_intake

POSITIVE_LABEL = "TUMOR_BEARING"
NEGATIVE_LABEL = "HEALTHY"
TARGET_TLC_PROFILE_ID = "client-device-tlc-pending"
FROZEN_EVAL_SUBJECTS = {f"CLIENT-MOUSE-{number:04d}" for number in range(30, 39)}
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


def assess_mouse_training_cohort(
    manifest_path: Path,
    *,
    min_per_class: int = 5,
    allow_empty: bool = False,
) -> tuple[dict, pd.DataFrame | None]:
    intake = validate_external_lct_intake(manifest_path, allow_empty=allow_empty)
    frame = pd.read_csv(manifest_path, keep_default_na=False)

    if frame.empty:
        return ({
            "status": "EMPTY_TEMPLATE",
            "binary_training_ready": False,
            "positive_subjects": 0,
            "negative_subjects": 0,
            "target_tlc_profile_id": TARGET_TLC_PROFILE_ID,
            "reasons": ["no new mouse training subjects supplied"],
            "clinical_claim": "NONE",
        }, None)

    missing = sorted(MOUSE_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"missing mouse cohort columns: {missing}")

    for field in DECLARED_MOUSE_FIELDS:
        if not _declared(frame, field):
            raise ValueError(f"{field} must be populated for every mouse row")

    if set(frame["species"].astype(str)) != {"mouse"}:
        raise ValueError("client mouse training cohort must contain mouse rows only")
    if set(frame["label"].astype(str)) - {POSITIVE_LABEL, NEGATIVE_LABEL}:
        raise ValueError("mouse native cohort labels must be TUMOR_BEARING or HEALTHY")
    if frame["subject_id"].duplicated().any():
        dup = frame.loc[frame["subject_id"].duplicated(), "subject_id"].tolist()
        raise ValueError(f"current lct-target-v1 mouse cohort requires one canonical image per animal; duplicates: {dup[:20]}")
    if set(frame["subject_id"]).intersection(FROZEN_EVAL_SUBJECTS):
        frozen = sorted(set(frame["subject_id"]).intersection(FROZEN_EVAL_SUBJECTS))
        raise ValueError(f"frozen client evaluation subjects cannot enter new training cohort: {frozen}")
    if (frame["split_group"].astype(str) != frame["subject_id"].astype(str)).any():
        raise ValueError("mouse split_group must equal subject_id for animal-level isolation")

    weights = pd.to_numeric(frame["weight_g"], errors="coerce")
    if weights.isna().any() or (weights <= 0).any():
        raise ValueError("weight_g must be a positive numeric value for every mouse")
    capture_order = pd.to_numeric(frame["capture_order"], errors="coerce")
    if capture_order.isna().any() or (capture_order < 1).any():
        raise ValueError("capture_order must be a positive integer-like value")
    if capture_order.duplicated().any():
        raise ValueError("capture_order must be unique within the supplied cohort")

    train = frame[frame["train_eligible"].map(_as_bool)].copy()
    reasons: list[str] = []
    if train.empty:
        reasons.append("no train_eligible new mouse subjects")
    if not train.empty and not (train["use_role"] == "TRAIN_CANDIDATE").all():
        reasons.append("all trainable mouse rows must use TRAIN_CANDIDATE")
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
            reasons.append(f"trainable pool must use one {description}; found {sorted(train[field].astype(str).unique().tolist())}")

    positive_subjects = int(train.loc[train["label"] == POSITIVE_LABEL, "subject_id"].nunique())
    negative_subjects = int(train.loc[train["label"] == NEGATIVE_LABEL, "subject_id"].nunique())
    if positive_subjects < min_per_class:
        reasons.append(
            f"need at least {min_per_class} NEW tumor-bearing training subjects; found {positive_subjects}; frozen nine are evaluation-only"
        )
    if negative_subjects < min_per_class:
        reasons.append(f"need at least {min_per_class} genuine control subjects; found {negative_subjects}")

    sessions_by_label = {
        label: sorted(group["capture_session_id"].astype(str).unique().tolist())
        for label, group in train.groupby("label")
    }
    if positive_subjects and negative_subjects:
        positive_sessions = set(sessions_by_label.get(POSITIVE_LABEL, []))
        negative_sessions = set(sessions_by_label.get(NEGATIVE_LABEL, []))
        if positive_sessions.isdisjoint(negative_sessions):
            reasons.append(
                "positive and control subjects have disjoint capture sessions; interleave classes across sessions to reduce session confounding"
            )

    ready = not reasons
    native_manifest = _native_subject_manifest(train) if ready else None
    result = {
        "status": "GREEN" if ready else "BLOCKED",
        "binary_training_ready": ready,
        "intake_status": intake["status"],
        "trainable_subjects": int(train["subject_id"].nunique()),
        "positive_subjects": positive_subjects,
        "negative_subjects": negative_subjects,
        "min_per_class": int(min_per_class),
        "target_tlc_profile_id": TARGET_TLC_PROFILE_ID,
        "frozen_eval_subjects_preserved": sorted(FROZEN_EVAL_SUBJECTS),
        "sessions_by_label": sessions_by_label,
        "reasons": reasons,
        "clinical_claim": "NONE",
    }
    return result, native_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--min-per-class", type=int, default=5)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--output-native-manifest", type=Path)
    args = parser.parse_args()

    result, native_manifest = assess_mouse_training_cohort(
        args.manifest,
        min_per_class=args.min_per_class,
        allow_empty=args.allow_empty,
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
