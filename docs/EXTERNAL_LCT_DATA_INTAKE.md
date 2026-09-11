# External contact-LCT data intake contract

This lane sits between external data acquisition and the existing native `TUMOR_LIKE / NO_TUMOR_LIKE` trainer. Its purpose is to make it impossible to treat an emailed folder of images as train-ready data without explicit provenance, rights and subject grouping.

## Required manifest

Use `data/external_lct_intake_template.csv` and validate it with:

```bash
python scripts/validate_external_lct_intake.py <manifest.csv>
```

Required fields:

- `subject_id` — stable de-identified subject/animal ID; no names, emails, phone numbers, DOB, MRN or national identifiers.
- `image_path` — one unique path/reference per image.
- `source_id` — study/cohort/source identifier.
- `modality` — must be exactly `contact-LCT` for this native intake path.
- `species` — e.g. `human` or `mouse`.
- `label` — supported ground-truth class (`MALIGNANT`, `CANCER`, `TUMOR_BEARING`, `BENIGN`, `HEALTHY`).
- `label_provenance` — how the class is known (histopathology, experimentally induced tumour/control assignment, documented follow-up, etc.).
- `split_group` — subject/animal grouping key. Multiple views from one subject must share the same group.
- `license_tag` — explicit rights/agreement tag.
- `tlc_profile_id` — TLC formulation/foil profile.
- `device_profile_id` — acquisition device/camera profile.
- `acquisition_profile_id` — protocol/session profile (foil range, view, illumination/acclimatisation protocol where applicable).
- `data_use_status` — explicit permission status.
- `redistribution_status` — whether raw data may be redistributed or must remain private.
- `use_role` — default should be reference/evaluation; only explicit candidates use `TRAIN_CANDIDATE`.
- `train_eligible` — defaults false. Setting true activates stricter checks; it does not override the native training readiness gate.
- `notes` — additional source restrictions or caveats.

## Train-candidate requirements

A row marked `train_eligible=true` is rejected unless all of the following are true:

1. `use_role=TRAIN_CANDIDATE`.
2. `data_use_status` explicitly allows model research (`MODEL_RESEARCH_ALLOWED` or `MODEL_RESEARCH_COMMERCIAL_ALLOWED`).
3. TLC, device and acquisition profiles are known rather than `unknown`, `pending` or `unspecified`.
4. rights/license status is explicit.
5. ground-truth provenance is substantive.
6. the row remains contact-LCT and uses a supported biological label.

This intake gate is intentionally separate from `scripts/train_lct_native_binary.py`. Even a manifest that passes intake can still fail the trainer's readiness rules because the native trainable pool must be coherent by species/TLC/device domain and contain genuine positive and negative subjects.

## Subject consistency

Multiple views are allowed for a subject, but the validator requires the same subject to keep a stable species, label, label provenance, split group, TLC profile, device profile, acquisition profile and data-use status. This prevents left/right or repeated views from leaking into different labels or splits.

## Privacy rule

The intake manifest is designed for de-identified research data. Direct identifier columns such as patient name, email, phone, date of birth, medical-record number or national ID are rejected. We do not need those fields for model development.

## Current project boundary

The nine client mouse images remain `FROZEN_TARGET_EVAL`, positive-only, and not train eligible. Open-license human case figures remain `REFERENCE_ONLY`. This intake layer does not synthesize negative labels and does not open the mouse binary training gate.

`clinical_claim=NONE` remains mandatory.
