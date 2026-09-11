# Client mouse native training cohort acquisition

Purpose: define the shortest valid route from the current client mouse set to a first target-domain contact-LCT binary research baseline without synthetic negatives, leakage or false independent-validation claims.

## Revised development-positive policy

The original nine client subjects (`CLIENT-MOUSE-0030` through `CLIENT-MOUSE-0038`) are genuine tumor-bearing mouse positives from the client's experiment. The client states that all nine animals were tumor-injected and captured under the same experimental conditions, with broadly similar weights and tumor size as the main biological difference.

The canonical target manifest remains immutable/non-trainable source evidence, but these nine subjects may now be **promoted by the cohort-preparation step into the internal development training manifest** once genuine controls pass the target-domain gate.

This means **new tumor-bearing mice are no longer required to open the first internal binary baseline**. The minimum engineering gate is now:

- the existing 9 client tumor-bearing development positives; plus
- at least 5 genuine same-domain control/negative mouse subjects.

More controls are strongly preferred; roughly 9 or more would make the first class balance much better.

This policy changes the evaluation claim. Because the original nine have already informed development, the first binary result is:

`INTERNAL_RESEARCH_CROSS_VALIDATION`

It is **not** independent external validation and must not be presented as such.

## What counts as a valid control

This repository does not prescribe new animal interventions or modify the client's study design. Use only controls already available under the client's approved experimental/ethics protocol:

- untreated/control animals already defined by that protocol; or
- sham controls only if a sham group already exists under that approved protocol.

Do not create negative labels from image appearance. Ground truth comes from experimental group assignment/documentation, not from the thermogram.

## Same-domain acquisition requirement

The new control images need to match the nine client positives as closely as the real experiment allows. Keep fixed or explicitly versioned:

- the same contact-LCT modality;
- TLC formulation/profile and preferably the same TLC batch;
- the same physical device/camera setup;
- camera exposure/white-balance/settings profile;
- illumination source/profile;
- acquisition geometry and canonical view;
- contact placement/pressure procedure and contact duration;
- ambient/acclimatisation procedure;
- operator procedure;
- image encoding/export path.

Contact-LCT colour is not a universal RGB temperature code. TLC colour response depends on the crystal formulation/profile and acquisition optics/illumination/viewing geometry, so cross-device or cross-foil controls can become a confound instead of a biological negative class.

Every supplemental training row must explicitly set `matches_client_positive_domain=true`. This is an acquisition/data-curation attestation, not a statement that the exact TLC calibration is already known. The current 417-feature contract therefore remains locked to `client-device-tlc-pending` until the client's exact TLC formulation/calibration is documented.

## Historical metadata limitation

The original nine positives do not currently carry complete strain/sex/weight/capture-session metadata in the canonical manifest. Their tumor-bearing status and shared experimental context are known, but some detailed acquisition fields were not captured when the source cohort was first ingested.

Therefore:

- controls must be documented as carefully as possible;
- matching strain/sex/weight range should be used when available under the existing protocol;
- this missing historical metadata must remain an explicit limitation in the first baseline report;
- the first run must not be described as independent validation.

## Avoid acquisition confounding

If new positive animals later become available, do not capture every positive in one session and every control in another. Session drift, lighting changes, TLC batch changes, camera settings and operator differences can become easier classification signals than biology.

For controls-only supplementation, legacy positive session IDs are unavailable, so session confounding against the original nine cannot be fully audited. Record control session IDs and keep the control capture protocol as close as possible to the original experiment.

## Subject and view contract

The current `lct-target-v1` 417-feature builder uses one canonical image per subject for the first native mouse baseline:

- one stable de-identified `subject_id` per animal;
- `split_group=subject_id`;
- one consistent canonical `view_id`;
- one canonical training image per animal in the v1 manifest.

Extra views may be archived privately for later multi-view work, but must not silently create duplicate subject rows.

## Metadata to record for controls

Start from `data/client_mouse_training_cohort_template.csv`. Record:

- `strain_id`;
- `sex`;
- `weight_g`;
- `experimental_group`;
- `capture_session_id`;
- `tlc_batch_id`;
- `camera_settings_id`;
- `illumination_profile_id`;
- `view_id`;
- `capture_order`;
- `matches_client_positive_domain=true` only when the curator has confirmed the control belongs to the same client target-domain setup.

For the first baseline, the supplemental trainable pool is deliberately strict: one strain, one TLC profile/batch, one device profile, one acquisition profile, one camera-settings profile, one illumination profile and one canonical view.

## Validation and bridge

Validate the supplemental cohort with:

```bash
python scripts/prepare_client_mouse_training_cohort.py \
  /path/to/private/client_mouse_training_manifest.csv
```

The script validates the supplemental rows, loads the nine canonical tumor-bearing positives from `data/lct_target_dataset_v1_manifest.csv`, and reports readiness.

With at least five valid controls the gate can return `binary_training_ready=true` without requiring any new tumor-bearing animals. Generate the combined internal-development manifest with:

```bash
python scripts/prepare_client_mouse_training_cohort.py \
  /path/to/private/client_mouse_training_manifest.csv \
  --output-native-manifest runtime/client-mouse-native/native_manifest.csv
```

The generated manifest promotes copies of the nine canonical positives to `TRAIN_CANDIDATE` **only for this internal research run**. The canonical source manifest itself stays unchanged and non-trainable.

Then run the 417-feature builder and `scripts/train_lct_native_binary.py`. Evaluation remains animal/subject-level and uses internal cross-validation.

## What this baseline can and cannot prove

It can answer whether, within the small client mouse dataset, the target-domain contact-LCT feature representation contains reproducible signal that separates documented tumor-bearing animals from documented controls.

It cannot establish independent clinical diagnostic performance, cancer probability, human breast-cancer performance, or external validation.

A later held-out mixed positive/control cohort is still needed for an independent test.

## Parallel tumor-burden experiment

The nine positives also remain useful for a separate experiment because the client reports different tumor sizes across otherwise similar experimental conditions. If the client supplies the tumor-size/volume mapping for `WA0030` through `WA0038`, run the existing tumor-burden validation path separately from the binary classifier.

Do not infer tumor sizes from thermogram appearance.

## Privacy / repository rule

Do not commit raw client animal images or private experimental records by default. Commit schemas, validators and non-sensitive aggregate artifacts unless the client explicitly authorizes otherwise.

`clinical_claim=NONE` remains mandatory. A research classifier is not a diagnosis or cancer probability.
