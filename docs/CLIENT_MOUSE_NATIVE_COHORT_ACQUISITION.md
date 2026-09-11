# Client mouse native training cohort acquisition

Purpose: define the shortest valid route from the current positive-only client mouse set to a trainable target-domain contact-LCT cohort without synthetic negatives, leakage or acquisition confounding.

## The important freeze constraint

The original nine client images (`CLIENT-MOUSE-0030` through `CLIENT-MOUSE-0038`) are already preserved as `FROZEN_TARGET_EVAL`. They must remain untouched evaluation evidence and must not be recycled as training positives.

Therefore **new controls alone are not sufficient** to open native binary training. The current software readiness gate needs at least:

- 5 **new** tumor-bearing mouse subjects eligible for training; and
- 5 genuine control/negative mouse subjects eligible for training.

That 5 + 5 threshold is only a minimum engineering gate for a first research baseline, not a claim of statistically adequate validation. More balanced subjects are strongly preferred.

## What animals to use

This repository does not prescribe new animal interventions or modify the client's study design. Use only animals/groups already available under the client's approved experimental/ethics protocol:

- tumor-bearing animals from the existing approved tumor protocol for the new positive training pool;
- untreated/control animals already defined by that protocol for the negative pool;
- sham controls only if a sham group already exists under the approved protocol.

Do not create labels from image appearance. Ground truth comes from experimental group assignment/documentation, not from the thermogram.

## Same-domain acquisition requirement

The point of this cohort is to train on the client's real target domain, so both classes should be captured under the same imaging domain wherever possible.

Keep fixed or explicitly versioned:

- TLC formulation/profile and TLC batch;
- device/camera;
- camera exposure/white-balance/settings profile;
- illumination source/profile;
- acquisition geometry and canonical view;
- contact placement/pressure procedure and contact duration used by the client system;
- ambient/acclimatisation procedure used by the existing client protocol;
- operator procedure;
- image encoding/export path.

Contact-LCT colour response is sensitive to foil/formulation, illumination, white balance, contact/handling and ambient conditions. Braster publications likewise use controlled, profile-specific foil and RGB acquisition rather than treating colour as a universal absolute-temperature signal. Useful public background:

- 2026 LCT review / Braster workflow: https://doi.org/10.1186/s43046-026-00383-6
- 2020 prospective Braster study: https://pmc.ncbi.nlm.nih.gov/articles/PMC7235966/

Human Braster timing/temperature parameters should not be copied blindly onto the mouse experiment. For the client cohort, reproduce and document the **actual client mouse acquisition protocol** so positives and controls occupy the same domain.

## Avoid session confounding

Do not capture every positive animal in one session/day and every control in another if this can be avoided. Session drift, lighting changes, TLC batch changes, camera settings and operator differences could become easier classification signals than biology.

Preferred practice:

- interleave tumor-bearing and control animals across capture sessions;
- keep the same TLC batch/settings within a first baseline cohort;
- record a `capture_session_id` and unique `capture_order`;
- keep the operator/procedure stable where possible;
- if multiple sessions are required, include both classes in the same sessions where feasible.

The repository validator explicitly blocks a cohort where positive and control classes occupy completely disjoint capture-session sets.

## Subject and view contract

The current `lct-target-v1` 417-feature builder is one canonical image per subject. For the first native mouse baseline:

- assign one stable de-identified `subject_id` per animal;
- set `split_group=subject_id`;
- choose one consistent canonical `view_id` across all trainable animals;
- keep exactly one canonical training image per animal in the v1 training manifest.

Extra raw views may be archived privately for future multi-view work, but they should not silently create duplicate subject rows in the current 417-feature table.

## Metadata to record

Start from `data/client_mouse_training_cohort_template.csv`. In addition to the common external-LCT intake fields, record:

- `strain_id`;
- `sex`;
- `weight_g`;
- `experimental_group`;
- `capture_session_id`;
- `tlc_batch_id`;
- `camera_settings_id`;
- `illumination_profile_id`;
- `view_id`;
- `capture_order`.

For the first baseline, the trainable pool is deliberately strict: one strain, one TLC profile/batch, one device profile, one acquisition profile, one camera-settings profile, one illumination profile and one canonical view. This is intended to reduce avoidable confounding while the cohort is small.

## Validation and bridge

Validate and assess readiness with:

```bash
python scripts/prepare_client_mouse_training_cohort.py \
  /path/to/private/client_mouse_training_manifest.csv
```

The script first runs the general external-data intake gate, then checks the mouse-specific domain/freeze/readiness rules.

When the result says `binary_training_ready=true`, write the subject-level manifest that can feed the current feature/training path:

```bash
python scripts/prepare_client_mouse_training_cohort.py \
  /path/to/private/client_mouse_training_manifest.csv \
  --output-native-manifest runtime/client-mouse-native/native_manifest.csv
```

Then, with the corresponding private images, run the existing 417-feature builder using the new source ID, followed by the native binary trainer. The exact training/evaluation split must remain subject-level.

## Evaluation plan

The original nine positives remain outside the new training pool. After a first native model is selected using only the new training/development cohort, the frozen nine can provide an additional positive target-domain evaluation signal. They still cannot measure specificity because they contain no negatives.

A proper held-out mixed positive/control evaluation set is still required before stronger performance claims.

## Privacy / repository rule

Do not commit raw client animal images or private experimental records by default. Commit only schemas, validators and non-sensitive aggregate artifacts unless the client explicitly authorizes otherwise.

`clinical_claim=NONE` remains mandatory. A research classifier is not a diagnosis or cancer probability.
