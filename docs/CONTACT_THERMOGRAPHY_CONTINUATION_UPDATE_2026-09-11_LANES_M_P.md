# Contact Thermography AI — Continuation Update after Lanes M–P

**Date:** 2026-09-11  
**Repository:** `omarkhair70-droid/contact-thermography-ai`  
**Canonical branch:** `integration`  
**Integration SHA before this docs-only update:** `825ed9e67e4746d5e9a937133d11eab3b820a733`

This file is the current delta on top of `docs/CONTACT_THERMOGRAPHY_MASTER_CONTINUATION_HANDOFF_2026-09-11.md`. Read the original master for the full project history, then use this update for the current next action. Do not restart old lanes.

## What changed since the master handoff

### Lane M / PR #26 — merged

`Lane M — open-license LCT references and DeepBraster precedent`

- added a case-level contact-LCT reference registry;
- preserved three pathology-linked CC-BY open-license reference cases;
- preserved five public Braster case records as copyright-restricted reference metadata only;
- forbids reference cases from silently becoming train eligible;
- documents DeepBraster / THERMACRAC / THERMARAK / INNOMED as architectural and validation precedent, not as our data or our performance.

### Lane N / PR #27 — merged

`Lane N — materialize open LCT references and negative-control hunt`

- converted the three open-license cases into a reproducible source-figure manifest/fetch path;
- fetches only explicit CC-BY reference assets and records provenance/hash metadata;
- excludes restricted Braster images from the open pack;
- records the latest negative/control hunt state;
- confirmed that no reusable public same-device mouse control cohort has been found.

The historical Stevens/Rogers mouse LCT work is useful methodology/biology precedent but contains tumor-bearing mice and normal-tissue comparisons, not a reusable independent negative/control mouse image cohort.

### Lane O / PR #29 — merged

`Lane O — external LCT intake gate and USM access packet`

- added `scripts/validate_external_lct_intake.py`;
- added `data/external_lct_intake_template.csv`;
- requires de-identified subject/image mapping, ground-truth provenance, TLC/device/acquisition provenance, rights/data-use declarations and subject-level grouping;
- rejects direct identifier columns and silent provenance gaps;
- keeps incoming rows non-trainable by default;
- requires explicit model-research permission and known target-domain profiles before a row may be a `TRAIN_CANDIDATE`;
- does not weaken the existing native readiness gate;
- added `docs/USM_DATA_ACCESS_REQUEST_PACKET.md` as a prepared draft only. No outreach email has been sent automatically.

### Lane P / PR #31 — merged

`Lane P — same-device mouse native cohort acquisition gate`

This closes an important ambiguity in the earlier plan: **controls alone are not enough**.

The nine original client mouse images remain:

- tumor-bearing positives;
- `FROZEN_TARGET_EVAL`;
- forbidden from reuse as training positives.

Therefore the minimum current engineering gate for a first native mouse baseline requires at least:

- **5 NEW tumor-bearing training subjects**, and
- **5 genuine control/negative subjects**,

with larger balanced cohorts strongly preferred. This threshold only opens a first research baseline; it is not a clinical-validation sample-size claim.

Lane P adds:

- `data/client_mouse_training_cohort_template.csv`;
- `scripts/prepare_client_mouse_training_cohort.py`;
- `tests/test_client_mouse_training_cohort.py`;
- `docs/CLIENT_MOUSE_NATIVE_COHORT_ACQUISITION.md`.

The gate protects against:

- reusing `CLIENT-MOUSE-0030` through `CLIENT-MOUSE-0038` for training;
- controls-only cohorts;
- mixed species or unsupported labels;
- duplicate animals / non-animal-level grouping;
- mixed TLC/device/acquisition domains;
- mixed strain or sex in the strict first baseline;
- mixed TLC batch, camera settings, illumination or canonical view;
- positive/control classes occupying completely disjoint capture-session sets;
- use of a TLC profile incompatible with the current 417-feature contract.

The current `lct-target-v1` 417-feature builder remains explicitly profile-locked to:

`client-device-tlc-pending`

Do not silently rename that profile to a fabricated calibrated version. When the real TLC formulation/calibration becomes known, add/version a real profile explicitly.

## Current end-to-end native path

The code/data-governance path is now:

`new target-domain image -> external intake gate -> mouse cohort gate -> subject-level native manifest -> lct-target-v1 417 features -> native Logistic/SVM baseline -> OOD/domain checks -> frozen/held-out evaluation`

The software path is ready to accept a valid new cohort.

## Current real blocker

The main blocker is now **new target-domain data**, not missing model code.

For the client mouse route, acquire a new cohort from groups already available under the client's approved study/ethics protocol, using the same target acquisition domain wherever possible. The repository does not prescribe new animal interventions.

For the human route, USM remains the strongest active direct-access lead. The prepared request packet asks only for de-identified original contact-LCT data plus case-level reference-standard, TLC/device/acquisition and rights metadata. Sending any external email still requires explicit user approval.

## Exact next actions

1. Obtain new client mouse training data: at least 5 new tumor-bearing + 5 genuine controls as the minimum technical gate, preferably more and balanced/interleaved across acquisition sessions.
2. Or obtain a coherent human contact-LCT positive/negative cohort through USM/Braster direct data access.
3. Run the external intake validator; do not hand-edit rows into trainability.
4. For the mouse route, run `scripts/prepare_client_mouse_training_cohort.py`.
5. Only if the gate is green, build the 417-feature table and run the existing native baseline.
6. Keep the original nine client positives frozen and untouched until model selection is complete; they may provide additional positive target-domain evaluation evidence, but cannot estimate specificity.
7. Preserve `clinical_claim=NONE`; no cancer probability or diagnosis.

## Do not redo

Do not re-audit model choice, rebuild DINO integration, rerun DMR-IR as if it were target LCT, merge stale PR #7/#8, synthesize negatives, train on the open reference figures, or reuse the frozen nine as training positives.

## New-chat continuation instruction

> Read `docs/CONTACT_THERMOGRAPHY_MASTER_CONTINUATION_HANDOFF_2026-09-11.md` and then `docs/CONTACT_THERMOGRAPHY_CONTINUATION_UPDATE_2026-09-11_LANES_M_P.md` in `omarkhair70-droid/contact-thermography-ai`. Inspect current `integration`. Do not redo merged lanes. Continue from data acquisition/readiness: the current mouse training gate requires new same-domain trainable positives and controls, while the original nine client positives remain frozen evaluation only. Preserve `clinical_claim=NONE`.
