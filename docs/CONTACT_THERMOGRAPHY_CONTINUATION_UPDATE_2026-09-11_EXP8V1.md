# Contact Thermography AI — continuation update: resolved 8/1 client map + experimental native model

Date: 2026-09-11

This file supersedes the unresolved-class-map state in `CONTACT_THERMOGRAPHY_CONTINUATION_UPDATE_2026-09-11_MIXED_CLIENT_COHORT.md`.

## Resolved client mouse class map

The nine supplied target-device mouse contact-LCT images are now mapped as:

- `WA0030` → `TUMOR_BEARING`
- `WA0031` → `TUMOR_BEARING`
- `WA0032` → `TUMOR_BEARING`
- `WA0033` → `TUMOR_BEARING`
- `WA0034` → `TUMOR_BEARING`
- `WA0035` → `TUMOR_BEARING`
- `WA0036` → `TUMOR_BEARING`
- `WA0037` → `TUMOR_BEARING`
- `WA0038` → `HEALTHY`

Therefore the available target cohort is **8 tumor-bearing / 1 no-tumor**.

The client also clarified that contact-LCT visibility is not a tumor-size measurement. A superficial smaller tumor may appear more visibly than a deeper tumor. The binary product target is presence/absence only: `TUMOR_LIKE` vs `NO_TUMOR_LIKE`. Do not infer or report tumor size from visible response area/intensity.

## Validation consequence

One negative subject is enough to fit an exploratory classifier but is not enough for defensible subject-level binary cross-validation or specificity estimation.

The canonical trainer remains strict. Do not lower its normal gate or present resubstitution as validation.

Data priority remains additional genuine same-domain controls. Under the current five-negative engineering target, four more independent no-tumor mice are needed. Augmentations or multiple views of WA0038 do not count as independent negative subjects.

## Experimental demo lane

A separate explicit demo lane was created so the product can exercise end-to-end binary inference before those controls arrive:

- branch: `feat/experimental-native-8v1-20260911`
- model version: `0.1.0-exp8v1-dino`
- model type: portable JSON Logistic Regression
- input block: official frozen `dinov2_vits14` 384-D embedding inside `lct-target-v1`
- target TLC profile: `client-device-tlc-pending`
- subjects: 9
- positives: 8
- negatives: 1 (`WA0038`)
- registry status: `ACTIVE_RESEARCH`
- activation scope: `INTERNAL_RESEARCH_DEMO_ONLY`
- validation status: `NOT_VALIDATED_SINGLE_NEGATIVE`
- `clinical_claim=NONE`

Artifact:

`artifacts/lct_native_exp8v1_dino.json`

Reproducible trainer:

`scripts/train_lct_single_negative_demo.py`

The trainer requires an explicit `--acknowledge-single-negative` flag and does not weaken `scripts/train_lct_native_binary.py`.

## Runtime behavior

For `client-device-tlc-pending` images, the live DINO analysis path can attach:

`native_binary_research`

with:

- `research_binary_class`
- `model_score`
- `decision_threshold`
- `validation_status`
- training class counts
- non-clinical semantics

The score is not a cancer probability, diagnosis, tumor-size estimate or validated clinical risk score.

Publication/reference TLC profiles do not receive this target-native binary classification.

## Deployment

Coolify production preparation has been completed manually:

- application repo: `omarkhair70-droid/contact-thermography-ai`
- deployment branch: `integration`
- build pack: Dockerfile
- exposed port: `8000`
- PostgreSQL resource: `contact-thermography-db`
- `DATABASE_URL`: Coolify internal PostgreSQL URL
- `APP_ENV=production`
- `STORAGE_BACKEND=filesystem`
- `STORAGE_ROOT=/var/lib/lct/storage`
- persistent volume destination: `/var/lib/lct/storage`
- healthcheck: HTTP GET `localhost:8000/health`, expected 200

After the experimental PR is green and merged to `integration`, redeploy the Coolify app and smoke-test `/health`, then submit a client-profile image and inspect `native_binary_research` in the analysis response/report.

## Next-chat instruction

Read the master handoff chain ending with this file. Inspect current `integration` and the state of the experimental 8-vs-1 PR. Do not revert the resolved 8/1 map, do not treat the model score as a cancer probability, and do not claim validation from the nine training subjects. Finish CI/merge/deployment, then obtain additional genuine same-domain no-tumor controls for a proper subject-level native baseline.
