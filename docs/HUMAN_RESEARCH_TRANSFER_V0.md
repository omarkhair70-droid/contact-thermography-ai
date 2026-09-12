# MumGuard human research transfer v0

## Product purpose

MumGuard already produces usable human Contact-TLC session evidence: reconstructed LEFT/RIGHT fields, focal/core hyperthermia evidence, bilateral asymmetry, abnormal thermal behaviour, optional local DINO evidence, explainable maps and measurement-quality status.

The missing product layer is a **research decision head** that can eventually convert those measurements into one of:

- `SUSPICIOUS_RESEARCH`
- `NOT_SUSPICIOUS_RESEARCH`
- `INCONCLUSIVE`

with a `research_concern_score` from 0 to 100 when inference is allowed.

This score is deliberately separate from `clinical_risk`. It is **not a cancer probability**, and `clinical_claim` remains `NONE`.

## Why v0 excludes absolute temperature

The existing DMR-IR auxiliary human experiment used 56 independent subjects (37 cancer / 19 healthy). Absolute/summary temperature features separated the source labels strongly, but the offset-removal stress test dropped markedly. That makes absolute temperature a serious acquisition/protocol confounding risk.

MumGuard live Contact-TLC currently exposes relative thermal signal and does not have a validated device-specific Celsius calibration. Therefore v0 does **not** transfer DMR-IR absolute temperature thresholds into MumGuard.

The first shared feature contract is offset-invariant thermal shape/spread:

1. standard deviation
2. p90 - p10
3. p50 - p10
4. p90 - p50
5. max - min
6. mean - median
7. p10 - min
8. max - p90

Adding a constant thermal offset does not change these features.

## Training and selection

`scripts/train_human_research_transfer_v0.py` expects a source-domain CSV with:

- `subject_id`
- `label`
- the eight shared features above

Rows are aggregated at the subject level before evaluation. Subjects are never split across folds.

The harness compares:

- class-balanced Logistic Regression
- calibrated class-balanced linear SVM

using stratified subject-level out-of-fold predictions. Balanced accuracy is the primary selection metric, Brier score breaks ties, and Logistic wins exact ties for simplicity.

The emitted bundle contains:

- fitted selected source model
- exact feature contract
- source-standardized subject feature vectors
- nearest-source OOD threshold
- model/source provenance
- `clinical_claim=NONE`

The default artifact status is `RESEARCH_TRANSFER_CANDIDATE`. It is not used by runtime until explicitly reviewed and marked `ACTIVE_RESEARCH_TRANSFER`.

## OOD / abstention gate

A MumGuard target feature vector is standardized with the source model scaler and compared with source subject vectors using nearest-source standardized Euclidean distance.

If target support is outside the frozen source threshold, runtime returns:

`INCONCLUSIVE / ABSTAIN_OOD`

No concern score is emitted in that case.

Measurement failure also returns `INCONCLUSIVE` before model inference.

## Current activation state

**Not active yet.**

The repository contains the DMR-IR acquisition registry and prior auxiliary experiment summaries, but it does not currently contain the exact source feature table or fitted transfer artifact required to reproduce and activate this v0 head. No weights or thresholds are fabricated from aggregate paper/results metrics.

Activation requires:

1. recover or regenerate the subject-level DMR-IR shared-feature CSV from the verified source acquisition;
2. run the training harness;
3. inspect OOF metrics and OOD support;
4. freeze the artifact/version;
5. then integrate the reviewed artifact into `mumguard_session_fusion` as a separate `research_decision` object.

## Separation from clinical calibration

The existing `human_decision.py` contract remains unchanged. A future outcome-linked MumGuard clinical calibration is a separate layer.

Research transfer may answer whether the current thermal pattern is source-model suspicious, not whether a patient has cancer. Clinical probability remains unavailable until verified MumGuard human outcomes and device-specific calibration exist.
