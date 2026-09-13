# MumGuard human research transfer v0.2.1 — artifact review

Date: 2026-09-13

## Review decision

**DO NOT activate this source-trained artifact as a live Contact-TLC disease decision head yet.**

The v0.2.1 source experiment is substantially stronger than v0.1 on DMR-IR and survives a harder subject-grouped side audit, but the current MumGuard Contact-TLC synthetic fixtures are outside the DMR-IR source-support gate. The correct target-domain behavior is therefore abstention / `INCONCLUSIVE`, not a transferred suspicious/not-suspicious label.

`clinical_claim = NONE` remains mandatory.

## Submitted artifact

Reviewed bundle: `MumGuard_DMR_IR_Transfer_v21_Artifacts.zip`.

The binary joblib itself is intentionally **not committed** by this review. The source extraction/training code remains reproducible and the artifact stays a research candidate.

### Source coverage

- DMR-IR subjects: **56**
- Cancer: **37**
- Healthy: **19**
- Usable source records: **1522**
- Failed source records: **0**
- LEFT coverage: 19 cancer subjects / 19 healthy subjects
- RIGHT coverage: 19 cancer subjects / 19 healthy subjects
- Subjects with both segmented sides: **1 cancer / 19 healthy**

This confirms that bilateral source availability is strongly label-confounded. Side presence/count must not be a disease feature and both sides must not be required for DMR-IR source inclusion.

## v0.2.1 feature contract

`dimensionless_local_spatial_v0_2_1`

The source-transfer feature vector contains only positive-affine-invariant local 2-D morphology:

- `std_z`
- `hotspot_peak_z`
- `local_contrast_p95`
- `gradient_p90`
- `hotspot_coherence`
- `hotspot_center_distance`
- `center_periphery_abs_diff`
- `left_right_abs_diff`
- `top_bottom_abs_diff`

Absolute Celsius is excluded. Side identity and number of sides are excluded. MumGuard bilateral asymmetry remains a separate target-domain evidence lane.

## Primary source-domain OOF result

The submitted artifact used 5-fold subject-level stratified out-of-fold evaluation. Logistic regression was selected over the calibrated linear SVM.

### Logistic — selected

- Balanced accuracy: **0.8393**
- Sensitivity: **0.7838**
- Specificity: **0.8947**
- AUROC: **0.8478**
- AUPRC: **0.9025**
- Brier: **0.1565**
- Confusion matrix: `[[17, 2], [8, 29]]`

These are **DMR-IR auxiliary source-domain research metrics on 56 subjects**. They are not MumGuard clinical accuracy, not Contact-TLC validation, and not a cancer probability claim.

### Calibrated linear SVM — not selected

- Balanced accuracy: **0.6038**
- Sensitivity: **0.8919**
- Specificity: **0.3158**
- AUROC: **0.7923**

## Side-availability robustness audit

Because healthy subjects usually contribute two segmented sides while cancer subjects usually contribute one, the review repeated the local-morphology experiment at side level with:

- subject-grouped folds, so one person never crosses train/test;
- no side identity or side count in the feature vector;
- total sample weight = 1.0 per subject regardless of one/two sides.

The harder grouped audit retained useful signal. A grouped logistic audit produced approximately:

- side-level balanced accuracy: **0.7763**
- side-level sensitivity: **0.8158**
- side-level specificity: **0.7368**
- side-level AUROC: **0.8421**

When the OOF side probabilities were averaged back to one value per subject, balanced accuracy remained approximately **0.8265** and AUROC approximately **0.8606**.

This does not prove clinical validity, but it makes it less likely that the source result is explained only by controls having twice as many source sides.

## Target-domain dry run: MumGuard synthetic Contact-TLC

The reviewed v0.2.1 feature contract was applied to the current synthetic MumGuard full-field fixtures using the current client-response segmentation and relative-hue signal path. The submitted source artifact OOD threshold is **2.8081** in standardized source-feature space.

### A — symmetric control

- LEFT nearest source distance: **4.8942** → `ABSTAIN_OOD`
- RIGHT nearest source distance: **4.8942** → `ABSTAIN_OOD`

### B — left local anomaly

- LEFT nearest source distance: **4.8160** → `ABSTAIN_OOD`
- RIGHT nearest source distance: **4.8942** → `ABSTAIN_OOD`

### C — low chromatic response

- LEFT visible-response support: **0** → insufficient measurement
- RIGHT visible-response support: **0** → insufficient measurement

The model's raw numeric probabilities on A/B are deliberately not treated as decisions because the OOD gate rejects those fields.

## Meaning

The useful result is not “the human model is ready.” The useful result is:

1. Local human thermal morphology contains a materially stronger signal than v0.1 on the DMR-IR source cohort.
2. The source-side confound was identified and isolated rather than silently learned.
3. The OOD gate correctly refuses to pretend that DMR-IR infrared and current MumGuard Contact-TLC are already the same domain.
4. The next missing evidence is **real human MumGuard Contact-TLC target-domain data**.

## Next target-domain closure

Before any live `SUSPICIOUS_RESEARCH` / `NOT_SUSPICIOUS_RESEARCH` activation, obtain real human MumGuard sessions from the same device/TLC profile and run the same local feature contract to establish target support. Unlabelled sessions are useful for OOD/domain-support mapping; outcome-linked sessions are required for real target calibration/validation.

Desired future target records:

- raw LEFT/RIGHT Contact-TLC captures;
- TLC profile/device profile and acquisition conditions;
- exam grouping and side/region identity;
- when available, outcome reference such as healthy / benign / cancer plus affected side/region.

Until then, MumGuard may expose the source-transfer lane only as provenance/domain-support research evidence. If a target session is out-of-domain or measurement support is weak, the research decision must remain `INCONCLUSIVE`.
