# Contact Thermography AI — continuation update after PR #33

Date: 2026-09-11

Repository: `omarkhair70-droid/contact-thermography-ai`

Canonical branch: `integration`

Integration SHA after PR #33: `24f8d03fd81fa7118366acf7a3b0906362b0c6ec`

This update supersedes the old Lane-P statement that five new tumor-bearing mice are required before native mouse binary training can begin.

## 1. Revised client-mouse policy

The client clarified that the nine existing target-device mouse subjects (`CLIENT-MOUSE-0030` through `CLIENT-MOUSE-0038`) were all experimentally tumor-injected, captured in the same experiment under broadly similar conditions, with tumor size as the main reported biological difference. Additional tumor-bearing mice may not be available.

PR #33 therefore changed the first native mouse baseline policy:

- the canonical nine remain non-trainable source evidence in `data/lct_target_dataset_v1_manifest.csv`;
- the cohort-preparation gate may promote copies of those nine into a generated **internal-development** manifest;
- new tumor-bearing mice are no longer required for the first internal binary baseline;
- the minimum engineering blocker is now at least **5 genuine same-domain control/negative mice**;
- more controls, preferably roughly nine or more, are strongly preferred for class balance;
- the first result is `INTERNAL_RESEARCH_CROSS_VALIDATION`, not independent external validation;
- `clinical_claim=NONE` remains mandatory.

The nine canonical rows now use animal-level `split_group=subject_id`, preventing cross-validation leakage if they are promoted into an internal-development manifest.

## 2. Supplemental-control gate

`data/client_mouse_training_cohort_template.csv` now includes:

`matches_client_positive_domain`

Every trainable supplemental mouse row must explicitly confirm this as true. Controls should match the client positives as closely as possible in contact-LCT modality, TLC profile/batch, device/camera, white balance/exposure, illumination, acquisition geometry/view, contact procedure, ambient/acclimatisation protocol and image export path.

The current 417-feature contract remains intentionally profile-locked to:

`client-device-tlc-pending`

Do not invent a calibrated `v1` profile until the client's actual TLC formulation/calibration is documented.

## 3. Historical metadata limitation

The original nine positives do not currently have complete strain/sex/weight/capture-session metadata in the canonical manifest. Their tumor-bearing status and shared experimental context are known, but not every acquisition covariate was recorded when they were first ingested.

Therefore the first internal binary report must explicitly disclose that same-session/covariate matching against the legacy positive cohort cannot be fully audited. This is another reason the result must not be called external validation.

## 4. Current no-tumor / control evidence hunt

A focused historical contact-LCT pass found stronger human control evidence than the earlier handoff recorded.

### Davison et al. 1972

`Detection of breast cancer by liquid crystal thermography. A preliminary report`

DOI: `10.1002/1097-0142(197205)29:5<1123::AID-CNCR2820290502>3.0.CO;2-8`

The publication reports:

- 197 apparently healthy women with no breast abnormalities whose liquid-crystal thermograms were classified into normal pattern types;
- a separate abnormal-breast cohort;
- 17 histologically proven carcinoma cases used to derive malignancy signs;
- known-lesion groups including fibrocystic disease, fibroadenoma and carcinoma.

This is strong evidence that a substantial human contact-LCT normal/control cohort historically existed. It is **not yet a verified downloadable subject-level raw dataset**. Article/figure access does not automatically imply model-training rights for every image.

### Ewing, Davison & Fergason 1973

`Effects of Activity, Alcohol, Smoking, and the Menstrual Cycle on Liquid Crystal Breast Thermography`

Ohio Journal of Science 73(1):55-58.

The study reports 10 apparently healthy women examined with liquid-crystal breast thermography every day for 28 to 45 consecutive days. A public/institutional PDF is available. This is useful evidence about repeated healthy contact-LCT patterns and acquisition stability, but it is not a verified raw machine-learning dataset and image reuse/training rights still require verification.

### Bichara 1975/1976 thesis

`Liquid Crystal Thermography: A New System for Breast Cancer Detection`

The Oregon State doctoral work describes a comparative study of 75 volunteer women using:

- liquid-crystal thermography with an elastic thin film;
- liquid-crystal thermography by direct spray over a blackened surface;
- infrared thermography.

The thesis figure list explicitly includes **Figure 28: a comparative thermogram of a normal pattern**, with both liquid-crystal views and an infrared view. This is currently the clearest discovered published lead to an actual normal/negative contact-LCT image, not merely a cohort count.

Treat it as a candidate reference only until an official repository copy, image-level provenance and reuse/model-training rights are verified. A third-party preview or accessible thesis scan is not sufficient by itself to authorize commercial model training.

### Malaysia observed cases 2026

The open-access 2026 LCT review provides one pathology-linked malignant contact-LCT exemplar and one biopsy-confirmed benign/fibrocystic true-negative exemplar under CC BY 4.0, subject to figure-specific attribution. These are useful reference exemplars, not a training cohort.

### Mouse target-domain state

No public, reusable same-device/same-TLC healthy mouse control cohort matching the client's setup has been found. Stevens & Rogers (1971) remains a direct mouse-LCT precedent but studies tumor-bearing animals and surrounding normal tissue rather than a clean independent healthy-control cohort, and the source is not an open raw training dataset.

## 5. What to request from the client

The most valuable client-side additions are now only:

1. any untreated/control/non-tumor mice already captured (or capturable under the existing approved protocol) with the same contact-LCT setup;
2. tumor-size/volume mapping for `WA0030` through `WA0038`;
3. TLC foil/formulation name, batch or active temperature range if known;
4. any missing acquisition metadata that can be recovered for the original nine.

Do not ask for new tumor-bearing animals solely to satisfy the old software gate.

## 6. Once valid controls arrive

Run, in order:

1. external intake validation;
2. `scripts/prepare_client_mouse_training_cohort.py` to generate the combined internal-development manifest;
3. 417-feature extraction (`lct-target-v1`);
4. native Logistic Regression and calibrated Linear SVM;
5. subject-level internal cross-validation;
6. feature-block ablations and OOD checks;
7. choose/freeze the research baseline;
8. integrate research classification into the app without changing `clinical_claim=NONE`.

Do not fine-tune a large DINOv2 backbone on this tiny cohort as the first move.

## 7. If no controls can be obtained

Do not fabricate negatives and do not relabel human/IR images as mouse controls.

Close the binary mouse head as data-blocked and finish the deliverable as a contact-LCT research analysis platform using QC, TLC response, morphology, DINOv2 representation, OOD/domain analysis and the separate tumor-burden experiment if tumor-size mapping is supplied.

## 8. Immediate continuation rule

Do not redo Lanes M/N/O/P or revert to the pre-PR-33 `5 new positives + 5 controls` requirement.

Current blocker for the first native binary mouse baseline:

**genuine same-domain mouse controls.**
