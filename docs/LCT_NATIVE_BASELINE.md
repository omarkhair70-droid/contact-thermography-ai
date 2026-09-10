# Contact-LCT native binary baseline

## Purpose

This lane creates the first binary training path that is allowed to fit only on a coherent contact-LCT target-domain pool. It does **not** claim that the current target dataset is sufficient to train a clinically meaningful classifier.

The desired product output remains a research signal such as `TUMOR_LIKE` / `NO_TUMOR_LIKE`, with `clinical_claim=NONE`, plus an OOD/abstention state. It must not be described as cancer probability or diagnosis.

## Current state

The merged target Dataset v1 contains two observed human Braster/contact-LCT case exemplars from the 2026 Malaysian review and nine frozen client mouse images. The two human exemplars are reference-only. The nine client images are all tumor-bearing according to the client and therefore cannot supply the negative class.

The native trainer therefore **must fail closed today**. That is a feature, not a blocker to bypass.

## Training gate

The trainer requires, at minimum:

- modality = `contact-LCT` for every trainable row;
- one species in a training run;
- one TLC formulation/profile in a training run;
- one device profile in a training run;
- `use_role=TRAIN_CANDIDATE` and `train_eligible=true`;
- at least five positive and five negative subjects for a technical baseline;
- subject-level features with no missing/non-finite values.

Five per class is only the minimum to make a provisional cross-validation run technically possible. It is not enough to establish clinical validity.

## Candidate features

The feature CSV is intentionally generic (`feature_*`) so a later target-native feature builder can supply:

- frozen DINOv2 embedding features;
- TLC colour/response morphology;
- region area, connected components, compactness and spatial distribution;
- local/relative contrast rather than a hard-coded assumption that tumors are always hotter;
- QC/glare/focus/coverage features where appropriate;
- bilateral features for human left/right exams when same-profile pairs are available.

## Why temperature direction cannot be hard-coded

The historical 1971 mouse contact-liquid-crystal study explicitly notes that altered tumor perfusion can produce a **higher or lower** skin temperature over a superficial tumor. The client model must therefore learn relative response and morphology rather than a rule equivalent to `hot = tumor`.

Source: Stevens JD, Rogers W. *Liquid Crystal Thermography of Transplantable Mouse Tumors*. Vascular Surgery. 1971;5(4):186-192. DOI: 10.1177/153857447100500404. Raw article images remain copyright-restricted.

## Human observed exemplars

The 2026 open-access review contains two observed case-level LCT examples from an ongoing Malaysian cohort:

- Fig. 3: true-positive LCT case, histopathology-confirmed invasive breast carcinoma;
- Fig. 4: true-negative LCT case, biopsy-confirmed fibrocystic change.

The article is CC BY 4.0 unless figure-specific credit says otherwise, but the cases are explicitly preliminary/exploratory and do not form a training cohort. They remain reference-only in Dataset v1.

Source: Mohd Sha’ari AB et al. *Liquid crystal thermography for breast cancer detection: principles, current evidence, limitations, and future directions*. Journal of the Egyptian National Cancer Institute. 2026;38:47. DOI: 10.1186/s43046-026-00383-6.

## Historical human cohort lead

Davison et al. reported 105 women with abnormal breast characteristics, including 17 histologically proven carcinomas used to derive six thermographic signs, and separately described LCT patterns in 197 apparently healthy women. The article is available to read through several indexing/library routes, but figure-level reuse/training rights must be established before extracting images into a dataset.

Source: Davison TW et al. *Detection of breast cancer by liquid crystal thermography. A preliminary report.* Cancer. 1972;29(5):1123-1132. PMID 4553757.

## Execution path

1. Keep the current client mouse cohort frozen for evaluation and tumor-burden analysis.
2. Continue target-data acquisition with priority on **mouse contact-LCT controls/negatives from the same acquisition setup**, followed by same-device/client controls when available.
3. Build target-native `feature_*` rows using the existing DINO/TLC signal engines.
4. Run `scripts/train_lct_native_binary.py --readiness-only` before every training attempt.
5. Train Logistic Regression and calibrated Linear SVM only after the gate opens.
6. Evaluate at subject/animal level and retain OOD abstention.
7. Integrate the research classification into the web app only after an eligible target-domain run exists.
