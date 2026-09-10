# DMR-IR auxiliary baseline

## Purpose

DMR-IR is an auxiliary infrared-thermography dataset used to validate the supervised training/evaluation harness before any target-domain contact-LCT classifier is claimed. It is a different modality from contact liquid-crystal thermography and must remain reported separately.

## Verified acquisition routes

1. Original UFF Visual Lab dynamic thermography database page: `https://visual.ic.uff.br/en/proeng/thiagoelias/`
2. Original database archive linked by that page: `https://visual.ic.uff.br/proeng/thiagoelias/database.rar`
3. Figshare mirrors of DMR-IR material published under CC BY 4.0, including DOI `10.6084/m9.figshare.21225389`.
4. Hugging Face derivative used by the BreastCATT research project: `SemilleroCV/DMR-IR`; pin the exact revision before experiments and preserve patient IDs.

The UFF page states that the dynamic protocol records 20 sequential images after cooling. Published DMR-IR studies consistently describe the commonly used binary cohort as 56 subjects: 37 cancer and 19 healthy. Cancer cases are reported as histopathologically proven in multiple papers. Do not split sequential images from the same patient across folds.

## First baseline cohort

Preferred first experiment: one frontal thermogram per independent subject where possible, producing a subject-level 56-case baseline rather than thousands of correlated frames. A second experiment may use all dynamic frames only with grouped patient-level cross-validation.

Binary research mapping for this auxiliary experiment:

- original healthy -> `NO_TUMOR_LIKE`
- original cancer/sick -> `TUMOR_LIKE`

Keep the original source label and provenance in the manifest. This mapping is for auxiliary infrared research only and is not a contact-LCT diagnostic claim.

## Model order

1. Frozen official DINOv2 ViT-S/14 embeddings.
2. StandardScaler + class-balanced Logistic Regression.
3. StandardScaler + calibrated/score-producing linear or RBF SVM as a comparison.
4. Five-fold subject-level stratified evaluation where feasible, plus a leave-one-subject-out sensitivity check if runtime permits.

Report sensitivity, specificity, AUROC, AUPRC, balanced accuracy, confusion matrix, and Brier score for probability-producing models. Persist out-of-fold predictions keyed by subject ID.

## Transfer boundary

A strong DMR-IR score only demonstrates that the harness can learn an infrared thermal abnormality signal. It must not be presented as evidence that the client contact-LCT device works. Transfer to contact-LCT is a separate experiment with OOD/domain-shift reporting, and the frozen 9-image client mouse cohort is never used as its own training proof.
