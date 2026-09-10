# Lane F — Tumor / No-Tumor Research Classifier + Data Acquisition

## Product target

The product should progress beyond reference-unusualness and expose a **research binary classification** for each target-domain examination:

- `TUMOR_LIKE`
- `NO_TUMOR_LIKE`

This is a research output, not a clinical diagnosis. `clinical_claim` remains `NONE` until appropriate target-domain validation exists.

DINOv2 is already the visual AI backbone. Lane F adds the missing supervised classifier head and the data program needed to train/evaluate it.

## Why the current system is not yet the final classifier

The 26 publication-derived LCT plates are reference-only and not a clean patient-level supervised tumor/control dataset. The first 9 client-device mouse images are all tumor-bearing, so they provide target-domain positives, domain adaptation and tumor-burden experiments, but no true negative animals. Lane E also measured a strong client-vs-publication embedding domain shift; therefore the old reference set cannot be treated as interchangeable ground truth.

## Data acquisition program

### Tier A — target-domain contact/LCT data

Highest priority. Search open papers, repositories, theses and supplementary material for legally usable contact/liquid-crystal thermograms with case-level tumor/control or pathology labels. Preserve source URL/DOI, license, modality, TLC/device, subject/case ID, label provenance, and split group. Publication figures are weak-label/reference data unless raw images and labels are explicitly available.

The client 9-image cohort stays isolated as a frozen target-domain positive/blind tumor-burden cohort. It must not be trained on and then reused as proof of performance.

### Tier B — auxiliary public thermal data

Infrared thermography is a different modality, but can supply thermal-abnormality representation learning and controlled transfer experiments.

Verified public candidates:

1. **DMR-IR / Database for Mastology Research with Infrared Image** — public human breast infrared thermography. A Figshare copy is CC BY 4.0. Recent literature describes DMR-IR as the most widely used public breast thermography dataset and reports hundreds of subjects. Use only as auxiliary IR data, never as direct LCT validation.
2. **Breast Thermography, Mendeley Data DOI 10.17632/mhrt4svjxc.3** — 119 women, three IR views per patient, pathology-backed labels; 84 benign and 35 malignant patients. Dataset license is Attribution-NonCommercial 3.0. Use as auxiliary IR supervised data with patient-level grouping.

### Tier C — non-thermal breast data

Mammography, ultrasound and histology datasets may be used only for generic representation experiments or metadata research, not as thermal-image training labels. They do not solve the LCT target-domain gap.

## What must never be used as ground truth

- model predictions relabeled as truth;
- synthetic images as real tumor/control evidence;
- background patches from tumor-bearing animals as subject-level `NO_TUMOR` cases;
- IR labels silently treated as contact-LCT labels;
- multiple views of the same subject split across train/test.

## Model ladder

### Baseline A — target-domain fused shallow classifier

Frozen DINOv2 ViT-S/14 embedding + explicit LCT/QC/morphology features. Evaluate Logistic Regression and SVM first. This is the preferred low-data baseline because it has fewer trainable parameters and is auditable.

### Baseline B — nonlinear fused classifier

A small regularized MLP and/or tree ensemble over the same fused representation. Keep target-domain subject grouping and probability calibration.

### Transfer C — auxiliary IR thermal abnormality pretraining

Train an IR abnormality representation/head using public IR datasets, then test transfer into LCT using frozen generic DINO features plus domain adaptation such as CORAL/MMD. Report IR and LCT-domain results separately. A good IR score is not evidence that the LCT classifier works.

### Fine-tuning D — later only

Fine-tune DINO only after enough independent labeled target-domain LCT subjects exist. Do not fine-tune on the 9-client cohort and report on the same 9.

## Evaluation contract

Every candidate must use subject/animal-level grouped splitting and report sensitivity, specificity, AUROC, AUPRC, confusion matrix, calibration/Brier score, and class counts. OOD/domain-shift status remains next to the binary research class. A forced binary class may be produced for experiment comparison, but the product must also expose evidence/OOD status so a result is not presented as validated diagnosis.

## Immediate build order

1. Build a source registry and legal-use metadata.
2. Add reproducible download/ingestion for permitted public thermal datasets.
3. Extract frozen DINO embeddings and thermal features with subject IDs preserved.
4. Train/evaluate Logistic/SVM baselines on auxiliary IR first to validate the training harness.
5. Assemble every available labeled contact-LCT case and run a target-domain-only experiment separately.
6. Add domain adaptation experiments only after the baselines are reproducible.
7. Integrate the best research head into the web API as `research_binary_class`, plus OOD/evidence status.
8. Keep Lane E tumor-size association as a separate auxiliary target, not a substitute for tumor/control labels.

## Current evidence constraint

Modern raw labeled contact-LCT datasets do not appear to be openly available at useful scale. Braster has published LCT studies (including a 274-patient prospective pilot and a much larger observational program) and describes AI analysis, but the raw image dataset is not publicly exposed. This is exactly why Lane F must maintain separate target-domain and auxiliary-domain evidence rather than pretending one dataset solves both.
