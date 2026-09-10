# Lane F data hunt status — 2026-09-10

## Goal

Assemble legally usable thermography data for a research-only tumor-like / no-tumor-like classifier while keeping contact-LCT target-domain evidence separate from infrared auxiliary evidence.

## Strongest immediately usable auxiliary source

### DMR-IR

Recent peer-reviewed work consistently describes a dynamic frontal subset with 56 independent subjects: 37 subjects with breast cancer and 19 healthy subjects. Image counts are reported around 1120–1125 depending preprocessing/export. The important unit for splitting is the subject, not the image. Several sources state that the cancer cases were histopathologically proven and that healthy cases did not contain benign findings.

Use: auxiliary infrared cancer-vs-healthy representation / classifier research only.

Do not use: direct proof that a contact-LCT classifier works.

Acquisition candidates:
- original DMR/PROENG project: http://visual.ic.uff.br/dmi/
- Kaggle mirror: `asdeepak/thermal-images-for-breast-cancer-diagnosis-dmrir`

Licensing/provenance gate: the Kaggle mirror says `Data files © Original Authors`; redistribution and commercial derivative use must be checked against the original source terms rather than inferred from the mirror.

## Second auxiliary source

### Breast Thermography v3 — Mendeley Data

DOI: `10.17632/mhrt4svjxc.3`

119 subjects, 357 radiometric JPEG thermograms, three views per subject. The accompanying Data in Brief article reports 84 benign and 35 malignant patients, with diagnosis obtained from pathology reports. Dataset page license: CC BY-NC 3.0.

Use: research-only benign-vs-malignant auxiliary experiment with subject-level grouping.

Commercial gate: because the dataset is noncommercial, do not include a model trained on it in a commercial deliverable without separate permission/legal review.

## Relevant but unavailable dataset

### DBT-TU-JU

The project page reports 100 subjects and 1100 thermograms: 45 normal, 33 benign, 13 malignant and 6 unknown, plus expert hotspot ground truth and mammography/FNAC/clinical validation. The same page still says dataset release permission is pending from the funding agency.

Use now: methodology benchmark and possible outreach target only.

## Closest human contact-LCT evidence

### Braster / ThermaALG

Clinical trial NCT03858738 enrolled 274 women and used contact liquid-crystal thermography with standard diagnostic imaging and biopsy for indicated cases. No public subject-level raw thermogram release was identified in the targeted search.

Use now: acquisition/interpretation design precedent and possible data-access outreach target.

## Mouse contact-LCT precedent

Stevens & Rogers (1971), DOI `10.1177/153857447100500404`, directly studied cholesteric liquid-crystal thermography over transplantable mouse tumors. It supports the biological experiment design but is not an open raw training dataset.

## Current project data boundary

- 26 publication-derived LCT plates: reference-only.
- 9 client-device mouse images: all tumor-bearing, frozen target-domain positive cohort for domain and tumor-burden evaluation.
- No fabricated subject-level negatives.
- No model prediction may be relabeled as ground truth.
- Infrared and contact-LCT metrics must be reported separately.
- `clinical_claim = NONE` remains mandatory.

## Next data actions

1. Acquire DMR-IR into private experiment storage and build a subject-level manifest before any model split.
2. Acquire Mendeley v3 for research-only comparison and preserve all three views under one subject ID.
3. Continue target-domain search for open raw contact-LCT with independent tumor/control labels.
4. Keep the nine client images completely out of any train/test cycle used to claim binary performance.
