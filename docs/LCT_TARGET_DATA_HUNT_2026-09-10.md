# LCT target-domain data hunt — 2026-09-10

## Purpose

The end goal remains a research classifier that can accept contact liquid-crystal thermography (LCT) images and return a guarded `TUMOR_LIKE` / `NO_TUMOR_LIKE` result together with OOD/domain evidence. Public infrared datasets such as DMR-IR are auxiliary only and must not be treated as final validation for the client device.

`clinical_claim = NONE`

## Verified target-domain sources

### 1. 2026 LCT review + Malaysian observational examples

- Source: Mohd Sha’ari et al., *Liquid crystal thermography for breast cancer detection: principles, current evidence, limitations, and future directions*, J Egypt Natl Canc Inst (2026), DOI `10.1186/s43046-026-00383-6`.
- Modality: contact LCT / Braster-style RGB thermochromic foil imaging.
- License: article is CC BY 4.0; images are included under the article license unless a figure-specific credit says otherwise.
- Observed labelled examples in the article:
  - Case 1 / Fig. 3: true-positive LCT, biopsy-confirmed invasive breast carcinoma, no special type.
  - Case 2 / Fig. 4: true-negative LCT, biopsy-confirmed fibrocystic change (benign).
- These two cases are useful as **reference-only labelled examples**, not as a train/test dataset. The article explicitly describes the underlying Malaysian cohort as preliminary and unpublished.
- Action: obtain the full-resolution figures and isolate the LCT panels only after checking each figure credit. Record case-level provenance and do not mix ultrasound/mammography panels into the LCT image pool.

### 2. Bichara doctoral thesis (1975/1976)

- Title: *Liquid Crystal Thermography: A New System for Breast Cancer Detection* — Kamal Fouad Bichara, Oregon State University.
- Study description: 75 volunteer women examined using a liquid-crystal thin film, direct-spray liquid crystals, and infrared thermography. Positive thermograms were followed by xeroradiography; the thesis also describes an interpretation scheme evaluated on a much larger historical set.
- Why it matters: it is one of the most direct historical sources for contact-LCT breast images and interpretation patterns.
- Public discovery routes currently found: Google Books record and a Docsity scan/preview.
- Rights status: **REVIEW_REQUIRED**. Do not ingest images into a distributable/commercial training set until the original repository or rights status is verified.
- Action: locate the institutional thesis copy or a rights-cleared scan, then inspect figures/tables for image-level labels and patient/case linkage.

### 3. Stevens & Rogers mouse-tumour LCT paper (1971)

- Title: *Liquid Crystal Thermography of Transplantable Mouse Tumors*.
- DOI: `10.1177/153857447100500404`; PMID `4329631`.
- Modality/species: direct liquid-crystal thermography on transplantable tumours in mice.
- Relevance: this is the closest published precedent to the client’s current tumour-bearing mouse experiment.
- Public availability: abstract/indexing is public; publisher article is restricted. A publisher preview is visible through ResearchGate/SAGE discovery routes.
- Rights status: **COPYRIGHT_RESTRICTED / REFERENCE_ONLY** unless permission or an open copy is found.
- Action: use the methods and qualitative tumour-vs-normal observations for experiment design; do not scrape copyrighted figures into a training corpus.

### 4. Braster prospective pilot study

- Study: *A Prospective Pilot Study on Use of Liquid Crystal Thermography to Detect Early Breast Cancer* (2020), 274 consecutive women enrolled, 255 evaluable after acquisition exclusions.
- Modality: Braster liquid-crystal contact thermography.
- Reference standard: standard breast imaging with biopsy verification in relevant cases.
- Public status: paper is available; no raw labelled thermogram dataset was found publicly.
- Action: preserve as evidence/protocol reference. Do not claim the study images are available for model training.

### 5. INNOMED / Braster multicentre study

- Approx. 3000 women underwent Braster contact LCT in a Polish multicentre observational study.
- Public Braster material describes high-BI-RADS and control groups and automatic/expert thermogram analysis.
- Raw image dataset was not found publicly.
- Action: keep as a high-value potential data-access lead; if needed later, approach the institutions/rights holder rather than assuming public access.

### 6. Sterns et al. contact-LCT IDC cohort

- Study: 420 women with invasive ductal carcinoma underwent liquid-crystal contact thermography.
- Paper reports relationships between thermal abnormality and tumour characteristics, including tumour size.
- Raw images were not found publicly.
- Action: use as biological/feature-design evidence and a potential archival-data lead, not as an image dataset.

## Current target-domain inventory

1. Client device: 9 real mouse contact-TLC images, all tumour-bearing according to the client; frozen as target-domain positives / burden cohort, not binary training evidence.
2. Publication Dataset Zero: 26 extracted LCT reference plates from supplied publication figures; reference-only, labels are not sufficient for clinical classifier ground truth.
3. 2026 review: at least two observed, case-level labelled human contact-LCT examples (one biopsy-confirmed malignant, one biopsy-confirmed benign), pending clean panel extraction and figure-credit verification.
4. Historical sources: several strong leads, but no rights-cleared bulk labelled contact-LCT dataset has yet been found.

## Decision rules

- Never convert a paper caption, our own visual interpretation, or model output into ground truth unless the case diagnosis is explicitly linked by the source.
- Keep species, modality, TLC formulation/profile, device, source and label provenance in the manifest.
- Do not combine IR and contact-LCT as if they are one modality.
- Do not use the 9 client positives as both training and evaluation data.
- For any final binary research head, use animal/patient-level splits and an OOD/abstain gate.

## Immediate next actions

1. Extract and register the rights-cleared LCT panels from the 2026 CC BY review as `REFERENCE_ONLY_LABELLED` examples.
2. Continue searching institutional repositories/theses for image-level contact-LCT material with explicit pathology/control labels.
3. Build an OOD-gated transfer probe that can accept auxiliary IR model scores but abstains on client LCT when the domain distance is beyond the learned source range.
4. Keep DMR-IR as auxiliary model development data while target-domain acquisition continues.
5. When genuine contact-LCT negatives become available, start the first target-domain binary baseline before attempting DINO fine-tuning.
