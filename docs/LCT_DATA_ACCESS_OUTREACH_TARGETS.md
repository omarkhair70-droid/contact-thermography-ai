# LCT data-access outreach targets

Public repository searches have not produced a downloadable, case-labelled raw contact-LCT dataset large enough for native binary training. The next acquisition route is direct research-data access, while preserving the existing rights/readiness gates.

## Priority 1 — Universiti Sains Malaysia target-domain program

This is currently the strongest external contact-LCT data lead.

The 2026 USM review describes an observational Malaysian cohort with **108 participants planned and 42 completed at manuscript submission**. Its interim analysis reported sensitivity 72.2% and specificity 58.3%, but the cohort was incomplete and the reference standard was not uniformly biopsy-confirmed at that stage. The underlying raw RGB contact-LCT images were not released publicly with the review.

Two additional USM records indicate that case-linked LCT material exists inside the same research program:

- 2025 Radiology dissertation by Dr Nor Hafizzi Mohamad: `Sensitivity and Specificity of Liquid Crystal Thermography to Detect Breast Cancer: A Preliminary Study In Hospital Universiti Sains Malaysia`, supervised by Dr Bazli Md Yusoff.
- Ongoing Pathology MSc project by Ainnur Bazilah Mohd Sha'ari: `Correlation of Liquid Crystal Thermography and Imaging with Histopathological Diagnosis and Metabolic Profiling in Early Detection of Breast Cancer`, supervised by AP Dr Wan Faiziah Wan Abd Rahman.
- A 2024 TransMec case-series abstract from the USM group compared LCT with conventional imaging and histopathological examination.

These records may represent overlapping subjects. Any supplied data must therefore preserve stable de-identified subject IDs so overlap can be detected before train/validation/test construction.

Request scope: de-identified original RGB contact-LCT images, stable subject/case identifier, image-to-subject mapping, malignant/benign/control reference-standard provenance, acquisition date/timepoint where available, TLC foil/formulation/range, device/camera profile, acquisition protocol, and explicit permission terms for AI research/model development and any intended commercial prototype use.

Publicly discoverable academic contact routes to verify before sending include:

- Heba Mohammed Arafat — `hebaarafat@usm.my` (an alternate public address `hebaarafat4@hotmail.com` also appears in academic publications).
- Tengku Ahmad Damitri Al-Astani Tengku Din — `damitri@usm.my`.
- AP Dr Wan Faiziah Wan Abd Rahman — `wfaiziah@usm.my`.
- Dr Bazli Md Yusoff — `bazliyusoff@usm.my`.

Do not ingest the 42-case interim cohort as if every label were biopsy-confirmed unless the supplied case-level reference standard verifies that. The two observed biopsy-linked figures in the 2026 review remain reference exemplars only.

## Priority 2 — 2020 Braster prospective investigators

Target investigator: Diana Hodorowicz-Zaniewska / Jagiellonian University group. The prospective study enrolled 274 women and analysed 255 after acquisition exclusions, with cancer/noncancer comparison and histology-linked disease assessment. No raw image dataset has been found publicly.

The article is CC BY-NC 4.0. That license applies to the publication, not automatically to any unpublished raw thermography dataset, and it does not authorize commercial reuse. Request an explicit data-use agreement if the prototype may become commercial.

Request scope: de-identified original Braster RGB thermograms, patient-level group IDs, pathology/noncancer outcome and provenance, acquisition/foil range, device profile, image position/view, plus explicit research and commercial-use terms.

A public corresponding-author route in the article is `diana.hodorowicz-zaniewska@uj.edu.pl`; verify before sending.

## Priority 3 — historical archives and mouse LCT

Davison et al. 1972, Bichara 1975 thesis, and Stevens/Rogers 1971 mouse LCT are strong morphology/method sources but their image reuse rights are restricted or unclear. Seek archive/rightsholder access rather than copying publisher previews into training data.

The Stevens/Rogers mouse study is the closest historical modality/species match to the client: it used liquid-crystal thermography in tumor-bearing mice and reinforces that tumor-associated surface response may be higher or lower depending on perfusion. It does not provide an open labelled raw-image dataset.

For the Bichara thesis, prioritize Oregon State University archival/library channels because third-party preview sites do not establish reuse rights.

## Minimum requested data contract

For any supplied dataset, require at minimum:

- stable de-identified subject/case ID;
- image-to-subject mapping;
- contact-LCT modality confirmation;
- species;
- label and label provenance/reference standard;
- TLC formulation/foil/range if known;
- device/camera/acquisition profile if known;
- view/position and timepoint if multiple images exist;
- explicit data-use and redistribution terms.

Multiple views or longitudinal images from one subject must remain grouped in train/validation/test splitting. Overlapping USM studies must be deduplicated at subject level before combining them.

Do not request unnecessary personal identifiers. Do not ingest a dataset into the native trainer until its rights, label provenance, TLC/device provenance, and split grouping pass the repository validators.

`clinical_claim=NONE` remains mandatory during this research stage.
