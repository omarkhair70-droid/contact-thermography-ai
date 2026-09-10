# LCT data-access outreach targets

Public repository searches have not produced a downloadable, case-labelled raw contact-LCT dataset large enough for native binary training. The next acquisition route is direct research-data access, while preserving the existing rights/readiness gates.

## Priority 1 — Universiti Sains Malaysia 2026 observed LCT cohort

Target investigators: Heba Mohammed Arafat and Tengku Ahmad Damitri Al-Astani Tengku Din, Universiti Sains Malaysia. The 2026 review includes two observed biopsy-linked contact-LCT examples and describes an ongoing/recent local cohort, but states that no dataset was generated or analysed for the review itself.

Request scope: de-identified RGB contact-LCT images from the underlying observed cohort, one case/subject identifier, malignant/benign/control ground truth provenance, acquisition/TLC foil profile, device profile, and permission terms for research/model development. Ask for subject-level metadata rather than names or direct identifiers.

Publicly discoverable academic contact routes include `hebaarafat@usm.my` and `damitri@usm.my`. Verify current addresses before sending.

## Priority 2 — 2020 Braster prospective investigators

Target investigator: Diana Hodorowicz-Zaniewska / Jagiellonian University group. The prospective study enrolled 274 women and linked thermography with histology/noncancer controls, but no raw image dataset was found publicly. The article is CC BY-NC 4.0, which does not grant commercial reuse of an unpublished raw dataset.

Request scope: de-identified original Braster RGB thermograms, patient-level group IDs, pathology/noncancer outcome, age-band and acquisition/foil profile, plus an explicit data-use agreement for AI research and any intended commercial prototype use.

A public corresponding-author route in the article is `diana.hodorowicz-zaniewska@uj.edu.pl`; verify before sending.

## Priority 3 — historical archives

Davison et al. 1972, Bichara 1975 thesis, and Stevens/Rogers 1971 mouse LCT are strong morphology/method sources but their image reuse rights are restricted or unclear. Seek archive/rightsholder access rather than copying publisher previews into training data.

For the Bichara thesis, prioritize Oregon State University archival/library channels because the author is deceased and third-party preview sites do not establish reuse rights.

## Minimum requested data contract

For any supplied dataset, require at minimum: stable de-identified subject/case ID, image-to-subject mapping, contact-LCT modality confirmation, species, label and label provenance, TLC formulation/foil profile if known, device/acquisition profile if known, and explicit data-use terms. Multiple views from one subject must remain grouped in train/validation/test splitting.

Do not request unnecessary personal identifiers. Do not ingest a dataset into the native trainer until its rights, label provenance, and split grouping pass the repository validators.

`clinical_claim=NONE` remains mandatory during this research stage.
