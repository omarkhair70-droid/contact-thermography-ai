# USM contact-LCT data access request packet

Status: **draft/prepared only — not sent**.

Purpose: request de-identified raw contact liquid-crystal thermography (LCT) images and case-level reference-standard metadata from the current Universiti Sains Malaysia programme so the project can evaluate a true target-domain model without fabricating labels or mixing infrared datasets into LCT.

## Why USM is the priority target

Public USM and 2026 publication records establish that:

- the current programme uses Braster/contact-LCT in an ongoing Malaysian breast cohort;
- 108 participants were planned and 42 had completed the study at manuscript submission;
- the 2026 paper explicitly says the interim cohort is incomplete and the reference standard is not uniformly biopsy-confirmed yet;
- a 2025 USM Radiology dissertation is titled `Sensitivity and Specificity of Liquid Crystal Thermography to Detect Breast Cancer: A Preliminary Study In Hospital Universiti Sains Malaysia` (Dr Nor Hafizzi Mohamad; supervisor Dr Bazli Md Yusoff);
- the USM Pathology postgraduate programme lists Ainnur Bazilah Mohd Sha'ari's ongoing MSc project, `Correlation of Liquid Crystal Thermography and Imaging with Histopathological Diagnosis and Metabolic Profiling in Early Detection of Breast Cancer`;
- the 2026 open-access paper contains two pathology-linked observed LCT exemplars, but it states that no dataset was generated or analysed for the review itself. The underlying cohort therefore requires direct research-data access rather than scraping article figures.

Public source routes:

- 2026 LCT review: `https://doi.org/10.1186/s43046-026-00383-6`
- USM Radiology postgraduate listing: `https://radiology.kk.usm.my/`
- USM Pathology postgraduate research listing: `https://medic.usm.my/pathology-postgraduate/pgstudents-research.html`

## Public contact routes to verify at send time

- Heba Mohammed Arafat — `hebaarafat@usm.my` (also `hebaarafat4@hotmail.com` appears in public publications)
- Wan Faiziah Wan Abdul Rahman — `wfaiziah@usm.my`
- Tengku Ahmad Damitri Al-Astani Tengku Din — `damitri@usm.my`
- Bazli Md Yusoff — `bazliyusoff@usm.my`

These are public academic contact routes. Re-verify them immediately before sending.

## Minimum requested data package

Ask for the smallest de-identified package that preserves scientific validity:

1. original RGB contact-LCT/Braster thermograms;
2. stable de-identified subject/case IDs;
3. image-to-subject and left/right/view mapping;
4. malignant / benign / normal-control outcome per subject;
5. reference-standard provenance per subject (histopathology, imaging follow-up, clinical follow-up, or other documented standard);
6. TLC foil/formulation or temperature-range information, including foil type/batch if available;
7. Braster/device/camera profile;
8. acquisition protocol metadata (view/position, room/acclimatisation protocol, acquisition date/timepoint where available);
9. explicit permission terms for AI/model research;
10. explicit statement whether prototype/commercial R&D use is allowed and whether raw data may be redistributed or must remain private.

Do not request names, national IDs, phone numbers, addresses, medical-record numbers or other unnecessary direct identifiers.

## Data-handling promise

The repository now has a strict intake gate (`scripts/validate_external_lct_intake.py`). Incoming data remains non-trainable by default. A case cannot become `TRAIN_CANDIDATE` until rights, label provenance, TLC/device/acquisition provenance and subject grouping are explicit. Subject-level grouping is preserved to prevent leakage across training and evaluation.

The current project is research-only and retains `clinical_claim=NONE`; the request is not based on a claim that the current model diagnoses cancer.

## Draft email

Subject: Research data request — de-identified contact liquid-crystal thermography images for AI validation

Dear Dr Arafat, Assoc. Prof. Wan Faiziah, Dr Tengku Din and Dr Bazli,

I am working on a research software/model pipeline specifically for contact liquid-crystal thermography (LCT), with profile-aware TLC processing, morphology/QC features and DINOv2 image representations. We found your current USM LCT programme and the 2026 review describing the ongoing Malaysian cohort, as well as the related USM Radiology and Pathology postgraduate work.

Our main scientific limitation is not model code but access to a coherent target-domain cohort containing genuine positive and negative/control cases with case-level reference-standard provenance. We are deliberately not using infrared thermography as if it were interchangeable with contact-LCT, and we do not create synthetic negative labels.

Would your group be open to discussing research access to a de-identified subset of the original RGB contact-LCT/Braster thermograms, with stable subject IDs, image/view mapping, malignant/benign/control outcome and its reference-standard provenance, plus TLC/device/acquisition metadata where available?

We can work under restricted/no-redistribution terms. If model-development or commercial-prototype use requires a separate agreement, we would prefer to document that explicitly rather than assume publication access grants raw-data reuse rights. We do not need participant names or other direct identifiers.

The immediate goal would be a subject-level research validation of a target-domain LCT classifier with an untouched evaluation split and explicit out-of-domain abstention. Any outputs would remain research-only and non-diagnostic unless a proper validation programme supports stronger claims.

If sharing raw images is not possible, even guidance on the appropriate USM data-access/ethics route or the correct investigator to approach would be very helpful.

Thank you for considering the request.

Best regards,
Omar Khair

## Send boundary

Do not send this message automatically. Sending is an external action and requires explicit user approval in chat after the recipient route is re-verified.
