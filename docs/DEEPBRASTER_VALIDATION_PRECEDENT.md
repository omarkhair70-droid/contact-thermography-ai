# DeepBraster / historical contact-LCT precedent

Purpose: capture public evidence that should shape our research architecture and validation design without treating proprietary Braster data or claims as our own model performance.

## What the public record establishes

BRASTER worked on automated interpretation of contact liquid-crystal thermograms before this project. Public company reporting from 2016 describes a combined ThermaCRAC + ThermaRAK pool of 325 examinations, including 180 described as non-pathological and 145 pathological, with 260 examinations assigned to training and 65 to validation. The same report described an automatic interpretation stage with reported sensitivity 73.7% and specificity 76.9%. These historical figures are precedent, not expected performance for our model.

The later INNOMED programme / NCT03735550 was designed as a much larger multicentre data-collection programme, on the order of 3000 women, with thermographic imaging plus standard imaging/clinical evidence and histopathology where applicable. Public records and company material describe subsequent algorithm-development work (including DeepBraster), but the underlying case-labelled raw thermograms are not publicly downloadable and must remain `DATA_ACCESS_REQUIRED`.

The Braster public website also exposes case-linked contact-LCT examples with ultrasound/mammography/pathology context. Those pages are useful for morphology and failure-mode review, but the site is copyrighted Braster material; they are `COPYRIGHT_RESTRICTED_REFERENCE` and are not copied into our training set without permission.

## Design lessons we adopt

1. Split by subject/examination, never by individual image. Multiple foils/views from the same subject must remain in one split.
2. Keep side-specific and bilateral representations. A useful output can include whole-exam signal plus left/right or paired asymmetry evidence rather than one opaque global score.
3. Prefer relative and profile-aware thermochromic information over a universal absolute-temperature threshold. TLC foil chemistry, temperature response interval, acquisition conditions, illumination, and white balance change the visible colour response.
4. Do not encode `hot = malignant`. Public LCT literature contains benign thermal abnormalities, false positives, and false negatives; animal work also shows that tumor-associated surface contrast can vary with model and physiology.
5. Keep a domain/OOD gate. A model trained on infrared thermography or another TLC/device formulation cannot silently produce a target-domain disease score.
6. Preserve an untouched validation cohort after model/threshold selection. Public Braster reporting explicitly separated training and validation data; our native path should do the same once the target cohort is large enough.
7. Report sensitivity, specificity, AUROC/AUPRC, calibration and confusion counts together. Small enriched cohorts can make a single accuracy number misleading.

## What this does not establish

- It does not grant access or reuse rights to the Braster/ThermaCRAC/ThermaRAK/INNOMED images.
- It does not validate our current 417-feature representation.
- It does not make DMR-IR an LCT training set.
- It does not open the native binary gate for the nine client mice, because those nine are all tumor-bearing and do not provide genuine same-device negatives.
- It does not support a clinical cancer-probability claim.

## Current project consequence

The integrated native path remains:

`client-device image -> profile-aware TLC response + morphology + engineering QC + normalized DINOv2 embedding -> 417-feature contract -> same-domain binary model only when the readiness gate is satisfied`.

Until then, target images can be analyzed for response morphology, DINO representation, OOD/domain shift and tumor-burden association, but `clinical_claim=NONE` remains mandatory.

## Public sources

- BRASTER S.A. half-year 2016 report: https://www.braster.eu/media/wysiwyg/raportypolroczne/1_BRASTER_Raport_polroczny_2016.pdf
- Braster technology / case studies: https://www.braster.eu/en/technology
- Hodorowicz-Zaniewska et al. prospective LCT pilot: https://doi.org/10.1177/1534735420915778
- NCT03735550 / INNOMED programme: https://clinicaltrials.gov/study/NCT03735550
- 2026 LCT review and Malaysian observational examples: https://doi.org/10.1186/s43046-026-00383-6
