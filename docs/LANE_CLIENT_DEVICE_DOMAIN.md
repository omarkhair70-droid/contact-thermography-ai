# Lane E — Client-device TLC domain adaptation + murine tumor-burden validation

## Current client facts

The first real client-device cohort is now available: 9 contact-TLC images (`IMG-20260910-WA0030.jpg` through `IMG-20260910-WA0038.jpg`). The client reports that all animals are tumor-bearing mice, acquisition conditions are approximately the same, body weight is similar, and tumor size is the intended biological variable that differs between animals.

Therefore this cohort is **not** a tumor-vs-control training set. It is useful now for device-domain adaptation, image-quality review, within-cohort response analysis, and a blind tumor-response-burden experiment. Tumor-size measurements should be compared only after the blind ranking is frozen.

## First real-device review

The generic `client-device-tlc-pending` mask inherited the publication reference thresholds (`S > 50`, `V > 38`, 18 px minimum component) and used the central-disk assumption. On the real device images this over-segmented dim saturated background texture. This is a concrete domain-shift failure, not merely a cosmetic difference.

A separate experimental client selector was therefore added in `app/services/client_device_domain.py`. Its current provisional values are:

- saturation > 55
- pixel value > 85
- connected component area >= 150 px after 256x256 letterboxing
- connected component mean value >= 160
- 3x3 opening + 5x5 closing

These numbers are **not temperature calibration values** and are not yet promoted into the main analysis path. Their purpose is to isolate visible thermochromic response while rejecting low-brightness saturated background noise. White glare is mostly rejected because it has low saturation; clipped response regions still require QC review.

## Blind response-area ranking

A 45-configuration threshold sweep was run over the 9-image cohort using saturation values 45–65, pixel-value thresholds 70/85/100, and component-mean-value thresholds 150/160/170. The median within-cohort response-area rank was:

| order | image | median response-area fraction | median rank | rank IQR |
|---:|---|---:|---:|---:|
| 1 | WA0037 | 0.1426 | 1 | 1–3 |
| 2 | WA0036 | 0.1240 | 2 | 1–4 |
| 3 | WA0034 | 0.1222 | 3 | 2–4 |
| 4 | WA0033 | 0.1103 | 3 | 2–4 |
| 5 | WA0032 | 0.1061 | 5 | 5–6 |
| 6 | WA0035 | 0.1066 | 6 | 5–6 |
| 7 | WA0038 | 0.0608 | 7 | 7–8 |
| 8 | WA0031 | 0.0447 | 8 | 7–8.5 |
| 9 | WA0030 | 0.0378 | 9 | 8–9 |

Semantics: this is a **thermochromic response-area ranking only**. It is not tumor size, tumor probability, cancer risk, or diagnosis. The correct next validation is to freeze this ranking and compare it against the client’s true tumor-size mapping when available.

The broad rank ranges for WA0036 and WA0033 show that their measured area is more threshold-sensitive than WA0037/WA0034/WA0030. That uncertainty must be retained rather than hidden.

## Measured colour-domain shift vs publication references

Using the 26 publication-derived reference feature rows as the baseline, the first client cohort shows a strong acquisition/colour-domain shift. Median feature differences measured in reference IQR units were approximately:

- Hue mean: -0.19 IQR
- Hue spread: +0.85 IQR
- Saturation mean: -0.80 IQR
- Value/brightness mean: **+8.31 IQR**
- Lab a*: **-4.95 IQR**
- Lab b*: +0.87 IQR

The large brightness and Lab-a shifts are enough to reject the assumption that publication colour thresholds can be treated as interchangeable with the real client device. These shifts can reflect TLC formulation, camera/white-balance, illumination, contact setup, or combinations of them; they do not identify disease.

## DINOv2 gate

The batch runner can optionally execute the integrated official DINOv2 path with `--attempt-dino`. DINO outputs must be generated only in an environment where the pinned official backbone is available. Do not substitute surrogate/random embeddings.

For the 9-image cohort, the DINO analysis should save:

1. 384-d embedding per image,
2. nearest publication reference plates and cosine distances,
3. publication-reference unusualness signal,
4. pairwise 9x9 client-cohort cosine-distance matrix,
5. a client-vs-reference embedding-domain shift summary.

Because the reference corpus uses a different TLC domain, cross-domain DINO scores remain research signals. Within-client-cohort distances are more directly interpretable for this stage.

## Literature basis for the experiment design

- Stevens JD, Rogers W. **Liquid Crystal Thermography of Transplantable Mouse Tumors.** Vascular Surgery. 1971;5(4):186–192. DOI `10.1177/153857447100500404`. This is direct historical precedent for cholesteric liquid-crystal thermography over transplantable mouse tumors.
- Tepper M et al. **Thermographic investigation of tumor size, and its correlation to tumor relative temperature, in mice with transplantable solid breast carcinoma.** Journal of Biomedical Optics. 2013;18(11):111410. DOI `10.1117/1.JBO.18.11.111410`. This is infrared rather than contact TLC, so it is not training data for this product; it supports evaluating thermal-image area against independent caliper tumor-size measurements.
- TLC calibration literature consistently shows colour-temperature response is sensitive to illumination spectrum, viewing angle, white balance, hysteresis, film thickness and other setup variables. This supports a formulation/device-specific profile rather than one global hue rule.

No public modern labelled contact-TLC mouse-tumor dataset has been identified in the targeted search to date. Infrared datasets must remain a separate modality and must not be silently mixed into contact-TLC training.

## Next execution gate

1. Run `scripts/client_device_batch.py` on all 9 originals and archive the feature/ranking/contact-sheet outputs.
2. Review every mask visually; adjust only the client-domain selector if a mask includes setup/background or misses obvious TLC response.
3. Run official live DINOv2 on the frozen normalized images and save client-client plus client-reference embedding analyses.
4. Freeze the blind response ranking before tumor-size labels are revealed.
5. When the client supplies the image-to-tumor-size mapping, calculate Spearman rank association and simple continuous association without fitting and validating on the same 9 animals.
6. If later control/pre-injection animals arrive, treat them as a separate extension and only then investigate tumor-vs-control classification.
7. Keep `clinical_claim=NONE` and absolute temperature disabled until the required formulation-specific calibration/clinical evidence exists.
