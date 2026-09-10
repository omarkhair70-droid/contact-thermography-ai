# Lane E — Client-device TLC domain adaptation + murine tumor-burden validation

## Current client facts

The first real client-device cohort is available: 9 contact-TLC images (`IMG-20260910-WA0030.jpg` through `IMG-20260910-WA0038.jpg`). The client reports that all animals are tumor-bearing mice, acquisition conditions are approximately the same, body weight is similar, and tumor size is the intended biological variable that differs between animals.

Therefore this cohort is **not** a tumor-vs-control training set. It is useful for device-domain adaptation, image-quality review, within-cohort response analysis, and a blind tumor-response-burden experiment. Tumor-size measurements must be compared only after the blind ranking is frozen.

## Real-device selector — implemented in the app

The generic client profile originally inherited publication-reference thresholds (`S > 50`, `V > 38`, 18 px minimum component) and the publication central-disk assumption. On the real device images that over-segmented dim saturated background texture. This was a concrete domain-shift failure.

The client path now uses `app/services/client_device_domain.py` for `client-device-tlc-pending` whole-image fallback uploads. Its current provisional selector is:

- saturation > 55
- pixel value > 85
- connected component area >= 150 px after 256x256 letterboxing
- connected component mean value >= 160
- 3x3 opening + 5x5 closing

These are visible-response selection parameters, **not temperature calibration values**. They are isolated to the client-device profile. Publication-reference uploads retain their original analysis path. White glare is largely rejected through low saturation, while clipped/edge-adjacent response still requires human QC review.

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

Semantics: **thermochromic response-area ranking only**. It is not tumor size, tumor probability, cancer risk, or diagnosis. WA0036 and WA0033 are more threshold-sensitive than WA0037/WA0034/WA0030, so that uncertainty must remain visible.

The batch runner now writes `blind_freeze_manifest.json`, hashing every original image plus `blind_response_area_ranking.csv`. This creates an auditable pre-label freeze before any tumor-size mapping is revealed.

## Measured colour-domain shift vs publication references

Using the 26 publication-derived reference feature rows as baseline, the client cohort shows strong acquisition/colour-domain shift. Median feature differences measured in reference-IQR units were approximately:

- Hue mean: -0.19 IQR
- Hue spread: +0.85 IQR
- Saturation mean: -0.80 IQR
- Value/brightness mean: **+8.31 IQR**
- Lab a*: **-4.95 IQR**
- Lab b*: +0.87 IQR

This is enough to reject the assumption that publication colour thresholds are interchangeable with the real device. The shift may reflect TLC formulation, camera/white balance, illumination, contact setup, or combinations of them; it does not identify disease.

## DINOv2 execution gate — artifact path completed

`scripts/client_device_batch.py --attempt-dino` now preserves the live official-DINO outputs instead of discarding the embedding vectors. When the pinned official backbone is available, the run writes:

1. `client_device_dinov2_embeddings.npy` — 9 x 384 float embeddings;
2. `client_device_dinov2.csv` — per-image live DINO/reference research outputs;
3. `client_client_dinov2_cosine_distance.csv` — the 9 x 9 within-client distance matrix;
4. `client_reference_nearest_dinov2.csv` — nearest publication reference and distance per image;
5. `dinov2_embedding_domain_shift.json` — client-vs-reference shift relative to the empirical reference self-neighbour distance distribution.

The embedding-domain metric is deliberately referenced to the 26-plate corpus's own nearest-neighbour distance distribution. It is a visual/domain-shift measure, not a tumor score.

Do not replace the official DINOv2 encoder with surrogate/random embeddings. If the pinned backbone is unavailable, the batch must report DINO unavailable rather than fabricate results.

## Post-freeze tumor-size comparison — ready before labels arrive

`scripts/evaluate_tumor_size_mapping.py` is the only intended first-cohort tumor-size comparison path. It verifies that the ranking CSV still matches the pre-label SHA256 freeze, requires the tumor-size mapping to match the exact frozen cohort, and then reports descriptive association only:

- Spearman response-area vs tumor-size association;
- Pearson response-area vs tumor-size association;
- descending-rank agreement;
- a descriptive straight-line fit and R².

This is **not** model training and must not be presented as clinical validation on 9 animals. The 9 images remain the blind first-cohort evaluation set; they must not be fine-tuned and then reused as their own proof of performance.

## Literature basis for the experiment design

- Stevens JD, Rogers W. **Liquid Crystal Thermography of Transplantable Mouse Tumors.** Vascular Surgery. 1971;5(4):186–192. DOI `10.1177/153857447100500404`. Direct historical precedent for cholesteric liquid-crystal thermography over transplantable mouse tumors.
- Tepper M et al. **Thermographic investigation of tumor size, and its correlation to tumor relative temperature, in mice with transplantable solid breast carcinoma.** Journal of Biomedical Optics. 2013;18(11):111410. DOI `10.1117/1.JBO.18.11.111410`. Infrared rather than contact TLC, so not training data; useful only as experiment-design precedent for comparing independent tumor measurements with thermal-image response.
- TLC calibration literature shows colour-temperature response sensitivity to illumination spectrum, viewing angle, white balance, hysteresis, film thickness and other setup variables. This supports a formulation/device-specific profile rather than one global hue rule.

No public modern labelled contact-TLC mouse-tumor dataset has been identified in the targeted search to date. Infrared datasets remain a separate modality and must not be silently mixed into contact-TLC training.

## Remaining execution gates

1. Run the frozen 9 originals through the current batch and archive CSV/JSON/contact-sheet outputs outside the source tree or in approved private experiment storage.
2. Human-review all 9 masks; do not change the frozen ranking after tumor-size labels are known.
3. Run official live DINOv2 on the same frozen normalized inputs and archive the five DINO artifacts listed above.
4. Only then compare against independently supplied tumor sizes through `scripts/evaluate_tumor_size_mapping.py`.
5. If later controls/pre-injection animals arrive, treat them as a separate dataset extension before investigating tumor-vs-control classification.
6. Keep `clinical_claim=NONE` and absolute-temperature interpretation disabled until formulation-specific calibration and appropriate validation exist.
