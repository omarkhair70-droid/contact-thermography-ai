# Client-device batch run

Run from repository root after placing the 9 client originals in a local/private directory that is **not committed**:

```bash
python scripts/client_device_batch.py \
  --input-dir /path/to/client-images \
  --output-dir runtime/client-device-first-pass \
  --pattern 'IMG-20260910-WA003*.jpg'
```

This deterministic pass writes:

- `client_device_features.csv`
- `client_device_qc.csv`
- `threshold_sweep.csv`
- `blind_response_area_ranking.csv`
- `blind_freeze_manifest.json`
- `colour_domain_shift.json`
- `client_device_contact_sheet.png`
- `manifest.json`

The blind-freeze manifest hashes every original image plus the ranking file before tumor-size labels are revealed.

## Official DINOv2 pass

In an environment that can load the pinned official DINOv2 backbone, run:

```bash
python scripts/client_device_batch.py \
  --input-dir /path/to/client-images \
  --output-dir runtime/client-device-first-pass \
  --pattern 'IMG-20260910-WA003*.jpg' \
  --attempt-dino
```

The same run additionally writes:

- `client_device_dinov2.csv`
- `client_device_dinov2_embeddings.npy` (9 x 384)
- `client_client_dinov2_cosine_distance.csv`
- `client_reference_nearest_dinov2.csv`
- `dinov2_embedding_domain_shift.json`

Never substitute random/surrogate embeddings when the official backbone is unavailable.

## After independent tumor sizes are supplied

Prepare a private CSV containing exactly the frozen filenames and an independent size measurement, for example:

```csv
source_image,tumor_size
IMG-20260910-WA0030.jpg,12.3
...
```

Then run:

```bash
python scripts/evaluate_tumor_size_mapping.py \
  --ranking runtime/client-device-first-pass/blind_response_area_ranking.csv \
  --freeze-manifest runtime/client-device-first-pass/blind_freeze_manifest.json \
  --labels /path/to/private/tumor_sizes.csv \
  --size-column tumor_size \
  --output-dir runtime/client-device-first-pass/tumor-size-comparison
```

The evaluator refuses to run if the ranking no longer matches the pre-label SHA256 freeze or if the supplied mapping does not match the exact frozen cohort. It reports descriptive rank/continuous association only; it does not train a model or convert the nine animals into clinical validation.

Do not commit raw client images or private tumor-size measurements by default. All outputs remain research-only and `clinical_claim` must remain `NONE`.
