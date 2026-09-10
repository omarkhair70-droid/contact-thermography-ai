# Client-device batch run

Run from repository root after placing client images in a local directory that is **not committed**:

```bash
python scripts/client_device_batch.py \
  --input-dir /path/to/client-images \
  --output-dir runtime/client-device-first-pass \
  --pattern 'IMG-20260910-WA003*.jpg'
```

To add official DINOv2 outputs in an environment that can load the pinned backbone:

```bash
python scripts/client_device_batch.py \
  --input-dir /path/to/client-images \
  --output-dir runtime/client-device-first-pass \
  --pattern 'IMG-20260910-WA003*.jpg' \
  --attempt-dino
```

Expected deterministic outputs are `client_device_features.csv`, `threshold_sweep.csv`, `blind_response_area_ranking.csv`, `colour_domain_shift.json`, `client_device_contact_sheet.png`, and `manifest.json`. If DINOv2 is available, `client_device_dinov2.csv` is added.

Do not commit the raw client images by default. The output semantics remain research-only; `clinical_claim` must remain `NONE`.
