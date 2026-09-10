# Contact Thermography AI

Canonical repository for an **AI-assisted Contact Liquid Crystal Thermography (LCT) analysis platform**.

## Product flow

`New Exam -> Upload -> Plate extraction -> QC -> LCT signal/morphology -> DINOv2 reference analysis -> LEFT/RIGHT registration -> bilateral analysis -> persistence -> report`

This is a **web application**, not a Flutter/mobile app. A mobile client can be added later on top of the same API.

## Current baseline: RC0.5

- 26 reference contact thermograms extracted from supplied publication/reference figures.
- Deterministic LCT colour/morphology engine.
- Upload-time QC.
- LEFT / RIGHT / POSITION metadata and live bilateral registration.
- FastAPI product shell with history and report generation.
- Real `dinov2_vits14` embeddings generated on Kaggle T4 GPU: **26 x 384**.
- Fused LCT + DINOv2 reference results.
- DINOv2 bilateral pair representation on 11 provisional reference pairs.
- Live `dinov2_vits14` inference for uploaded normalized plates, fused LCT/DINO
  scoring and learned LEFT/RIGHT pair scoring.
- TLC/device profile provenance with profile-scoped normalization, segmentation
  and engineering-QC thresholds.
- Docker runtime and provider-neutral GPU training harness.

## Important clinical status

Current learned scores are **reference/research unusualness signals only**. They are not cancer probabilities, and no diagnostic sensitivity/specificity claim is made. A supervised clinical head is gated on reliable patient-level ground truth and proper validation.

## Local setup

```bash
pip install -r requirements-vision.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

The first live analysis downloads the official DINOv2 ViT-S/14 weights from a
pinned `facebookresearch/dinov2` revision into the standard PyTorch Hub cache.
Set `DINOV2_DEVICE=cpu` or `DINOV2_DEVICE=cuda` to override automatic device
selection. TLC profiles are resolved from `config/tlc_profiles.json`; each
metadata item may carry `tlc_profile_id` and optional `device_profile_id`.

Regression tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

Docker:

```bash
docker compose up --build
```

The canonical Git source keeps generated examination files and runtime databases out of version control. Small reference ledgers are committed as text so the baseline can be reproduced without carrying forward old ZIP snapshots.

## Branching

- `main` — last green stable baseline.
- `integration` — canonical active integration branch.
- `feat/live-dinov2` — live learned-vision inference.
- `feat/product-ui` — client-demo examination UI.
- `feat/oracle-deploy` — Oracle/PostgreSQL staging deployment.
- `feat/qa-hardening` — regression, failure-mode and claim-guardrail tests.

All agent work should open a pull request into `integration`; do not create independent project copies.

See `docs/MASTER_HANDOFF.md` for the current state and agent boundaries.
