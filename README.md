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
- Fused LCT + DINOv2 reference head.
- DINOv2 bilateral pair representation on 11 provisional reference pairs.
- Docker runtime and GPU training harness.

## Important clinical status

Current learned scores are **reference/research unusualness signals only**. They are not cancer probabilities, and no diagnostic sensitivity/specificity claim is made. A supervised clinical head is gated on reliable patient-level ground truth and proper validation.

## Setup

The repository keeps generated runtime files out of Git. Reference images/model artifacts are stored in one compact bundle.

```bash
python scripts/unpack_reference_bundle.py
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

Docker:

```bash
python scripts/unpack_reference_bundle.py
docker compose up --build
```

## Branching

- `main` — last green stable baseline.
- `integration` — canonical active integration branch.
- feature branches — isolated parallel agent lanes.

See `docs/MASTER_HANDOFF.md` for the current state and agent boundaries.
