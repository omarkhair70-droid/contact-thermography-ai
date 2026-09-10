# Contact Thermography AI — Master Handoff

## Source of truth

Repository: `omarkhair70-droid/contact-thermography-ai`

Baseline: **RC0.5 DINOv2 Integrated**.

From this point onward, do not create independent project copies. All work must land in this repository through isolated feature branches and be integrated into `integration` before promotion to `main`.

## What the product is

A browser-based application for analysis of **contact liquid crystal thermography**, not infrared-camera thermography.

The user creates an examination, uploads one or more contact-thermography images, assigns LEFT/RIGHT and position metadata, and receives image QC, extracted plates, thermochromic response/morphology features, DINOv2 reference analysis, bilateral comparison, visual maps, persistence and a report.

## Current green capabilities

- FastAPI application and browser dashboard.
- Multi-image exam upload.
- Automatic circular plate extraction.
- Engineering QC gate.
- LCT colour/morphology features.
- LEFT/RIGHT/POSITION metadata.
- Translation registration and structural/Hue bilateral maps.
- Exam persistence, history and HTML report.
- Real DINOv2 ViT-S/14 embeddings generated on Kaggle T4 GPU: 26 x 384.
- Fused 18 explicit LCT + 384 DINOv2 reference model.
- Learned DINOv2 bilateral representation: L, R, |L-R| and L*R.
- Docker and provider-neutral GPU training harness.

## TLC formulation / device domain requirement

The client stated that images from their own device will look different from the supplied publication/reference images because their **TLC formulation/material is different**. This is a real domain shift and must not be ignored.

Every examination must preserve a `tlc_profile_id` and, when known, a `device_profile_id`. Colour thresholds and any colour-to-temperature interpretation must be resolved per TLC profile rather than assumed globally. The 26 publication-derived plates belong to `reference-publication-unknown`; client-device images belong to `client-device-tlc-pending` until their exact formulation/calibration is known.

Geometry/morphology and LEFT/RIGHT comparison may remain more transferable after profile-specific signal segmentation, but this must be tested on client-device images. See `docs/TLC_VARIANT_HANDLING.md` and `config/tlc_profiles.example.json`.

## Hard scientific boundary

The current 26 images are publication/reference material, not a clinically representative labelled dataset. All anomaly/asymmetry outputs are **reference-only**. Never surface them as cancer probability or diagnosis. The clinical classifier lane stays disabled until reliable patient-level labels / ground truth exist and a proper validation protocol is run.

## Immediate product target

A client-demo candidate that can be opened from a public/staging URL and run end-to-end:

`Login/demo entry -> New Exam -> Upload -> QC -> visual analysis -> DINO analysis -> bilateral comparison -> findings -> saved exam -> report`

## Parallel lanes

1. `feat/live-dinov2` — make DINOv2 inference run on newly uploaded plates and feed the fused/pair heads. It must preserve `tlc_profile_id` and treat client-device images as a separate domain/profile.
2. `feat/product-ui` — turn the current technical dashboard into a polished examination workflow without altering model semantics. The exam workflow should expose/select the TLC/device profile without forcing the user to understand model internals.
3. `feat/oracle-deploy` — production/staging Docker deployment, PostgreSQL and durable object storage on Oracle.
4. `feat/qa-hardening` — integration tests, bad-input tests, TLC-profile/domain-shift tests, non-clinical guardrails, regression suite and demo smoke script.

Agents must not edit another lane's owned files unless integration requires it. `integration` is the only branch where cross-lane changes are reconciled.
