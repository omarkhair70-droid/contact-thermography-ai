# MumGuard unified session-fusion handoff — 2026-09-12

## Canonical product architecture

The software target is a **human-first bilateral MumGuard examination**, not an isolated-JPEG mouse classifier.

```text
MumGuard Contact-TLC captures
  -> per-frame acquisition/QC + visible TLC response support
  -> RGB/HSV signal extraction
  -> relative thermal field (or calibrated temperature field when profile calibration exists)
  -> overlap registration / de-duplication for multiple captures of each side
  -> LEFT side field + RIGHT side field
  -> multiscale local thermal anomaly analysis
  -> mirrored bilateral comparison
  -> three primary measurement channels
       1. focal/core hyperthermia-like evidence
       2. right-vs-left asymmetry
       3. abnormal skin thermal behaviour
  -> optional local DINO patch novelty evidence
  -> fused evidence maps + session scores
  -> replaceable downstream decision/calibration layer
```

The mouse cohort remains development/evaluation material. Mouse labels do not define the measurement architecture.

## Integrated foundation

PR #42 is built directly on the full PR #41 Codex research foundation, including:

- explicit acquisition/domain provenance;
- full-field geometry and observability state handling;
- local region-vs-surrounding analysis;
- frozen DINOv2 patch descriptors;
- human-first intake/split contracts;
- abstention and missing-channel handling;
- the nine-subject bounded development challenge and its artifacts.

## Added signal/session lane

Implemented modules:

- `app/services/tlc_signal_processing.py`
  - HSV/Hue signal extraction;
  - relative thermal response map;
  - empirical Hue-to-temperature upgrade path;
  - static summaries;
  - transient summaries for future ordered timepoints.

- `app/services/thermal_anomaly_engine.py`
  - label-free multiscale local-vs-surrounding comparison;
  - signed contrast;
  - cross-scale persistence;
  - anomaly/evidence map;
  - adapter for optional visual/AI evidence fusion.

- `app/services/bilateral_session_engine.py`
  - sequential overlap registration;
  - overlap de-duplication through common side fields;
  - coverage accounting;
  - mirrored LEFT/RIGHT comparison;
  - three-channel session evidence.

- `app/services/mumguard_session_fusion.py`
  - end-to-end MumGuard session runtime;
  - visible-TLC support is explicitly not relabeled as confirmed tissue contact;
  - optional DINO patch novelty;
  - persistent session evidence JSON and map bundle.

## Live API/report wiring

`POST /api/exams/analyze` now automatically builds `mumguard_session_evidence` when a client-profile examination contains LEFT and RIGHT Contact-LCT captures.

The result contains:

- the three session scores;
- side reconstruction provenance and offsets;
- bilateral support statistics;
- whether AI evidence was included;
- URLs to persisted session JSON and numerical maps;
- `clinical_claim: NONE`.

The HTML report surfaces the unified session block before legacy plate/pairwise reference sections.

## Contact-mask policy

Independent contact annotations remain valuable for strict tissue-contact experiments and are preserved as a research gate in the PR #41 local-evidence lane.

They are **not** an architectural blocker for the new TLC measurement lane. The session runtime can analyze high-confidence visible TLC response using explicit semantics:

`PROVISIONAL_VISIBLE_TLC_RESPONSE_NOT_CONFIRMED_TISSUE_CONTACT`

No missing/weak response is silently converted into a healthy label.

## Locked current-cohort challenge

`scripts/run_mumguard_signal_locked_eval.py` provides a frozen no-fit experiment for the existing nine private mouse originals.

Rules:

- protocol is written before outcome scoring;
- no disease head is fitted;
- no threshold is selected;
- image hashes are verified;
- features are extracted before labels are used for ranking;
- the sole negative is used only for an engineering positive-vs-negative ranking report;
- output is explicitly development-only, not independent validation.

This experiment evaluates the new static signal/physics representation. It cannot exercise true human bilateral reconstruction because the current nine mouse originals are single-capture development subjects rather than a full LEFT/RIGHT MumGuard human session.

## What can be added without redesign

When available, the same architecture accepts:

- exact device/TLC Hue-to-temperature calibration points;
- deterministic scanner coordinates instead of image-derived overlap registration;
- repeated/temporal captures for transient thermal features;
- genuine human bilateral MumGuard sessions;
- a small human outcome-linked decision/calibration head.

These are data/profile upgrades to the current system rather than reasons to replace the core architecture.
