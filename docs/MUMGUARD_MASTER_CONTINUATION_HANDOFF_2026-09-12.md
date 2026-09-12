# MumGuard / Contact-TLC AI — MASTER CONTINUATION HANDOFF
**Date:** 2026-09-12  
**Repo:** `omarkhair70-droid/contact-thermography-ai`  
**Canonical integration branch:** `integration`  
**Current merged integration SHA before this handoff commit:** `ce5957cf87ec7a1e88060a92ccb66c20784d8018`  
**Canonical architecture issue:** `#40 — Canonical R&D framing: TLC signal-first AI pipeline + assumption policy`

---

## 0) READ THIS FIRST — WHAT THIS PROJECT ACTUALLY IS

This project is **not** a simple image classifier and it is **not** a mouse-tumor prototype.

The real product target is a **human-first MumGuard breast examination intelligence system** for Contact / Liquid Crystal Thermography (Contact-TLC / LCT).

The final intended product flow is:

```text
Human MumGuard examination
  -> many Contact-TLC captures for LEFT breast
  -> many Contact-TLC captures for RIGHT breast
  -> same-device/profile-aware color normalization
  -> visible TLC response / observability analysis
  -> TLC color -> relative thermal signal
  -> overlap registration / duplicate-tissue suppression
  -> one side-level field for LEFT + one for RIGHT
  -> local thermal anomaly analysis
  -> focal/core hyperthermia evidence
  -> right-vs-left bilateral asymmetry evidence
  -> abnormal skin thermal-distribution evidence
  -> local DINOv2 / AI visual evidence
  -> explainable fusion
  -> session-level evidence
  -> later: calibrated human decision/risk layer
  -> report / API / UI
```

**The mouse images are only a development challenge used to test algorithms.** They do not define the architecture and should never be described as the final product.

---

# 1) THE ORIGINAL MISTAKE — AND HOW WE CORRECTED IT

## Wrong framing we started with

At first, the problem drifted toward:

```text
collect labeled tumor/no-tumor images
-> train image classifier
-> output tumor-like / no-tumor-like
```

That framing was too conventional and did not match what the client was actually asking for.

The client had explicitly implied that if this were just a normal labeled-dataset problem, they would already have built the model.

The project also temporarily leaned too heavily on:
- the 9 mouse images;
- whole-image DINO embeddings;
- supervised tumor/no-tumor classification;
- asking for more data as the default answer.

That was not the real problem.

## Correct framing

After understanding the TLC paper, the device behavior, Maryam's explanations, and the actual acquisition method, the problem became:

```text
How do we extract stable, meaningful thermal/biophysical evidence
from MumGuard's own TLC response,
then use AI/ML to interpret that evidence?
```

So the architecture became **signal-first + AI**, not “AI instead of signal processing”.

The correct mental model is:

```text
MumGuard measurement layer
-> thermal-response representation
-> spatial/bilateral/local features
-> AI/local representation
-> decision layer
```

AI is still central. The TLC physics/calibration/signal-processing layer is the **measurement layer feeding the AI**, not a replacement for AI.

---

# 2) CRITICAL CLIENT INFORMATION FROM MARYAM

This information materially changed the architecture and must be preserved.

## 2.1 MumGuard fundamentally outputs TLC colors

The device response is based on **liquid crystal color**.

The actual color response depends on the specific TLC formulation/setup/device conditions.

Therefore:
- color meaning is **device/setup/profile specific**;
- another TLC product's red/green/blue mapping cannot automatically be applied to MumGuard;
- public/external images can help us understand patterns and algorithms;
- but MumGuard's own same-device captures are the primary domain for training/evaluation.

Do **not** silently mix external TLC images as if their Hue-to-temperature mapping is identical.

## 2.2 Human acquisition uses multiple partial images per breast

A single breast may be imaged in several adjacent partial captures. Adjacent images may overlap.

Therefore the same tissue region can appear in more than one image.

If we treat every JPEG independently, duplicated tissue can be counted several times and bias the model.

This is why the system now has **overlap registration + de-duplication + side-level reconstruction**.

## 2.3 Both breasts are available

MumGuard can acquire many images for both the right and left breasts.

That is extremely important because the patient can act as her own internal reference.

Useful internal comparisons include:

```text
region vs surrounding tissue
RIGHT vs LEFT
possibly repeated acquisition vs repeated acquisition
```

This reduces dependence on a huge labeled disease dataset for the measurement/anomaly layer.

## 2.4 Maryam described three main things they inspect

These are now first-class algorithmic channels.

### 1. Core / focal hyperthermia
A localized area showing elevated thermal response relative to surrounding tissue.

Important: not “red pixel = cancer”; use local contrast, spatial coherence, persistence, region-vs-ring comparison, multiscale evidence.

### 2. Right-vs-left asymmetry
The thermal behavior/distribution of one breast may differ from the contralateral breast.

The system should compare homologous regions robustly, not rely on perfect pixel-by-pixel symmetry.

### 3. Abnormal skin thermal behavior
The spatial distribution itself may look abnormal: unusual gradients, texture, topology, hotspot organization, distribution changes, DINO/local feature anomalies.

These are **measurement/evidence channels**, not yet a clinically calibrated cancer diagnosis.

---

# 3) OTHER IMPORTANT CLIENT CLARIFICATIONS

## Mouse ground truth supplied by Maryam

- `0030` = TUMOR_BEARING
- `0031` = TUMOR_BEARING
- `0032` = TUMOR_BEARING
- `0033` = TUMOR_BEARING
- `0034` = TUMOR_BEARING
- `0035` = TUMOR_BEARING
- `0036` = TUMOR_BEARING
- `0037` = TUMOR_BEARING
- `0038` = HEALTHY / NO_TUMOR

Current development cohort: `8 tumor-bearing / 1 healthy-no-tumor`.

## Tumor size rule

Maryam clarified that TLC visibility is **not** a direct tumor-size measurement.

Signal visibility can depend on depth, superficiality, contact/measurement conditions, and thermal response. A small superficial tumor may show strongly.

Therefore:

```text
visible response strength/area != tumor size
```

Do not infer tumor size from response area.

---

# 4) WHAT THE IMPORTANT TLC PAPER TAUGHT US

Main paper: **Stasiek, Jewartowski, Kowalewski (2014), The Use of Liquid Crystal Thermography in Selected Technical and Medical Applications—Recent Development**.

Key engineering lessons:

- TLC behaves as an optical temperature indicator; color/wavelength changes with temperature through a color-play interval.
- RGB -> HSI/HSV is useful and Hue is particularly useful for TLC analysis.
- Hue -> temperature is not universal and can be strongly nonlinear.
- Absolute temperature requires calibration of the combined TLC + lighting + camera + viewing setup.
- Current software therefore supports relative thermal response now, with an empirical Hue-to-temperature upgrade path later.
- Do **not invent Celsius values** before MumGuard-specific calibration exists.
- Viewing angle, lighting, reflections and setup conditions matter and belong in QC/profile handling.
- Static and transient methods both exist.
- Maryam's multiple images may be spatial tiles, not temporal frames. Do not calculate temporal slope from spatially adjacent captures.
- Acquisition semantics should distinguish `SPATIAL_TILE`, `TEMPORAL_FRAME`, `UNKNOWN`.
- Wind-tunnel/Nusselt/Reynolds/PIV content from the paper is not a direct breast-disease feature set.

---

# 5) EXTERNAL AI / THERMOGRAPHY RESEARCH — HOW TO USE IT

Generic AI thermography literature is useful for preprocessing, segmentation, normalization, spatial alignment, CNN/transformer representations, sequential modeling when a true time series exists, and feature fusion.

But Contact-TLC is not identical to radiometric infrared thermography. Do not import IR assumptions blindly.

External/contact-TLC images may help algorithm research, morphology, robustness or representation learning if justified. They are **not automatically valid same-domain color training data**.

---

# 6) CURRENT CANONICAL ARCHITECTURE

```text
MumGuard examination/session
    |
    +-- LEFT captures
    |      -> device/TLC profile normalization
    |      -> visible-response / observability
    |      -> TLC color -> relative thermal signal
    |      -> overlap detection + registration
    |      -> duplicate evidence suppression
    |      -> LEFT side thermal field
    |
    +-- RIGHT captures
           -> device/TLC profile normalization
           -> visible-response / observability
           -> TLC color -> relative thermal signal
           -> overlap detection + registration
           -> duplicate evidence suppression
           -> RIGHT side thermal field

LEFT + RIGHT
   |
   +-> local multiscale thermal anomaly
   +-> focal/core hyperthermia
   +-> mirrored bilateral comparison
   +-> abnormal spatial/skin thermal behavior
   +-> local DINOv2 patch evidence
   -> session fusion
   -> evidence maps + scores + uncertainty/status
   -> future calibrated human risk/decision layer
```

---

# 7) CURRENT IMPLEMENTED CODE

Everything below is in merged integration history through PR #42.

## `app/services/tlc_signal_processing.py`
Implements BGR/RGB -> HSV physical channels, Hue in degrees, saturation/value normalization, relative thermal-response map, active/observable support mask, empirical Hue->temperature calibration adapter, signal summaries, spatial summaries, and transient sequence summaries.

Relative map is **not Celsius**. Calibrated absolute mapping requires real profile calibration points.

## `app/services/thermal_anomaly_engine.py`
Implements multiscale local-vs-surrounding comparison, ring references, robust signed contrast, robust local anomaly scoring, cross-scale persistence, evidence maps, observable-mask preservation, and label-free anomaly features.

## `app/services/bilateral_session_engine.py`
Implements side-level frame handling, sequential overlap registration, offsets, coverage tracking, overlap de-duplication, LEFT field, RIGHT field, normalized/mirrored bilateral comparison, and session evidence channels.

## `app/services/mumguard_session_fusion.py`
Implements the end-to-end MumGuard session runtime connecting TLC signal extraction, side reconstruction, thermal anomaly, bilateral evidence, optional DINO evidence, and evidence persistence.

Visible TLC response can be used as provisional measurement support without lying that it is an independently reviewed tissue-contact mask.

Conceptual semantic label:

```text
PROVISIONAL_VISIBLE_TLC_RESPONSE_NOT_CONFIRMED_TISSUE_CONTACT
```

---

# 8) CODEX PR #41 — WHAT IT BUILT

Branch: `research/mumguard-local-contrast-v1`  
Commit: `874406e3cf9975d3c2b819469ff03f87ef003d77`  
PR: `#41 — Add inactive human-first MumGuard research candidate`

This work was **not discarded**. PR #42 was built on top of it and carried its foundation into integration.

Important Codex files:
- `app/services/contact_field.py`
- `app/services/tlc_observability.py`
- `app/services/local_contrast_features.py`
- `app/services/dinov2_patch_encoder.py`
- `app/services/mumguard_research_runtime.py`
- `app/services/mumguard_intake.py`

It also added human intake templates, split specification, research protocols, ablations, perturbation/permutation tests, acquisition receipts, model cards, provenance, and subject-level evidence artifacts.

---

# 9) IMPORTANT CORRECTION TO CODEX'S CONTACT POLICY

Codex initially kept the local research path very conservative:

```text
no independent reviewed contact mask
-> measurement not eligible
-> abstain
```

That is appropriate for strict “tissue-contact confirmed” experiments, but too strict as a blocker for the entire TLC signal pipeline.

Correction:

```text
no independent contact annotation != stop all TLC signal analysis
```

But also:

```text
visible TLC response != confirmed tissue contact
```

The new session measurement layer can use visible/high-confidence TLC response while explicitly marking contact certainty as unknown/provisional.

---

# 10) DINO / AI ROLE

DINOv2 stays in the architecture.

Current role:

```text
local patch-level visual representation
+ anomaly/novelty evidence
+ complementary channel to thermal/physics evidence
```

The final system should be hybrid: not physics-only and not DINO-classifier-only.

---

# 11) ORIGINAL WHOLE-IMAGE DINO BASELINE

Existing artifact: `artifacts/lct_native_exp8v1_dino.json`

Properties:
- DINOv2 ViT-S/14;
- 384-dimensional embeddings;
- logistic regression;
- species: mouse;
- profile: `client-device-tlc-pending`;
- 9 subjects;
- 8 positive / 1 negative;
- limited validation status.

Historical resubstitution scores:
- 0030: ~0.570133
- 0031: ~0.568223
- 0032: ~0.598274
- 0033: ~0.619351
- 0034: ~0.567608
- 0035: ~0.555280
- 0036: ~0.563967
- 0037: ~0.596718
- 0038: ~0.400324

These are **not cancer probabilities**. This remains a baseline/reference channel, not the final architecture.

---

# 12) CODEX RESEARCH FINDINGS

Codex completed a large research lane and reported roughly:
- full suite at the time: `134 passed, 1 skipped`;
- pinned DINOv2 execution on all 9 originals;
- saved whole-image baseline reproduced very closely;
- multiple local/morphology/pooling candidates tested;
- local candidates did not clearly beat the saved global baseline under frozen promotion rules;
- no mouse disease head was promoted;
- reusable human-first local sensing engine retained;
- runtime preserves explicit abstention.

Important interpretation: this does **not** mean “local signal/physics failed”. Codex did not yet have Maryam's full multi-image breast reconstruction, overlap de-duplication, human LEFT/RIGHT session structure, or complete three-channel MumGuard technique.

---

# 13) PR #42 — FINAL UNIFIED MERGE

PR: `#42 — Integrate MumGuard TLC physics and bilateral session fusion`

Merged into: `integration`

Merge SHA: `ce5957cf87ec7a1e88060a92ccb66c20784d8018`

PR #42 contains the complete PR #41 foundation plus the signal/session work.

Final PR-head QA before merge:

```text
153 passed
1 skipped
2 warnings
```

The remaining two warnings were dependency/deprecation warnings, not thermal-engine runtime warnings.

At the time this master handoff was written:
- code is merged into `integration`;
- PR #42 is merged;
- PR #41 is superseded/contained by #42;
- post-merge Coolify/live smoke must still be explicitly confirmed if a later chat has not done it.

Do not claim Coolify is green unless actually checked.

---

# 14) CURRENT LIVE API BEHAVIOR IN CODE

Application version was moved to `0.6.0`.

Main analysis endpoint:

```text
POST /api/exams/analyze
```

For a client-profile Contact-LCT examination containing both `LEFT` and `RIGHT`, the backend can build:

```text
mumguard_session_evidence
```

Current result can contain:
- three session scores;
- side reconstruction provenance;
- offsets/coverage;
- bilateral support statistics;
- whether AI evidence was included;
- evidence JSON URL;
- map bundle URL;
- regular plate analysis;
- legacy bilateral pair analysis;
- session-level research status.

Current model status can become:

```text
MUMGUARD_SESSION_EVIDENCE_RESEARCH
```

Current code deliberately still has:

```text
clinical_risk = None
clinical_claim = NONE
```

This is extremely important.

---

# 15) WHAT THE SYSTEM CAN DO NOW

Implemented now:
- accept TLC images and preserve device/TLC profile provenance;
- avoid treating incompatible mixed color domains as interchangeable;
- convert RGB/HSV/Hue into relative thermal-response representation;
- support future empirical absolute calibration;
- perform local region-vs-surrounding multiscale analysis;
- generate evidence maps, signed contrast and persistence;
- handle multiple captures per side;
- register overlap and build side-level fields;
- track coverage and suppress duplicate weighting;
- reconstruct LEFT and RIGHT fields;
- calculate bilateral asymmetry;
- calculate core hyperthermia-like evidence;
- calculate abnormal thermal-behavior evidence;
- use whole-image and local DINOv2 evidence;
- fuse research evidence;
- abstain / mark inconclusive when needed;
- preserve provenance;
- avoid inventing absolute temperatures;
- avoid treating scores as cancer probabilities;
- expose API, persistence, maps and reports.

---

# 16) WHAT IS NOT COMPLETE YET

## 16.1 No clinically calibrated risk percentage yet

Omar ultimately wants human MumGuard output such as risk / indication / confidence.

The current system does **not** yet have a validated human probability/risk head. Current evidence scores are not interchangeable with `72% cancer risk`.

The code correctly leaves `clinical_risk = None`.

## 16.2 Human decision/calibration layer still needs completion

Scaffolding already exists, including:
- `scripts/train_mumguard_human_head.py`
- `scripts/calibrate_mumguard_decision.py`
- human intake/split files.

Desired future layer:

```text
session evidence vector
  [core hyperthermia,
   bilateral asymmetry,
   abnormal pattern,
   DINO/local evidence,
   QC/coverage,
   profile metadata]
-> calibrated human decision model
-> low/intermediate/high indication or calibrated risk
-> abstain if measurement support is weak
```

## 16.3 Actual MumGuard Hue->temperature calibration is pending

Relative thermal field is implemented and useful. Absolute temperature requires same-device calibration points from the real MumGuard TLC setup.

## 16.4 Real human multi-image sessions must exercise the architecture

Overlap, side reconstruction and bilateral comparison exist in code, but the 9-mouse cohort cannot genuinely validate the human bilateral path.

## 16.5 Dedicated human operator UI needs finishing

Backend/session capability exists. Complete operator flow should provide:

```text
New examination
-> session ID
-> device/TLC profile
-> upload LEFT captures
-> upload RIGHT captures
-> optional order/position metadata
-> Analyze
-> coverage/reconstruction
-> three channels
-> evidence maps
-> AI evidence
-> measurement quality
-> later calibrated indication/risk
-> export report
```

## 16.6 Clinical validation is separate from engineering completion

Clinical validity needs human outcomes, locked subject splits, independent positives/negatives, and preferably prospective/blind evaluation. Do not confuse software capability, R&D evidence and clinical validation.

---

# 17) LOCKED CURRENT-COHORT SIGNAL EXPERIMENT

Script: `scripts/run_mumguard_signal_locked_eval.py`

Artifact folder: `artifacts/mumguard-signal-locked-v1/`

Rules:
- protocol frozen before label scoring;
- no disease head fitted;
- no threshold chosen;
- original private image hashes verified;
- feature extraction before outcome ranking;
- current cohort already development-exposed, so not strict independent validation.

Results:

### Core hyperthermia direction
**7 / 8** tumor-bearing captures ranked above the sole no-tumor capture.

### Frozen static composite
**6 / 8** tumor-bearing captures ranked above the sole no-tumor capture.

### Generic unilateral abnormal-pattern formulation
**0 / 8** above the sole no-tumor capture.

We did **not** hide or reweight this failure after seeing labels.

Interpretation:
- focal/core thermal contrast looks promising in this development set;
- current generic unilateral abnormality formula is weak for this mouse challenge;
- bilateral human internal-control data may be more relevant to the third channel.

These are **development ranking results**, not sensitivity/specificity or cancer probabilities.

---

# 18) WHY 7/8 IS INTERESTING BUT NOT “MODEL ACCURACY”

There is only one negative subject and the cohort was already development-exposed.

Do not write `87.5% diagnostic accuracy`.

Correct wording:

```text
Under the frozen development ranking protocol,
7 of 8 tumor-bearing captures produced a higher directed
core-hyperthermia signal than the sole no-tumor capture.
```

---

# 19) BLIND TEST POLICY GOING FORWARD

For a true next test:
1. get a **new** MumGuard capture/session never used in development;
2. keep tumor/no-tumor outcome hidden;
3. freeze code/config/version;
4. hash input files;
5. run inference without changing weights/thresholds after seeing images;
6. save outputs;
7. lock results;
8. only then reveal ground truth;
9. compare;
10. for multiple cases, reveal outcomes after all predictions are frozen if possible.

Best input is a real human session with multiple LEFT and RIGHT captures, original order/metadata, same device/TLC setup, and outcome withheld until inference is frozen.

---

# 20) SAME-DEVICE DATA POLICY

Primary domain:

```text
MumGuard's own device/setup captures
```

External data can inform algorithms, morphology, robustness and perhaps representation learning, but do not assume external hue/color equals MumGuard hue/color.

Device/TLC profile is first-class.

---

# 21) SESSION DATA MODEL WE WANT

Conceptually every capture should carry:

```text
examination_id
subject_id
side: LEFT | RIGHT
capture_id
capture_role: SPATIAL_TILE | TEMPORAL_FRAME | UNKNOWN
sequence_index
tlc_profile_id
device_profile_id
timestamp (if available)
optional position/orientation/scanner coordinate
raw image
```

Derived fields include visible-response/observability mask, relative/calibrated thermal map, overlap edges/transforms, side coverage, side-level field, local anomaly map, bilateral map, channel scores, DINO features, and uncertainty/abstain status.

---

# 22) OVERLAP / STITCHING POLICY

Preferred path:

```text
detect overlap
-> estimate transform
-> place captures in common side coordinates
-> blend/support-average repeated tissue
-> track coverage
-> do not multiply evidence because region appeared multiple times
```

If full panorama/stitching is unstable on real human data, do **not** force a visually perfect panorama. Fallback can be a tile graph/set with pairwise overlap edges, coordinate relations and deduplicated aggregation.

The objective is correct evidence aggregation, not panorama aesthetics.

---

# 23) BILATERAL MATCHING POLICY

Do not require perfect pixel mirror. Human pose may differ.

Use robust matching: normalized side coordinates, mirroring, coverage-aware support, regional descriptors, multiscale comparison, and eventual landmarks/scanner coordinates if available.

Contralateral breast is an internal reference, not guaranteed perfect symmetry.

---

# 24) THREE-CHANNEL SCORING PHILOSOPHY

## Core hyperthermia
Use local center vs surrounding ring, robust statistics/MAD, multiscale persistence, connected spatial evidence, directionality and boundary/gradient behavior. Do not use “red = tumor”.

## Bilateral asymmetry
Compare relative thermal field, hotspot burden, regional distributions, local features and mirrored/normalized zones. Output both a map and global score/support.

## Abnormal skin behavior
Potential features include quantiles/distribution, entropy, gradients, thermal roughness, component organization, topology, persistence and DINO patch anomalies. This channel may need better human/bilateral formulation based on real sessions.

---

# 25) FUSION POLICY

Do not collapse everything too early into one opaque number.

Desired result:

```text
Measurement quality
Core hyperthermia evidence
Bilateral asymmetry evidence
Abnormal pattern evidence
AI/local visual evidence
Overall fused evidence
Decision/risk layer status
Reason for abstention if applicable
```

The eventual clinical risk model should consume these channels, not erase them.

---

# 26) CURRENT PRODUCT STATUS — MODULE BY MODULE

| Area | Status |
|---|---|
| TLC color/signal extraction | Implemented |
| Relative thermal representation | Implemented |
| Empirical absolute calibration adapter | Implemented, real MumGuard calibration points pending |
| Local multiscale anomaly | Implemented |
| DINOv2 local/patch evidence | Implemented |
| Observability/provenance | Implemented |
| Multi-image overlap registration | Implemented v1 |
| Duplicate tissue suppression | Implemented v1 |
| LEFT side reconstruction | Implemented v1 |
| RIGHT side reconstruction | Implemented v1 |
| Bilateral asymmetry | Implemented v1 |
| Core hyperthermia channel | Implemented |
| Abnormal-pattern channel | Implemented but current unilateral formulation needs human evaluation/improvement |
| Session fusion | Implemented |
| API session evidence | Implemented |
| Evidence/map persistence | Implemented |
| HTML report session block | Implemented |
| Locked mouse development experiment | Completed |
| Full current test suite on PR head | Green: 153 passed, 1 skipped |
| Human intake scaffolding | Implemented |
| Human risk/decision model | Not yet clinically trained/calibrated |
| Human risk percentage | Not available yet |
| Real human bilateral validation | Pending |
| Actual MumGuard absolute thermal calibration | Pending |
| Dedicated polished operator UI | Needs completion |
| Blind independent human evaluation | Pending |
| Post-merge Coolify/live smoke | Must be explicitly verified |

---

# 27) WHAT “PROJECT COMPLETE” SHOULD MEAN FOR OMAR

Omar does **not** want the endpoint to be “we built a research library, now ask the client for data.”

He wants a complete integrated software product path:

```text
1. robust human session intake
2. multiple LEFT/RIGHT uploads
3. overlap-aware reconstruction / aggregation
4. profile-aware TLC signal analysis
5. three-channel evidence
6. local AI evidence
7. explainable maps
8. measurement QC / inconclusive path
9. risk/indication decision layer
10. report
11. production deployment
12. locked blind evaluation workflow
13. versioned model/profile configuration
```

Where human outcomes/calibration are genuinely required, the software scaffold should already exist so new data plugs into the current architecture instead of triggering a rewrite.

---

# 28) IMMEDIATE NEXT ENGINEERING TASKS

Do **not** restart the architecture. Continue from merged `integration`.

## A. Verify the merged production path
Check GitHub state after merge, Coolify deployment, `/health`, storage, DB, report, and `mumguard_session_fusion` health flag. Run live smoke. Do not delete volumes.

## B. Build / finish the dedicated human examination UI
Support new exam/session, LEFT image bucket, RIGHT image bucket, order/position if known, profile/device metadata, analysis progress, side reconstruction, evidence maps, channel scores, uncertainty and report.

## C. Make the risk layer explicit as a separate component
Do not rename `overall_measurement_evidence_score` to `cancer risk`.

Create a true downstream interface such as:

```text
HumanDecisionInput
HumanDecisionResult
```

Possible pre-calibration output:

```text
decision_status:
  NOT_CALIBRATED
  INCONCLUSIVE
  INDICATION_LOW
  INDICATION_INTERMEDIATE
  INDICATION_HIGH

risk_score:
  null until calibrated
```

Then train/calibrate when outcome-linked human cases arrive.

## D. Improve abnormal-skin behavior channel
Current mouse locked challenge showed generic unilateral formulation was poor. Do not tune on the same nine labels just to make a prettier result. Better path: bilateral human comparison, distribution/gradient/topology descriptors, DINO local anomaly and real session behavior, then freeze before new blind evaluation.

## E. Real human session challenge
Get at least one true multi-image LEFT/RIGHT session from MumGuard to test overlap, reconstruction, contralateral comparison and three-channel session fusion.

---

# 29) WHAT TO ASK / NOT ASK MARYAM

Do not send a long beginner questionnaire.

Only ask things that materially change implementation and cannot be abstracted.

Useful future inputs:
- real LEFT/RIGHT session captures;
- acquisition order / scanner position if available;
- exact device/TLC profile identity;
- calibration measurements they already have;
- human verified outcome when doing validation;
- blind cases with outcome withheld.

Do not ask things already known just because a new chat forgot them.

---

# 30) WHAT TO SAY TO MARYAM ABOUT WHAT WE BUILT

Plain-language version:

```text
We changed the problem from treating each thermographic image
as an independent classification image into a MumGuard-specific
measurement pipeline.

The system now converts the TLC color response into a relative
thermal field, analyzes local hyperthermia, can combine overlapping
captures of each breast, compares right and left breasts, and preserves
an AI visual-evidence channel.

The output is currently an explainable research evidence result,
not yet a clinically calibrated cancer probability.

The next high-value validation is a real MumGuard human bilateral
session and then outcome-linked calibration of the final decision layer.
```

---

# 31) IMPORTANT OPERATING RULES FOR THE NEXT CHAT

1. **Human-first always.** Never call this a mouse model/product.
2. **Mouse = development challenge only.**
3. **Same-device color domain is primary.**
4. **Do not invent absolute temperature.**
5. **Do not call evidence score a cancer probability.**
6. **Do not make “more data” the default endpoint.** Build everything that can be built without it.
7. **Use working hypotheses and test them.** Do not ask unnecessary clarifying questions.
8. **Preserve uncertainty explicitly.** Unknown contact remains unknown.
9. **No response != healthy.**
10. **BENIGN != necessarily no lesion/no tumor** unless endpoint ontology explicitly defines it.
11. **Spatial tiles != temporal frames.**
12. **Do not double-count overlapping tissue.**
13. **Do not silently mix incompatible TLC color profiles.**
14. **Keep physics + AI.** Do not replace one with the other.
15. **Do not tune a frozen experiment after looking at labels.**
16. **Do not claim independent validation on the 9 development-exposed mice.**
17. **Direct action over endless planning.** If repo access exists and the next task is obvious, implement/review/test it.

---

# 32) REPO / PR HISTORY TO REMEMBER

- PR #37: experimental native DINO mouse baseline merged earlier.
- PR #38: CPU PyTorch deployment work.
- PR #39: DINO writable-cache Coolify fix.
- Issue #40: canonical signal-first architecture and assumption policy.
- PR #41: Codex human-first local evidence research foundation.
- PR #42: unified session fusion built on #41 and merged into `integration`.

Merged SHA after #42: `ce5957cf87ec7a1e88060a92ccb66c20784d8018`.

---

# 33) PRIVATE DATA POLICY

Private mouse originals, raw human scans, private token maps, overlays or sensitive research files should not be accidentally committed.

The repo includes ignore rules for private/work/result directories.

Keep raw client data outside git unless explicitly intended and authorized.

---

# 34) DEPLOYMENT CONTEXT

Earlier deployment architecture includes Oracle/Coolify, Docker, PostgreSQL, durable generated/uploaded image storage, health endpoint, restart policy and reverse-proxy/HTTPS readiness.

Important operational rule: **Do not delete database/storage volumes.**

For final live check, verify current `integration` SHA deployed, health endpoint, DINO runtime, storage, DB, known smoke cases and session evidence path if a bilateral test session is available.

---

# 35) WHAT THE CURRENT SYSTEM IS, IN ONE SENTENCE

**A human-first MumGuard Contact-TLC intelligence platform that turns device-specific TLC color response into relative thermal fields, reconstructs multi-image breast sessions, measures focal hyperthermia and bilateral/spatial abnormalities, fuses local AI evidence, and produces explainable session-level research evidence ready for a later calibrated human risk layer.**

---

# 36) WHAT IT IS NOT

It is not:
- a finished clinically validated cancer diagnostic;
- a mouse-only classifier;
- an IR thermography clone;
- a red/blue color heuristic;
- a tumor-size estimator;
- a system that can honestly output a cancer probability yet;
- a reason to throw away AI and use only thermal rules;
- a reason to throw away physics and use only DINO.

---

# 37) EXACT PROMPT FOR A NEW CHAT

> اقرأ `docs/MUMGUARD_MASTER_CONTINUATION_HANDOFF_2026-09-12.md` بالكامل من ريبو `omarkhair70-droid/contact-thermography-ai`، وبعدها راجع آخر حالة `integration` وCoolify قبل ما تجاوب. اعتبر الملف Source of Truth للمشروع والسياق والقرارات والغلطات اللي اتصلحت. ما ترجعش المشروع لفكرة image classifier أو mouse prototype، وما تسألنيش أسئلة اتجاوبت في الملف. الهدف النهائي مشروع MumGuard بشري كامل: multi-image LEFT/RIGHT intake، TLC signal، overlap de-dup، bilateral comparison، hyperthermia/asymmetry/abnormal behavior، local AI/DINO، fusion، explainable maps، ثم human decision/risk layer حقيقي منفصل عن measurement evidence. كمل من قسم “Immediate Next Engineering Tasks” ونفّذ على الريبو، وما تعتبرش “هات data أكتر” نهاية التاسك.

---

# 38) FINAL STATE FOR CONTINUATION

The major architecture question is **closed for now**.

Do not reopen:

```text
Should this be a simple image classifier?
```
Answer: no.

Do not reopen:

```text
Should the product be a mouse model first?
```
Answer: no.

Do not reopen:

```text
Does missing independent contact annotation mean the entire project stops?
```
Answer: no.

Do not reopen:

```text
Can external TLC colors be assumed equivalent to MumGuard?
```
Answer: no.

The next work is **product completion + human-session execution + real decision-layer calibration**, built on the merged architecture.

---

## END OF MASTER HANDOFF
