# MumGuard session / bilateral measurement specification — 2026-09-12

## Canonical client acquisition facts

The current MumGuard technique should be implemented as a **bilateral examination session**, not as an isolated JPEG classifier.

The client clarified that:

- TLC colour response is device/profile dependent; colour appearance from another TLC setup cannot be assumed to have the same numeric meaning.
- External/reference images may teach algorithms or morphology, but primary quantitative interpretation must stay bound to the MumGuard/device TLC profile.
- One breast side is captured in multiple partially overlapping images.
- Adjacent captures may repeat tissue from the previous frame, so repeated regions must not be counted as independent evidence.
- A session can contain many images for both left and right breasts.
- The technique reasons from three primary signals:
  1. focal/core hyperthermia;
  2. right-vs-left breast asymmetry;
  3. abnormal skin thermal behaviour / abnormal spatial temperature distribution.

## Human-first system contract

```text
MumGuard bilateral session
  -> LEFT ordered/related TLC captures
  -> RIGHT ordered/related TLC captures
  -> profile-aware TLC signal mapping
  -> per-frame observability / quality
  -> overlap registration and de-duplication
  -> one side-level thermal field for LEFT
  -> one side-level thermal field for RIGHT
  -> local multi-scale region-vs-surrounding analysis
  -> bilateral mirrored comparison
  -> three measurement channels
       A. core hyperthermia
       B. bilateral asymmetry
       C. abnormal skin thermal behaviour
  -> optional local visual/DINO evidence
  -> evidence fusion
  -> examination evidence map / research decision layer
```

This architecture is **human-first and label-free through the measurement/evidence stages**. Mouse subjects are development/evaluation material, not the architecture target.

## Overlap handling

A sequence of side captures must not be treated as independent samples when they share anatomy.

The implementation:

1. estimates pairwise translation between neighbouring captures using local image features, with phase correlation fallback;
2. accumulates frame positions into one side coordinate system;
3. composes signal maps on a shared canvas;
4. averages signal in overlapping pixels rather than double-counting it;
5. stores `coverage_count` so downstream logic can identify repeatedly observed tissue.

If a later hardware protocol exposes deterministic scanner coordinates, those coordinates should replace image-derived registration without changing downstream side/bilateral contracts.

## Bilateral comparison

Right and left side fields are normalized into a common side coordinate system. The right side is mirrored so homologous lateral anatomy faces the same normalized coordinates. Each side is robustly standardized before subtraction; therefore the asymmetry channel emphasizes **distribution and behaviour disagreement**, not a global exposure or colour offset.

Output includes:

- signed left-minus-right field;
- absolute asymmetry map;
- joint observable coverage;
- tail-based asymmetry score;
- explicit inconclusive state when bilateral overlap is insufficient.

## Three evidence channels

### 1. Core hyperthermia

Use local multi-scale thermal contrast to identify coherent positive local elevation relative to surrounding observable tissue. The temperature direction belongs to the TLC/device profile. The algorithm itself does not learn tumour appearance.

### 2. Bilateral asymmetry

Use the opposite breast as a patient-specific internal control. Compare homologous normalized locations after side reconstruction and mirroring.

### 3. Abnormal skin thermal behaviour

Use persistent multi-scale local anomaly structure, spatial heterogeneity and abnormal distribution instead of reducing the scan to one hottest pixel.

## AI relationship

AI remains a separate evidence channel, not a replacement for measurement physics.

```text
TLC/thermal measurement evidence
+ local DINO/visual representation evidence
+ acquisition/observability support
= session evidence fusion
```

The final disease-specific head is replaceable. The measurement, overlap, bilateral and anomaly engines are intended to survive future human outcome calibration rather than being rebuilt around a new image classifier.

## Integrated implementation

Fusion branch: `feat/mumguard-session-fusion-20260912`

This branch starts from Codex PR #41 (`research/mumguard-local-contrast-v1`) and adds:

- `app/services/tlc_signal_processing.py`
- `app/services/thermal_anomaly_engine.py`
- `app/services/bilateral_session_engine.py`
- `app/services/mumguard_session_fusion.py`

The session fusion layer deliberately distinguishes visible TLC response support from independently reviewed tissue contact. Missing contact annotation no longer prevents the relative TLC measurement engine from running; it still remains explicit provenance and must not be reinterpreted as confirmed tissue or healthy tissue.
