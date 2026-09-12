# TLC Signal-First Implementation — 2026-09-12

## Status

This document is the canonical engineering interpretation of the client-supplied Stasiek 2014 TLC paper for MumGuard work.

The project is no longer blocked on a generic "need more data" statement. The remaining work is primarily implementation: turn photographed Contact-TLC response into a stable thermal-signal representation, then feed that representation into the AI decision layer.

## Confirmed from supplied evidence

- Current demonstrated client input is photographed Contact-TLC RGB/JPEG response (mouse cohort 0030-0038).
- Thermochromic liquid crystals act as optical temperature indicators: reflected colour/hue changes with temperature.
- RGB capture can be transformed into HSI/HSV for analysis.
- Hue-to-temperature interpretation is profile-specific and nonlinear.
- Illumination, observation angle, TLC formulation/range, camera/optics, and acquisition conditions affect the measurement.
- The paper supports both steady-state and transient thermal analysis.
- The medical section explicitly treats TLC thermography as usable for cancer/tumour screening/monitoring contexts and shows a breast-cancer thermogram example.
- The paper separates measurement conditions from interpretation: patient/environment conditions belong in the thermological report.

## Engineering assumptions used to proceed

These are working assumptions, not open questions unless contradicted by new client evidence.

1. The first-class input contract is `TLC_RGB_CAPTURE`.
2. The nine supplied client mouse images demonstrate the current capture modality well enough to build the software path now.
3. Absolute Celsius calibration is profile data, not an architectural blocker.
4. Until device-specific Hue->temperature calibration points are available, the pipeline will produce a **relative thermal/chromatic response map** rather than fabricate absolute temperatures.
5. When calibration points become available, the same pipeline upgrades to absolute temperature mapping through the profile layer; no redesign is required.
6. Static single captures use steady-state spatial analysis.
7. If multiple timepoints/frames are present, the same signal layer extends to transient analysis.
8. Current DINOv2 mouse classifier remains a baseline/development head. It is not the final human MumGuard decision head.
9. Unknown or weakly observable TLC response is not silently interpreted as biologically negative.
10. Do not stop execution merely because manufacturer/product code, exact colour-play interval, or absolute calibration constants are not yet populated. Those values are profile parameters.

## What from the paper maps directly into software

### 1. RGB -> HSI/HSV

Convert camera RGB/BGR capture to hue, saturation, and intensity/value channels. Hue is the primary chromatic coordinate for TLC response; saturation/value are used for observability/QC and segmentation.

### 2. TLC response mask

Use saturation/value and morphology to isolate high-confidence visible TLC response. Keep acquisition-valid region separate from actual observable TLC response.

### 3. Relative thermal map

For uncalibrated profiles, map active-response hue into a bounded relative thermal index while preserving explicit semantics:

- `0.0` = low end of configured TLC colour-play direction
- `1.0` = high end
- output is **not degrees Celsius**

This supports spatial comparison, local contrast, asymmetry, gradients, and model features immediately.

### 4. Absolute temperature map (profile upgrade)

When device-specific calibration points are supplied, map Hue -> temperature using empirical calibration points stored in the TLC profile. The paper demonstrates nonlinear calibration and a sixth-order polynomial example; implementation should prefer explicit empirical interpolation by default and optionally support a fitted polynomial only when enough calibration points exist.

### 5. Steady-state features

For a single capture, derive at minimum:

- response coverage
- hue/relative-thermal distribution statistics
- hot/cold tail fractions
- spatial centroid
- connected response regions
- local gradient/contrast descriptors
- morphology / vascular-like complexity descriptors
- left/right or region-to-region asymmetry where geometry supports it

### 6. Transient features

For ordered captures/timepoints, derive:

- per-pixel or regional delta response
- rate of change
- peak response
- time-to-peak
- recovery/decay slope
- temporal variance/stability
- spatial persistence of anomalous regions

Transient support is additive; static captures remain valid inputs.

### 7. QC and acquisition provenance

Record, where available:

- TLC profile id
- camera/device profile id
- capture mode (`steady_state` / `transient`)
- lighting/acquisition profile
- ambient/patient preparation metadata
- calibration mode (`relative` / `absolute`)
- calibration provenance/version

The Stasiek paper's room/patient preparation guidance informs acquisition metadata/QC, not the disease model itself.

## What does NOT transfer literally from the paper

The paper also contains engineering demonstrations involving wind tunnels, water baths, electric heaters, Reynolds/Nusselt numbers, heat-transfer coefficients, flow tracers, and PIV. Those sections establish TLC measurement physics and calibration practice but are not MumGuard breast-analysis targets.

Do not implement wind-tunnel-specific quantities (`Re`, `Nu`, plate heat-transfer coefficient) into the breast decision pipeline unless a future MumGuard protocol explicitly measures the required physical inputs.

## Canonical runtime direction

```text
TLC_RGB_CAPTURE
  -> acquisition/profile normalization
  -> observable TLC response segmentation
  -> RGB -> HSI/HSV
  -> relative thermal map OR calibrated temperature map
  -> spatial evidence features
  -> optional temporal evidence features
  -> hybrid representation (TLC signal + morphology + frozen visual representation)
  -> species/domain-specific decision head
  -> research result + evidence map + QC/abstain
```

## Delivery rule

The next phase is implementation, not another open-ended discovery pass.

Missing profile constants are handled as configuration/calibration upgrades. They do not block delivery of the relative-signal pipeline, evidence extraction, AI integration, runtime contracts, tests, and reporting.

The only thing that must never be fabricated is a claimed absolute temperature or calibrated human cancer probability when the required calibration/validation evidence does not exist. The system should continue to function using explicit relative-signal semantics until those upgrades are available.
