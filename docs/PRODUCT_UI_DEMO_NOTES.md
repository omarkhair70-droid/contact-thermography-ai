# Product UI Demo Notes

Branch: `feat/product-ui`

## Client-demo flow

The browser UI now exposes four clear workspaces:

1. **Dashboard** — system snapshot, research boundary and recent examinations.
2. **New Exam** — exam ID, TLC/device profile, drag/drop image upload, friendly LEFT/RIGHT/position assignment, analysis progress and visual review.
3. **History** — saved examinations with in-app review and report access.
4. **Reference** — Dataset Zero reference plates, visually separated from client-device analysis.

## Examination review

Each analyzed plate is presented with:

- original normalized plate image;
- thermochromic response mask;
- QC status/flags;
- side and matched position;
- morphology descriptor;
- response-area feature;
- best available reference-AI score, with graceful fallback while the live-DINO lane is integrated;
- nearest reference plates when returned by the backend.

Matched bilateral results show the combined alignment panel plus difference/aligned imagery and the existing research asymmetry metrics.

## TLC/device provenance

The new-exam form defaults to `client-device-tlc-pending` and optionally captures a device profile ID. The API change is additive: existing callers that do not submit these fields continue to work, while new examinations persist the profile identifiers in exam, source, plate and bilateral result records.

No UI text treats different TLC formulations as sharing one calibrated colour scale.

## Clinical wording

The UI and report consistently describe current outputs as research/reference image analysis. They never label a current score as cancer probability, diagnosis, sensitivity or specificity.

## Integration note

The Live DINOv2 lane may extend the plate result contract with learned live-inference fields. The UI intentionally reads those fields opportunistically and falls back to the current reference anomaly percentile, minimizing merge coupling.
