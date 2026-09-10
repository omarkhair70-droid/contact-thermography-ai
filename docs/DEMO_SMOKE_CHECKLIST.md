# Demo smoke checklist — QA/Hardening lane

Use this checklist on the exact candidate image/URL that will be shown to the client. The current product is a research image-analysis system for contact liquid-crystal thermography; it is not a diagnostic system.

## Automated gate

1. Install runtime and test dependencies: `pip install -r requirements.txt -r requirements-dev.txt`.
2. Run `pytest -q` and require all tests to pass.
3. Run `scripts/docker_smoke.sh` from the repository root on a machine with Docker, curl and Python 3.
4. Do not promote the candidate if the smoke script fails health, DINO reference, upload, persistence/history, report, JSON-finiteness, TLC provenance, or clinical-claim assertions.

## Manual demo smoke

- Open `/health`; confirm HTTP 200, `status=ok`, and `clinical_claim=NONE`.
- Open the dashboard and start a new examination.
- Confirm the selected TLC formulation/profile is visible or otherwise supplied as `tlc_profile_id`; when no profile is supplied, the API must explicitly use `client-device-tlc-pending` rather than silently assuming the publication colour domain.
- Upload one valid contact-thermography image. Confirm at least one plate result, QC status, morphology/reference analysis, saved exam, and report link.
- Upload a known multi-plate image. Confirm more than one extracted plate and stable plate IDs/order.
- Upload matching LEFT and RIGHT images with the same position and device profile. Confirm one bilateral pair is created and provenance is present on the pair.
- Repeat with conflicting device profile IDs. Confirm the images are not paired and a pairing warning is returned.
- Attempt one exam containing both `reference-publication-unknown` and `client-device-tlc-pending`. Confirm HTTP 400 and an explicit mixed-TLC-domain error; do not accept silent cross-domain colour comparison.
- Upload an empty file, a text file renamed as an image, and a corrupt image. Confirm controlled 4xx responses rather than a server error.
- Open `/api/reference/dinov2`; confirm HTTP 200 and strict JSON output (no NaN/Infinity tokens).
- Open `/api/exams`, the saved `/api/exams/{exam_id}`, and `/reports/{exam_id}`. Confirm TLC/device provenance survives persistence and is displayed in the report.
- Inspect all JSON and report output for the clinical boundary: `clinical_claim` must remain `NONE`, `clinical_risk` must remain null where present, and no numeric cancer probability, diagnosis, diagnostic result, or calibrated-temperature claim may appear.

## Known limitations / demo language

- Dataset Zero contains 26 publication/reference plates and is not a clinically representative labelled patient dataset. Current anomaly/asymmetry scores are reference-only.
- The publication TLC formulation is unknown; `reference-publication-unknown` therefore does not imply an absolute temperature calibration.
- Client-device images remain `client-device-tlc-pending` until the exact formulation and calibration curve are available. Colour thresholds/segmentation in the current baseline are provisional and must not be described as calibrated temperature measurement.
- The QA lane preserves and separates TLC/device provenance, but it does not create a validated cross-formulation normalization model.
- LEFT/RIGHT pairing is conservative: same position plus compatible device provenance. Unmatched or conflicting device provenance is left unpaired rather than compared silently.
- The bundled DINOv2 endpoint is reference analysis. Any live-DINO lane integrated later must retain the same finite-JSON, profile-provenance, domain-shift, and no-clinical-claim guarantees.
- No current endpoint is allowed to return a cancer probability, diagnosis, diagnostic recommendation, or calibrated absolute temperature derived only from colour without formulation-specific calibration and later clinical validation.
