# Product UI Validation

Validation performed against the RC0.5 baseline while preparing the Product/UI lane.

## Green checks

- JavaScript syntax for the full new-exam/history/result client script: `node --check` — GREEN.
- Python compile check for the additive profile-provenance API changes — GREEN.
- FastAPI smoke: analyze the known 10-thermogram composite while submitting `tlc_profile_id=client-device-tlc-pending` and `device_profile_id=prototype-01` — HTTP 200, 10 plates detected.
- Persistence smoke: fetch the saved examination and confirm both TLC/device profile values persisted — GREEN.
- Backward-compatibility smoke: submit the existing analyze call without the two new profile fields — HTTP 200 and default TLC profile applied.
- Clinical guardrail smoke: response keeps `clinical_claim=NONE` — GREEN.
- Report compile/render smoke for the persisted examination — HTTP 200; report contains TLC profile, device profile and non-diagnostic wording.

## Integration expectations

The Live DINOv2 lane may add learned live-inference fields. The UI reads those fields when present and falls back to the current reference unusualness value, so the Product/UI branch does not require the AI lane to be merged first.
