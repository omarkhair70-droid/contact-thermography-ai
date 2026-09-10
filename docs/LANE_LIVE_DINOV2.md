# Lane: Live DINOv2

Branch: `feat/live-dinov2`

Goal: make the trained DINOv2 reference branch work on every newly uploaded plate while preserving the TLC/device domain.

Read first:
- `docs/MASTER_HANDOFF.md`
- `docs/TLC_VARIANT_HANDLING.md`
- `config/tlc_profiles.example.json`

Acceptance criteria:
- Load standard `dinov2_vits14` from a pinned/cacheable source.
- Encode each new normalized plate into a 384-d embedding.
- Run nearest-reference retrieval and DINO reference anomaly head.
- Fuse the new embedding with the 18 explicit LCT features and run the fused head.
- For true LEFT/RIGHT pairs, run the DINO pair representation/head after geometric pairing.
- Preserve `tlc_profile_id` and optional `device_profile_id` through upload, analysis, persistence and report provenance.
- Do not assume publication/reference Hue distributions are calibrated for client-device images with a different TLC formulation.
- Keep any formulation-specific colour thresholds/profile normalization behind a TLC-profile resolver rather than global constants.
- Return all learned outputs through the existing exam result contract.
- Preserve `clinical_risk=None` and `clinical_claim=NONE`.
- Add tests for a 10-plate composite upload, a true left/right pair, and a non-default TLC profile.
- Do not redesign the UI or deployment stack.
