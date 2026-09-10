# Lane: Live DINOv2

Branch: `feat/live-dinov2`

Goal: make the trained DINOv2 reference branch work on every newly uploaded plate.

Acceptance criteria:
- Load standard `dinov2_vits14` from a pinned/cacheable source.
- Encode each new normalized plate into a 384-d embedding.
- Run nearest-reference retrieval and DINO reference anomaly head.
- Fuse the new embedding with the 18 explicit LCT features and run the fused head.
- For true LEFT/RIGHT pairs, run the DINO pair representation/head after geometric pairing.
- Return all learned outputs through the existing exam result contract.
- Preserve `clinical_risk=None` and `clinical_claim=NONE`.
- Add tests for a 10-plate composite upload and a true left/right pair.
- Do not redesign the UI or deployment stack.
