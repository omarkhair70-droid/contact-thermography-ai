# Target-domain LCT hunt execution notes

- Branch: `feat/lct-target-data-hunt`
- Base: `integration`
- Scope: public/rights-aware contact-LCT source discovery + OOD abstention utility only.
- No clinical classifier has been enabled on client-device images.
- The current nine client mouse images remain a frozen tumour-bearing cohort.
- The DMR-IR classifier remains auxiliary infrared research.

## Current executable guard

`scripts/ood_abstain_gate.py` fits a cosine nearest-source threshold from leave-one-out source-neighbour distances and marks target embeddings outside that range as `ABSTAIN_OOD`. This is intended to prevent an auxiliary source-domain head from silently issuing a binary class on a visibly shifted target domain.

`clinical_claim = NONE`
