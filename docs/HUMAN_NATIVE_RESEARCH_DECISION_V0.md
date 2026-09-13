# MumGuard Native Human Research Decision v0

## Purpose

The primary MumGuard research finding must not become `INCONCLUSIVE` only because the auxiliary DMR-IR human-transfer lane is out of source domain.

This layer makes MumGuard's own human-session measurement evidence the primary research decision source. The DMR-IR v0.2.1 transfer remains auxiliary and can contribute only when its source-support gate passes.

## Inputs

The native layer consumes only already-computed human-session evidence:

- focal/core hyperthermia evidence;
- LEFT-vs-RIGHT bilateral asymmetry;
- abnormal thermal-distribution evidence;
- LEFT/RIGHT core-evidence gap;
- optional bilateral DINO visual-pattern corroboration;
- optional DMR-IR transfer score when `IN_SOURCE_SUPPORT`.

No mouse disease label or mouse classifier is used in this human decision path.

## Engineering fusion

Base native thermal research concern:

- core hyperthermia: 0.30
- abnormal thermal distribution: 0.25
- bilateral asymmetry: 0.30
- LEFT/RIGHT core gap: 0.15

When bilateral DINO corroboration is available, it contributes 0.10 and the native thermal component contributes 0.90.

When the reviewed DMR-IR transfer lane is in source support and executed, it contributes 0.15 and MumGuard-native evidence remains 0.85. `ABSTAIN_OOD` contributes zero and does not veto the native decision.

Engineering research bands:

- score < 0.52: `NOT_SUSPICIOUS_RESEARCH`
- score >= 0.64: `SUSPICIOUS_RESEARCH`
- otherwise: `INCONCLUSIVE`

These are versioned engineering thresholds for research-product behaviour. They are **not clinically calibrated thresholds**, sensitivity/specificity claims, or cancer probabilities.

## Fail-closed behaviour

If MumGuard measurement support is not `OK`, the result is `INCONCLUSIVE` and no research concern score is emitted.

## Output semantics

The primary result includes:

- `SUSPICIOUS_RESEARCH`, `NOT_SUSPICIOUS_RESEARCH`, or `INCONCLUSIVE`;
- Research Concern Score 0–100 when measurement support permits;
- `decision_origin=MUMGUARD_NATIVE_RESEARCH_V0`;
- contributing evidence values;
- whether DINO bilateral corroboration was used;
- whether the human-transfer reference was used;
- the transfer reference status (including `ABSTAIN_OOD`);
- `clinical_risk=null` and `clinical_claim=NONE`.

The score is explicitly not a cancer probability or diagnosis.
