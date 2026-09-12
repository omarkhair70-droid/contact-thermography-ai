# MumGuard Human Product Closure — 2026-09-12

This branch closes the remaining end-to-end human product path on top of the merged signal-first architecture from PR #42.

## Scope

1. Preserve the current human-first measurement stack: TLC signal, overlap/de-duplication, LEFT/RIGHT reconstruction, three-channel evidence, local DINO evidence and fusion.
2. Add an explicit downstream human decision contract instead of relabelling measurement evidence as cancer risk.
3. Finish the operator-facing human examination workflow and session-level result presentation.
4. Carry acquisition/QC metadata needed for real MumGuard sessions without turning ordinary engineering assumptions into client blockers.
5. Harden the report and API around decision/calibration status.
6. Run regression/CI, then verify deployment and live smoke separately before claiming production green.

## Decision-layer rule

`overall_measurement_evidence_score` is never treated as a cancer probability.

The decision layer returns:

- `INCONCLUSIVE` when measurement support is insufficient;
- `NOT_CALIBRATED` when human measurement evidence exists but no outcome-linked human decision calibration is active;
- `INDICATION_LOW`, `INDICATION_INTERMEDIATE`, or `INDICATION_HIGH` only when a separate human model score and explicit calibration artifact are supplied.

`risk_score` remains `null` in the current uncalibrated human path.

## Current branch progress

- explicit `HumanDecisionInput` / `HumanDecisionResult` contract added;
- fail-closed tests added;
- session fusion now emits `human_decision`;
- HTML report now exposes decision/calibration status without making a clinical claim.

Next on this branch: dedicated session UI/result block, acquisition metadata hardening, full regression, and deployment/live smoke verification.
