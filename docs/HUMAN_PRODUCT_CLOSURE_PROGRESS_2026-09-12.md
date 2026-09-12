# Human Product Closure Progress

Branch: `feat/human-product-closure-20260912`
Base: `integration`

Implemented so far:

- Added `app/services/human_decision.py` with explicit downstream decision contract.
- Added fail-closed regression tests in `tests/test_human_decision.py`.
- Wired `human_decision` into `mumguard_session_fusion` without converting measurement evidence into cancer probability.
- Updated the HTML report to display decision status, indication availability, calibration state and risk availability.

Remaining closure work on this branch:

- show session-level three-channel evidence and human decision state directly in the operator UI;
- expand human acquisition metadata/QC contract for real MumGuard sessions;
- add/adjust API regression coverage for the new human decision output;
- run full CI and repair any regressions;
- only after merge, verify the deployed `integration` SHA, `/health`, DB, storage, DINO runtime, report and bilateral session smoke on Coolify.
