# Human Product Closure Acceptance Criteria

The branch is not ready to merge until all of the following are true:

- a real human bilateral session can be entered as LEFT/RIGHT capture groups without manual API construction;
- session result visibly exposes core hyperthermia, bilateral asymmetry, abnormal skin behaviour, AI evidence and measurement support;
- the downstream decision layer is explicit and fail-closed;
- no uncalibrated evidence score is labelled cancer probability or clinical risk;
- acquisition metadata/QC needed for real MumGuard testing is persisted;
- report/API preserve profile and uncertainty provenance;
- regression/CI is green;
- the branch is merged into `integration` before any production deployment claim;
- post-merge live smoke confirms the deployed integration SHA, health, DB, storage, DINO runtime, report and bilateral session path.

Human outcome-linked calibration and clinical validation remain evidence activities after software closure; the software interfaces for them must already exist.
