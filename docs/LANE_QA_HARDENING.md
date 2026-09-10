# Lane: QA & Hardening

Branch: `feat/qa-hardening`

Goal: make the demo candidate difficult to break and prevent misleading clinical output.

Acceptance criteria:
- API regression tests for health, upload, history, report and DINO endpoints.
- Corrupt/empty/non-image input tests.
- Multi-plate extraction regression test.
- LEFT/RIGHT metadata and bilateral pairing tests.
- JSON NaN/Inf serialization tests.
- Assert that no current endpoint returns a cancer probability / diagnostic claim.
- Docker smoke-test script.
- Demo checklist and known-limitations document.
