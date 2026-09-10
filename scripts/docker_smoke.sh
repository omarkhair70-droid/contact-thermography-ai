#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-contact-thermography-ai:qa-smoke}"
CONTAINER_NAME="${CONTAINER_NAME:-contact-thermography-ai-qa-smoke}"
SMOKE_PORT="${SMOKE_PORT:-18000}"
BASE_URL="http://127.0.0.1:${SMOKE_PORT}"
TMP_DIR="$(mktemp -d)"

cleanup() {
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker build -t "${IMAGE_NAME}" .
docker run -d --name "${CONTAINER_NAME}" -p "${SMOKE_PORT}:8000" "${IMAGE_NAME}" >/dev/null

for _ in $(seq 1 40); do
  if curl -fsS "${BASE_URL}/health" >"${TMP_DIR}/health.json"; then
    break
  fi
  sleep 0.5
done

curl -fsS "${BASE_URL}/health" >"${TMP_DIR}/health.json"
curl -fsS "${BASE_URL}/api/reference/dinov2" >"${TMP_DIR}/dinov2.json"
curl -fsS "${BASE_URL}/api/exams" >"${TMP_DIR}/history-before.json"

python - "${TMP_DIR}/smoke.ppm" <<'PY'
from pathlib import Path
import sys

path=Path(sys.argv[1])
w=h=96
pixels=bytearray()
for y in range(h):
    for x in range(w):
        pixels.extend(((x * 2) % 256, (y * 2) % 256, ((x + y) * 2) % 256))
path.write_bytes(f"P6\n{w} {h}\n255\n".encode("ascii") + pixels)
PY

METADATA='[{"filename":"smoke.ppm","side":"LEFT","position":"P1","tlc_profile_id":"client-device-tlc-pending","device_profile_id":"docker-smoke-device"}]'
curl -fsS \
  -F "files=@${TMP_DIR}/smoke.ppm;type=image/x-portable-pixmap" \
  -F "exam_id=docker-smoke" \
  -F "metadata_json=${METADATA}" \
  "${BASE_URL}/api/exams/analyze" >"${TMP_DIR}/analyze.json"

curl -fsS "${BASE_URL}/api/exams/docker-smoke" >"${TMP_DIR}/detail.json"
curl -fsS "${BASE_URL}/api/exams" >"${TMP_DIR}/history-after.json"
curl -fsS "${BASE_URL}/reports/docker-smoke" >"${TMP_DIR}/report.html"

python - "${TMP_DIR}" <<'PY'
from pathlib import Path
import json
import math
import re
import sys

root=Path(sys.argv[1])

def load(name):
    return json.loads((root/name).read_text())

def assert_finite_json(value):
    if isinstance(value, dict):
        for child in value.values():
            assert_finite_json(child)
    elif isinstance(value, list):
        for child in value:
            assert_finite_json(child)
    elif isinstance(value, float):
        assert math.isfinite(value)

def assert_no_current_clinical_claim(value):
    forbidden={"cancer_probability","cancer_risk","clinical_probability","diagnosis","diagnostic_result","diagnostic_probability"}
    if isinstance(value, dict):
        for key, child in value.items():
            low=key.lower()
            if low == "clinical_claim":
                assert child == "NONE", (key, child)
            if low == "clinical_risk":
                assert child is None, (key, child)
            if low in forbidden:
                assert child in (None, False, "", "NONE"), (key, child)
            assert_no_current_clinical_claim(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_current_clinical_claim(child)

health=load("health.json")
assert health["status"] == "ok"
assert health["clinical_claim"] == "NONE"

dino=load("dinov2.json")
assert dino["backbone"] == "dinov2_vits14"
assert dino["clinical_claim"] == "NONE"

analysis=load("analyze.json")
assert analysis["exam_id"] == "docker-smoke"
assert analysis["tlc_profile_id"] == "client-device-tlc-pending"
assert analysis["profile_provenance"]["domain_status"] == "SEPARATE_TLC_DOMAIN"
assert analysis["profile_provenance"]["device_profile_ids"] == ["docker-smoke-device"]
assert analysis["clinical_claim"] == "NONE"

detail=load("detail.json")
assert detail["result"]["tlc_profile_id"] == "client-device-tlc-pending"

history=load("history-after.json")
assert any(item["exam_id"] == "docker-smoke" for item in history["items"])

for payload in (health,dino,analysis,detail,history):
    assert_finite_json(payload)
    assert_no_current_clinical_claim(payload)

report=(root/"report.html").read_text()
assert "client-device-tlc-pending" in report
assert "docker-smoke-device" in report
assert "Clinical claim:</b> NONE" in report
assert re.search(r"cancer probability\s*[:=]\s*(?:\d|0\.)", report, re.I) is None
assert re.search(r"diagnosis\s*[:=]", report, re.I) is None
print("Docker smoke: PASS")
PY
