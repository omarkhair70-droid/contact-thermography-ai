#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def fetch_json(base_url: str, path: str):
    request = Request(base_url.rstrip("/") + path, headers={"Accept": "application/json"})
    with urlopen(request, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"{path} returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def fetch_text(base_url: str, path: str):
    request = Request(base_url.rstrip("/") + path)
    with urlopen(request, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"{path} returned HTTP {response.status}")
        return response.read().decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description="Smoke-check a deployed Contact Thermography AI instance")
    parser.add_argument("base_url", help="Example: https://lct-staging.example.com")
    args = parser.parse_args()

    try:
        health = fetch_json(args.base_url, "/health")
        if health.get("status") != "ok":
            raise RuntimeError(f"health status is not ok: {health}")
        if health.get("database") not in {"sqlite", "postgresql"}:
            raise RuntimeError(f"unexpected database backend: {health}")
        if health.get("storage") != "filesystem":
            raise RuntimeError(f"unexpected storage backend: {health}")

        history = fetch_json(args.base_url, "/api/exams?limit=1")
        if "items" not in history or "count" not in history:
            raise RuntimeError("exam-history response is missing expected keys")

        root = fetch_text(args.base_url, "/")
        if "Contact Thermography" not in root and "Thermography" not in root:
            raise RuntimeError("dashboard response does not look like the expected application")
    except (HTTPError, URLError, RuntimeError, ValueError) as exc:
        print(f"SMOKE FAIL: {exc}", file=sys.stderr)
        return 1

    print("SMOKE PASS: health, database/storage readiness, exam-history API, and dashboard are reachable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
