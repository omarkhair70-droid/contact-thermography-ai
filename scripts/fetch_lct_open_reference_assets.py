from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


MAX_ASSET_BYTES = 25 * 1024 * 1024
ALLOWED_USE_ROLE = "OPEN_LICENSE_REFERENCE"


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_asset_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validate_asset_manifest(rows)
    return rows


def validate_asset_manifest(rows: list[dict[str, str]]) -> dict[str, object]:
    required = {
        "asset_id",
        "case_id",
        "source_figure",
        "asset_url",
        "article_url",
        "license_status",
        "license_evidence_url",
        "ground_truth_label",
        "asset_scope",
        "use_role",
        "train_eligible",
        "contains_non_lct_panels",
        "lct_region_notes",
        "attribution",
    }
    if not rows:
        raise ValueError("open-reference asset manifest is empty")

    seen: set[str] = set()
    labels: set[str] = set()
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"missing asset columns: {missing}")
        asset_id = row["asset_id"].strip()
        if not asset_id:
            raise ValueError("every asset requires asset_id")
        if asset_id in seen:
            raise ValueError(f"duplicate asset_id: {asset_id}")
        seen.add(asset_id)

        if row["use_role"].strip() != ALLOWED_USE_ROLE:
            raise ValueError("asset pack may contain only OPEN_LICENSE_REFERENCE rows")
        if not row["license_status"].strip().upper().startswith("CC-BY"):
            raise ValueError("every downloadable reference asset requires explicit CC-BY status")
        if _as_bool(row["train_eligible"]):
            raise ValueError("open reference assets must remain train_eligible=false")
        if not row["license_evidence_url"].strip():
            raise ValueError("every asset requires license evidence")
        if not row["attribution"].strip():
            raise ValueError("every asset requires attribution text")

        parsed = urlparse(row["asset_url"].strip())
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError(f"asset_url must be HTTPS: {row['asset_url']}")
        labels.add(row["ground_truth_label"].strip().upper())

    return {
        "status": "GREEN",
        "assets": len(rows),
        "malignant_assets": sum(r["ground_truth_label"].strip().upper() == "MALIGNANT" for r in rows),
        "benign_or_healthy_assets": sum(r["ground_truth_label"].strip().upper() in {"BENIGN", "HEALTHY"} for r in rows),
        "train_eligible_assets": 0,
        "labels": sorted(labels),
        "clinical_claim": "NONE",
    }


def _asset_suffix(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tif", ".tiff"} else ".bin"


def fetch_assets(rows: list[dict[str, str]], out_dir: Path, timeout: int = 30) -> list[dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict[str, object]] = []

    for row in rows:
        url = row["asset_url"].strip()
        destination = out_dir / f"{row['asset_id']}{_asset_suffix(url)}"
        request = Request(url, headers={"User-Agent": "contact-thermography-ai-reference-fetch/1.0"})
        digest = hashlib.sha256()
        size = 0

        with urlopen(request, timeout=timeout) as response, destination.open("wb") as handle:
            content_type = response.headers.get("Content-Type", "")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_ASSET_BYTES:
                    destination.unlink(missing_ok=True)
                    raise ValueError(f"asset exceeds {MAX_ASSET_BYTES} bytes: {row['asset_id']}")
                digest.update(chunk)
                handle.write(chunk)

        if size == 0:
            destination.unlink(missing_ok=True)
            raise ValueError(f"empty downloaded asset: {row['asset_id']}")

        downloaded.append(
            {
                "asset_id": row["asset_id"],
                "case_id": row["case_id"],
                "filename": destination.name,
                "bytes": size,
                "sha256": digest.hexdigest(),
                "content_type": content_type,
                "license_status": row["license_status"],
                "license_evidence_url": row["license_evidence_url"],
                "attribution": row["attribution"],
                "ground_truth_label": row["ground_truth_label"],
                "use_role": ALLOWED_USE_ROLE,
                "train_eligible": False,
                "clinical_claim": "NONE",
            }
        )

    (out_dir / "download_manifest.json").write_text(
        json.dumps(downloaded, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch only explicitly open-license contact-LCT reference figures.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runtime/lct_open_reference_assets"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    rows = load_asset_manifest(args.manifest)
    summary = validate_asset_manifest(rows)
    if args.dry_run:
        print(json.dumps({**summary, "asset_ids": [r["asset_id"] for r in rows]}, indent=2))
        return

    downloaded = fetch_assets(rows, args.out, timeout=args.timeout)
    print(json.dumps({**summary, "downloaded_assets": len(downloaded), "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
