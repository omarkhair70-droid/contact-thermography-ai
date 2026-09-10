from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.tumor_burden_validation import (
    evaluate_tumor_size_mapping,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a frozen pre-label client response ranking with later independent tumor-size measurements."
    )
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--size-column", default="tumor_size")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    expected_hash = str(freeze.get("ranking_sha256") or "")
    actual_hash = sha256_file(args.ranking)
    if not expected_hash or actual_hash != expected_hash:
        raise SystemExit(
            "Ranking file does not match the pre-label blind freeze manifest; aborting comparison."
        )
    if freeze.get("labels_seen") is not False:
        raise SystemExit("Freeze manifest does not represent a pre-label blind state")

    ranking = pd.read_csv(args.ranking)
    labels = pd.read_csv(args.labels)
    summary, joined = evaluate_tumor_size_mapping(
        ranking,
        labels,
        size_column=args.size_column,
    )
    summary["frozen_ranking_sha256"] = actual_hash
    summary["freeze_verified"] = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.output_dir / "tumor_size_association_rows.csv", index=False)
    (args.output_dir / "tumor_size_association_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
