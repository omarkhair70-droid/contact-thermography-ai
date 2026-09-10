from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.client_device_domain import (
    DEFAULT_CLIENT_CONFIG,
    analyze_client_image,
    robust_colour_domain_shift,
    summarize_threshold_sweep,
    threshold_sweep,
)


def _contact_sheet(items: list[tuple[str, np.ndarray, np.ndarray]], output: Path) -> None:
    if not items:
        return
    tile_w = 768
    cols = 3
    rows = int(np.ceil(len(items) / cols))
    canvas = np.full((rows * 310, cols * 780, 3), 245, dtype=np.uint8)
    for index, (name, image, mask) in enumerate(items):
        row, col = divmod(index, cols)
        overlay = image.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        panel = np.hstack([image, overlay, mask_bgr])
        panel = cv2.resize(panel, (tile_w, 256), interpolation=cv2.INTER_AREA)
        x = col * 780
        y = row * 310
        canvas[y + 30 : y + 286, x : x + tile_w] = panel
        cv2.putText(
            canvas,
            name,
            (x + 8, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def _attempt_live_dino(
    features: pd.DataFrame,
    normalized_by_name: dict[str, np.ndarray],
) -> tuple[str, pd.DataFrame]:
    try:
        from app.services import dinov2_service

        records = []
        for row in features.to_dict("records"):
            name = row["source_image"]
            lct_features = {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "source_image",
                    "morphology_descriptor",
                    "selected_components",
                    "segmentation_config",
                    "clinical_claim",
                    "absolute_temperature_interpretation",
                }
            }
            result, _ = dinov2_service.analyze_plate_rgb(
                cv2.cvtColor(normalized_by_name[name], cv2.COLOR_BGR2RGB),
                lct_features,
                "client-device-tlc-pending",
            )
            records.append({"source_image": name, **result})
        return "ok", pd.DataFrame(records)
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}", pd.DataFrame()


def run(input_dir: Path, output_dir: Path, pattern: str, attempt_dino: bool) -> dict:
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        raise SystemExit(f"No images matched {input_dir / pattern}")

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_rows = []
    visuals = []
    normalized_by_name = {}
    for path in paths:
        features, normalized, mask = analyze_client_image(path, DEFAULT_CLIENT_CONFIG)
        feature_rows.append(features)
        normalized_by_name[path.name] = normalized
        visuals.append((path.stem, normalized, mask))

    features = pd.DataFrame(feature_rows)
    features.to_csv(output_dir / "client_device_features.csv", index=False)

    sweep = threshold_sweep(paths)
    sweep.to_csv(output_dir / "threshold_sweep.csv", index=False)
    ranking = summarize_threshold_sweep(sweep)
    ranking.to_csv(output_dir / "blind_response_area_ranking.csv", index=False)

    reference_path = ROOT / "data" / "features_v0_1.csv"
    reference = pd.read_csv(reference_path)
    domain_shift = robust_colour_domain_shift(reference, features)
    (output_dir / "colour_domain_shift.json").write_text(
        json.dumps(domain_shift, indent=2), encoding="utf-8"
    )

    _contact_sheet(visuals, output_dir / "client_device_contact_sheet.png")

    dino_status = "not_requested"
    dino = pd.DataFrame()
    if attempt_dino:
        dino_status, dino = _attempt_live_dino(features, normalized_by_name)
        if not dino.empty:
            dino.to_csv(output_dir / "client_device_dinov2.csv", index=False)

    outputs = [
        "client_device_features.csv",
        "threshold_sweep.csv",
        "blind_response_area_ranking.csv",
        "colour_domain_shift.json",
        "client_device_contact_sheet.png",
    ]
    if not dino.empty:
        outputs.append("client_device_dinov2.csv")

    manifest = {
        "analysis_type": "client-device contact TLC research batch",
        "tlc_profile_id": "client-device-tlc-pending",
        "image_count": len(paths),
        "images": [path.name for path in paths],
        "segmentation_config": DEFAULT_CLIENT_CONFIG.__dict__,
        "dino_status": dino_status,
        "blind_ranking_semantics": (
            "within-cohort thermochromic response-area ranking only; "
            "not tumor size, probability, diagnosis, or cancer risk"
        ),
        "absolute_temperature_interpretation": "DISABLED_UNCALIBRATED_TLC",
        "clinical_claim": "NONE",
        "outputs": outputs,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.jpg")
    parser.add_argument("--attempt-dino", action="store_true")
    args = parser.parse_args()
    manifest = run(args.input_dir, args.output_dir, args.pattern, args.attempt_dino)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
