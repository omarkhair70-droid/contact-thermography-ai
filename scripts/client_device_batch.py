from __future__ import annotations

import argparse
import hashlib
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
    letterbox_square,
    robust_colour_domain_shift,
    summarize_threshold_sweep,
    threshold_sweep,
)
from app.services.client_device_embedding import (
    pairwise_cosine_distance_frame,
    summarize_reference_domain_shift,
)
from app.services.client_device_qc import assess_client_device_quality


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_blind_freeze(
    paths: list[Path],
    ranking_path: Path,
    output_dir: Path,
) -> dict:
    """Freeze exact inputs + pre-label response ranking before tumor sizes arrive."""

    payload = {
        "freeze_type": "pre-label blind client-device response ranking",
        "tlc_profile_id": "client-device-tlc-pending",
        "image_count": len(paths),
        "images": [
            {"source_image": path.name, "sha256": _sha256(path)} for path in paths
        ],
        "ranking_file": ranking_path.name,
        "ranking_sha256": _sha256(ranking_path),
        "segmentation_config": DEFAULT_CLIENT_CONFIG.__dict__,
        "labels_seen": False,
        "semantics": (
            "freezes a thermochromic response-area ranking before tumor-size labels; "
            "not tumor size, probability, diagnosis, or cancer risk"
        ),
        "clinical_claim": "NONE",
    }
    freeze_path = output_dir / "blind_freeze_manifest.json"
    freeze_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _json_safe_dino_record(record: dict) -> dict:
    safe = {}
    for key, value in record.items():
        if isinstance(value, (dict, list, tuple)):
            safe[key] = json.dumps(value, sort_keys=True)
        else:
            safe[key] = value
    return safe


def _attempt_live_dino(
    features: pd.DataFrame,
    normalized_by_name: dict[str, np.ndarray],
    output_dir: Path,
) -> tuple[str, pd.DataFrame]:
    try:
        from app.services import dinov2_service

        records = []
        embeddings = []
        client_ids = []
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
            result, embedding = dinov2_service.analyze_plate_rgb(
                cv2.cvtColor(normalized_by_name[name], cv2.COLOR_BGR2RGB),
                lct_features,
                "client-device-tlc-pending",
            )
            records.append(_json_safe_dino_record({"source_image": name, **result}))
            embeddings.append(np.asarray(embedding, dtype=np.float32))
            client_ids.append(name)

        embedding_matrix = np.stack(embeddings).astype(np.float32)
        np.save(output_dir / "client_device_dinov2_embeddings.npy", embedding_matrix)

        pairwise = pairwise_cosine_distance_frame(client_ids, embedding_matrix)
        pairwise.to_csv(output_dir / "client_client_dinov2_cosine_distance.csv")

        reference_ids = dinov2_service.REFERENCE_INDEX["plate_id"].astype(str).tolist()
        embedding_shift, nearest = summarize_reference_domain_shift(
            client_ids,
            embedding_matrix,
            reference_ids,
            dinov2_service.REFERENCE_EMBEDDINGS,
        )
        nearest.to_csv(output_dir / "client_reference_nearest_dinov2.csv", index=False)
        (output_dir / "dinov2_embedding_domain_shift.json").write_text(
            json.dumps(embedding_shift, indent=2), encoding="utf-8"
        )
        return "ok", pd.DataFrame(records)
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}", pd.DataFrame()


def run(input_dir: Path, output_dir: Path, pattern: str, attempt_dino: bool) -> dict:
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        raise SystemExit(f"No images matched {input_dir / pattern}")

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_rows = []
    qc_rows = []
    visuals = []
    normalized_by_name = {}
    for path in paths:
        features, normalized, mask = analyze_client_image(path, DEFAULT_CLIENT_CONFIG)
        original = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if original is None:
            raise ValueError(f"Could not read image for QC: {path}")
        _, valid = letterbox_square(original)
        qc = assess_client_device_quality(normalized, mask, valid)
        feature_rows.append(features)
        qc_rows.append(
            {
                "source_image": path.name,
                "status": qc["status"],
                "flags": json.dumps(qc["flags"]),
                **qc["metrics"],
                "clinical_claim": "NONE",
            }
        )
        normalized_by_name[path.name] = normalized
        visuals.append((path.stem, normalized, mask))

    features = pd.DataFrame(feature_rows)
    features.to_csv(output_dir / "client_device_features.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(output_dir / "client_device_qc.csv", index=False)

    sweep = threshold_sweep(paths)
    sweep.to_csv(output_dir / "threshold_sweep.csv", index=False)
    ranking = summarize_threshold_sweep(sweep)
    ranking_path = output_dir / "blind_response_area_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    freeze = _write_blind_freeze(paths, ranking_path, output_dir)

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
        dino_status, dino = _attempt_live_dino(features, normalized_by_name, output_dir)
        if not dino.empty:
            dino.to_csv(output_dir / "client_device_dinov2.csv", index=False)

    outputs = [
        "client_device_features.csv",
        "client_device_qc.csv",
        "threshold_sweep.csv",
        "blind_response_area_ranking.csv",
        "blind_freeze_manifest.json",
        "colour_domain_shift.json",
        "client_device_contact_sheet.png",
    ]
    if not dino.empty:
        outputs.extend(
            [
                "client_device_dinov2.csv",
                "client_device_dinov2_embeddings.npy",
                "client_client_dinov2_cosine_distance.csv",
                "client_reference_nearest_dinov2.csv",
                "dinov2_embedding_domain_shift.json",
            ]
        )

    manifest = {
        "analysis_type": "client-device contact TLC research batch",
        "tlc_profile_id": "client-device-tlc-pending",
        "image_count": len(paths),
        "images": [path.name for path in paths],
        "segmentation_config": DEFAULT_CLIENT_CONFIG.__dict__,
        "blind_freeze_ranking_sha256": freeze["ranking_sha256"],
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
