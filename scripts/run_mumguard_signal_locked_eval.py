"""Run a frozen, no-fit TLC signal challenge on the existing private mouse captures.

This evaluates the new signal/physics representation without training a disease head.
The protocol is written before outcome labels are used for scoring. The supplied
cohort is development-exposed and contains one no-tumor subject, so the output is
an engineering ranking challenge, not independent validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from app.services.client_device_domain import analyze_client_image
from app.services.thermal_anomaly_engine import build_multiscale_thermal_anomaly
from app.services.tlc_signal_processing import build_relative_thermal_map


PROTOCOL = {
    "version": "mumguard-signal-locked-v1",
    "feature_contract": "relative TLC signal + multiscale local thermal anomaly",
    "tlc_profile_id": "client-device-tlc-pending",
    "relative_hue_span_deg": [0.0, 240.0],
    "anomaly_radii_px": [3, 7, 15],
    "hotter_is_higher_working_assumption": True,
    "composite_weights": {
        "core_hyperthermia_score": 0.58,
        "abnormal_skin_behavior_score": 0.42,
    },
    "rank_features": [
        "core_hyperthermia_score",
        "abnormal_skin_behavior_score",
        "static_signal_evidence_score",
        "global_anomaly_score",
    ],
    "selection": "No fitting and no threshold selection. Compare frozen scores with the sole negative only after every feature row is complete.",
    "clinical_claim": "NONE",
}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _manifest_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError("manifest is empty")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _core_score(anomaly) -> float | None:
    evidence = np.asarray(anomaly.evidence_map, dtype=np.float32)
    signed = np.asarray(anomaly.signed_contrast_map, dtype=np.float32)
    valid = np.asarray(anomaly.observable_mask, dtype=bool) & np.isfinite(evidence) & np.isfinite(signed)
    if not valid.any():
        return None
    scale = max(float(np.std(signed[valid])), 1e-6)
    direction = signed if PROTOCOL["hotter_is_higher_working_assumption"] else -signed
    positive = valid & (direction > 0.0)
    if not positive.any():
        return 0.0
    strength = evidence[positive] * np.clip(direction[positive] / scale, 0.0, 4.0) / 4.0
    return float(np.mean(np.sort(strength)[-max(1, strength.size // 10):]))


def _pattern_score(anomaly) -> float:
    f = anomaly.features
    return float(np.clip(
        0.55 * float(f.get("global_anomaly_score", 0.0))
        + 0.25 * float(f.get("mean_persistence", 0.0))
        + 0.20 * min(1.0, 10.0 * float(f.get("high_evidence_fraction", 0.0))),
        0.0,
        1.0,
    ))


def extract_one(image_path: Path, subject_id: str) -> dict:
    features, normalized_bgr, response_mask = analyze_client_image(image_path)
    response_mask_bool = np.asarray(response_mask) > 0
    signal = build_relative_thermal_map(
        normalized_bgr,
        response_mask_bool,
        low_hue_deg=PROTOCOL["relative_hue_span_deg"][0],
        high_hue_deg=PROTOCOL["relative_hue_span_deg"][1],
        tlc_profile_id=PROTOCOL["tlc_profile_id"],
    )
    anomaly = build_multiscale_thermal_anomaly(
        signal.signal_map,
        signal.active_mask,
        radii=tuple(PROTOCOL["anomaly_radii_px"]),
        signal_mode="relative_hue_index",
    )
    core = _core_score(anomaly)
    pattern = _pattern_score(anomaly)
    composite = None if core is None else float(
        PROTOCOL["composite_weights"]["core_hyperthermia_score"] * core
        + PROTOCOL["composite_weights"]["abnormal_skin_behavior_score"] * pattern
    )
    return {
        "subject_id": subject_id,
        "filename": image_path.name,
        "response_area_fraction": float(features["response_area_fraction"]),
        "component_count": int(features["component_count"]),
        "morphology_descriptor": features["morphology_descriptor"],
        "observable_fraction": float(anomaly.features["observable_fraction"]),
        "comparable_fraction": float(anomaly.features["comparable_fraction"]),
        "global_anomaly_score": float(anomaly.features["global_anomaly_score"]),
        "mean_persistence": float(anomaly.features["mean_persistence"]),
        "core_hyperthermia_score": core,
        "abnormal_skin_behavior_score": pattern,
        "static_signal_evidence_score": composite,
        "clinical_claim": "NONE",
    }


def _rank_summary(rows: list[dict], labels: dict[str, str]) -> dict:
    negatives = [row for row in rows if labels[row["subject_id"]] == "absent"]
    positives = [row for row in rows if labels[row["subject_id"]] == "present"]
    if len(negatives) != 1:
        raise ValueError("locked development challenge expects exactly one absent subject")
    negative = negatives[0]
    features = {}
    for name in PROTOCOL["rank_features"]:
        neg = negative.get(name)
        usable = [row for row in positives if row.get(name) is not None and neg is not None]
        margins = [float(row[name]) - float(neg) for row in usable]
        features[name] = {
            "negative_subject": negative["subject_id"],
            "negative_score": neg,
            "positive_subjects_compared": len(usable),
            "positives_above_negative": int(sum(value > 0 for value in margins)),
            "ties": int(sum(abs(value) <= 1e-12 for value in margins)),
            "median_positive_minus_negative": float(np.median(margins)) if margins else None,
            "min_positive_minus_negative": float(np.min(margins)) if margins else None,
            "max_positive_minus_negative": float(np.max(margins)) if margins else None,
        }
    return {
        "semantics": "locked development ranking; no fitted disease head and no independent validation",
        "feature_results": features,
        "subject_count": len(rows),
        "positive_subjects": len(positives),
        "negative_subjects": len(negatives),
        "clinical_claim": "NONE",
    }


def run(manifest: Path, image_root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "protocol.json", PROTOCOL)

    manifest_rows = _manifest_rows(manifest)
    acquisitions = [
        {
            "subject_id": row["subject_id"],
            "filename": row["filename"],
            "source_sha256": row["source_sha256"],
        }
        for row in manifest_rows
    ]

    feature_rows = []
    for item in sorted(acquisitions, key=lambda value: value["subject_id"]):
        image_path = image_root / item["filename"]
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if _sha256(image_path) != item["source_sha256"]:
            raise ValueError(f"hash mismatch for {image_path.name}")
        feature_rows.append(extract_one(image_path, item["subject_id"]))

    fieldnames = list(feature_rows[0])
    with (out / "feature_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)

    labels = {row["subject_id"]: row["lesion_presence"] for row in manifest_rows}
    ranking = _rank_summary(feature_rows, labels)
    scored_rows = [
        {**row, "lesion_presence": labels[row["subject_id"]]}
        for row in feature_rows
    ]
    with (out / "scored_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored_rows[0]))
        writer.writeheader()
        writer.writerows(scored_rows)
    _write_json(out / "ranking_summary.json", ranking)
    _write_json(out / "execution.json", {
        "protocol_written_before_label_scoring": True,
        "disease_head_fitted": False,
        "threshold_selected": False,
        "manifest_sha256": _sha256(manifest),
        "subjects": [row["subject_id"] for row in feature_rows],
        "clinical_claim": "NONE",
    })
    print(json.dumps(ranking, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/mumguard_acquisitions_v1.jsonl"))
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/mumguard-signal-locked-v1"))
    args = parser.parse_args()
    run(args.manifest, args.image_root, args.out)


if __name__ == "__main__":
    main()
