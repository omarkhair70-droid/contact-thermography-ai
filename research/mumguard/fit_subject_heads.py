"""Exploratory mouse heads with subject grouping and leave-one-positive-out receipts.

Consumes the evidence JSON from local_contrast.py. No human training or deployment.
The negative remains in every fitted fold; its predictions are resubstitution only.
All annotations/segmentation on the development cohort preclude pristine validation.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

FEATURES = ["physical_q90", "physical_median", "hue_js_q90",
            "abs_saturation_delta_q90", "abs_value_delta_q90", "dino_distance_q90"]


def features(e):
    rows = e["candidates"]
    if not rows:
        raise ValueError("no candidates; do not encode missing signal as zero/healthy")
    get = lambda key: np.asarray([r[key] for r in rows], float)
    dino = [r["dino_local_cosine_distance"] for r in rows if r["dino_local_cosine_distance"] is not None]
    return np.array([np.quantile(get("physical_contrast"), .9), np.median(get("physical_contrast")),
        np.quantile(get("hue_js"), .9), np.quantile(np.abs(get("saturation_signed_delta")), .9),
        np.quantile(np.abs(get("value_signed_delta")), .9),
        np.quantile(dino, .9) if dino else np.nan])


def fit(X, y, ridge=1.):
    # Every row here is a subject, never an image/patch. Fold-local standardization.
    X, y = np.asarray(X, float), np.asarray(y)
    if X.ndim != 2 or y.shape != (len(X),) or not np.isfinite(X).all() or not set(y).issubset({0, 1}):
        raise ValueError("finite subject features and explicit binary endpoints required")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("positive finite ridge required")
    if len(set(y)) != 2:
        raise ValueError("two classes required")
    mean, scale = X.mean(0), np.maximum(X.std(0), .01)
    A = np.column_stack([np.ones(len(X)), (X-mean)/scale])
    weight = np.where(y == 1, .5/np.sum(y == 1), .5/np.sum(y == 0))
    beta = np.zeros(A.shape[1])
    penalty = np.diag([.01] + [ridge]*(A.shape[1]-1))
    for _ in range(80):
        p = 1 / (1 + np.exp(-np.clip(A @ beta, -30, 30)))
        gradient = A.T @ (weight*(p-y)) + penalty @ beta
        hessian = A.T @ ((weight*p*(1-p))[:, None]*A) + penalty
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return {"mean": mean.tolist(), "scale": scale.tolist(), "coef": beta[1:].tolist(),
            "intercept": float(beta[0]), "ridge": ridge}


def score(model, X):
    z = (X-np.array(model["mean"]))/np.array(model["scale"])
    return 1/(1+np.exp(-np.clip(z @ model["coef"]+model["intercept"], -30, 30)))


def load_manifest(path):
    grouped, hashes, domains = {}, {}, set()
    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["species"] != "mouse" or row["label"] not in {"TUMOR_BEARING", "HEALTHY"}:
            raise ValueError("this development lane only accepts explicit mouse tumor-bearing/healthy labels")
        key = row["subject_id"]
        file = Path(row["evidence_path"])
        if not file.is_absolute():
            file = path.parent/file
        evidence = json.loads(file.read_text())
        provenance = evidence["provenance"]
        for field in ("subject_id", "species", "tlc_profile_id", "device_profile_id"):
            if row[field] != provenance[field]:
                raise ValueError(f"manifest/evidence {field} mismatch")
        domains.add((row["species"], row["tlc_profile_id"], row["device_profile_id"]))
        source_hash = provenance["source_sha256"]
        if source_hash in hashes:
            raise ValueError("duplicate original image hash; repeats must be genuine acquisitions")
        hashes[source_hash] = key
        y = int(row["label"] == "TUMOR_BEARING")
        if key in grouped and grouped[key]["y"] != y:
            raise ValueError("inconsistent subject labels")
        grouped.setdefault(key, {"y": y, "features": []})["features"].append(features(evidence))
    if len(domains) != 1:
        raise ValueError("silent species/profile/device pooling is forbidden")
    ids = sorted(grouped)
    # Missing DINO remains missing; never silently impute a fused model input.
    X = np.array([np.median(grouped[s]["features"], axis=0) for s in ids])
    y = np.array([grouped[s]["y"] for s in ids])
    return ids, X, y, next(iter(domains))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--acknowledge-development-only", action="store_true", required=True)
    args = p.parse_args()
    ids, X, y, domain = load_manifest(args.manifest)
    expected_ids = [f"CLIENT-MOUSE-{i:04d}" for i in range(30,39)]
    expected_labels = [1]*8+[0]
    if ids != expected_ids or y.tolist() != expected_labels:
        raise ValueError("this bounded experiment expects the known 8/1 nine-subject cohort")
    rows, models = [], {}
    blocks = {"B_physical": [0], "F_physical_detail": [0,1,2,3,4]}
    if np.isfinite(X[:,5]).all():
        blocks["B_dino"] = [5]
        blocks["F_fused"] = [0,1,2,3,4,5]
    for name, cols in blocks.items():
        data = X[:, cols]
        for test in np.where(y == 1)[0]:
            train = np.arange(len(ids)) != test
            model = fit(data[train], y[train])
            pred = float(score(model, data[test:test+1])[0])
            negative_fit = float(score(model, data[y == 0])[0])
            rows.append({"candidate": name, "held_out_subject": ids[test], "held_out_label": 1,
                "held_out_score": pred, "negative_training_score": negative_fit,
                "held_out_minus_negative_training_score": pred-negative_fit,
                "train_subjects": [s for k,s in enumerate(ids) if train[k]],
                "independent_negative_test_subjects": 0,
                "status": "EXPLORATORY_DEVELOPMENT_LOPO_NOT_HUMAN_VALIDATION"})
        models[name] = {"features": [FEATURES[c] for c in cols], **fit(data, y)}
    result = {"clinical_claim": "NONE", "activation_status": "INACTIVE_RESEARCH_CANDIDATE",
        "species": domain[0], "tlc_profile_id": domain[1], "device_profile_id": domain[2],
        "subjects": ids, "validation_status": "NOT_VALIDATED_SINGLE_NEGATIVE",
        "folds": rows, "development_fit_models": models,
        "skipped_candidates": [] if "F_fused" in blocks else ["B_dino", "F_fused"],
        "threshold": None, "human_binary_inference": "DISABLED"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"subjects": len(ids), "candidates": list(models), "fold_results": len(rows),
                      "clinical_claim": "NONE"}))


if __name__ == "__main__":
    main()
