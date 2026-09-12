"""Reproduce bounded conclusions from committed evidence; never refit stored scores."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def audit(root):
    truth_path = root / "data/client_mouse_ground_truth_template.csv"
    blind_path = root / "data/client_mouse_blind_predictions_v1.csv"
    model_path = root / "artifacts/lct_native_exp8v1_dino.json"
    with truth_path.open(encoding="utf-8-sig") as handle:
        truth = list(csv.DictReader(handle))
    with blind_path.open(encoding="utf-8-sig") as handle:
        blind = list(csv.DictReader(handle))
    model = json.loads(model_path.read_text())
    labels = {r["subject_id"]: r["tumor_status"] == "TUMOR_BEARING" for r in truth}
    assert len(labels) == len(truth) == 9 and sum(labels.values()) == 8
    tp = sum(labels[r["subject_id"]] and r["blind_prediction"] == "LIKELY_TUMOR" for r in blind)
    fn = sum(labels[r["subject_id"]] and r["blind_prediction"] == "LIKELY_NO_TUMOR" for r in blind)
    tn = sum(not labels[r["subject_id"]] and r["blind_prediction"] == "LIKELY_NO_TUMOR" for r in blind)
    fp = sum(not labels[r["subject_id"]] and r["blind_prediction"] == "LIKELY_TUMOR" for r in blind)
    fitted = model["training"]["resubstitution_scores"]
    low_pos = min(r["score"] for r in fitted if r["y"] == 1)
    high_neg = max(r["score"] for r in fitted if r["y"] == 0)
    # Fold construction is a design receipt, not evaluated model performance.
    positives = sorted(s for s, y in labels.items() if y)
    folds = [{"held_out_positive": s, "train_subjects": sorted(set(labels)-{s}),
              "test_subjects": [s], "independent_negative_test_subjects": 0} for s in positives]
    return {"clinical_claim": "NONE", "independent_subjects": 9, "positive_subjects": 8,
        "negative_subjects": 1,
        "historical_blind_rule_audit": {"tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "positive_cases_flagged": f"{tp}/8", "negative_cases_unflagged": f"{tn}/1",
            "interpretation": "retrospective scoring of stored pre-map hypotheses; historical freeze not reverified; not clinical validation"},
        "stored_training_fit": {"correct": sum((r["score"] >= .5) == bool(r["y"]) for r in fitted),
            "total": len(fitted), "smallest_positive_margin_to_threshold": low_pos-.5,
            "threshold_separation_interval": {"exclusive_lower": high_neg, "inclusive_upper": low_pos},
            "interpretation": "resubstitution only; source embeddings absent from git checkout"},
        "label_permutation_resolution": {"distinct_8v1_assignments": math.comb(9,1),
            "minimum_one_sided_exhaustive_p": 1/math.comb(9,1),
            "interpretation": "requires exchangeable subjects and full refit for each label assignment; no significance test executed"},
        "illustrative_fixed_threshold_independent_validation": {
            "one_correct_negative_two_sided_95_lower": .025,
            "all_eight_positives_correct_two_sided_95_lower": .025**(1/8),
            "zero_false_positives_n_for_one_sided_95_upper_below_0_10": math.ceil(math.log(.05)/math.log(.90)),
            "zero_false_positives_n_for_one_sided_95_upper_below_0_05": math.ceil(math.log(.05)/math.log(.95)),
            "interpretation": "sample-size arithmetic only, not bounds on this training cohort"},
        "leave_one_positive_out_design": folds,
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [truth_path, blind_path, model_path]}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "leave_one_positive_out_design"}, indent=2))
