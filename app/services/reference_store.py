from pathlib import Path
import json
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"

def clean_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v

def clean_record(record):
    return {k: clean_value(v) for k, v in record.items()}

class ReferenceStore:
    def __init__(self):
        self.manifest = json.loads((ARTIFACTS / "model_manifest.json").read_text())
        self.plates = pd.read_csv(ARTIFACTS / "plate_reference_scores.csv")
        self.pairs = pd.read_csv(ARTIFACTS / "bilateral_reference_scores.csv")
        self.features = pd.read_csv(DATA / "features_v0_1.csv")
        self.dino_index = pd.read_csv(ARTIFACTS / "dinov2_reference" / "plate_index.csv")
        self.dino_fused = pd.read_csv(ARTIFACTS / "dinov2_reference" / "fused_reference_scores.csv")
        self.dino_pairs = pd.read_csv(ARTIFACTS / "dinov2_reference" / "bilateral_dinov2_reference_scores.csv")

    def plate_records(self):
        merged = self.plates.merge(self.features, on="plate_id", how="left", suffixes=("", "_feature"))
        merged = merged.merge(
            self.dino_index[["plate_id","dinov2_reference_anomaly_score_0_1"]],
            on="plate_id", how="left"
        ).merge(
            self.dino_fused[["plate_id","dinov2_lct_fused_reference_score_0_1"]],
            on="plate_id", how="left"
        )
        keep = [
            "plate_id",
            "reference_anomaly_score_0_1",
            "heuristic_morphology_v0",
            "response_area_fraction",
            "component_count",
            "largest_component_fraction",
            "largest_elongation",
            "branch_pixels",
            "hue_mean",
            "hue_std",
            "dinov2_reference_anomaly_score_0_1",
            "dinov2_lct_fused_reference_score_0_1",
        ]
        records = []
        for raw in merged[keep].to_dict("records"):
            r = clean_record(raw)
            r["image_url"] = f"/static/plates/{r['plate_id']}.png"
            r["score_semantics"] = "reference unusualness only; not cancer probability"
            records.append(r)
        return records

    def pair_records(self):
        records = []
        for raw in self.pairs.to_dict("records"):
            r = clean_record(raw)
            r["visual_url"] = f"/static/pairs/{r['bilateral_pair_id']}.png"
            r["score_semantics"] = "layout-inferred reference pair unusualness only"
            records.append(r)
        return records

store = ReferenceStore()
