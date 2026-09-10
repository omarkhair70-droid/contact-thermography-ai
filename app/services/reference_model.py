from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

ENGINEERED_COLS = [
    "response_area_fraction","component_count","largest_component_fraction",
    "largest_eccentricity","largest_solidity","largest_perimeter_px",
    "largest_elongation","skeleton_length_px","branch_pixels","endpoint_pixels",
    "response_centroid_x_norm","response_centroid_y_norm",
    "hue_mean","hue_std","saturation_mean","value_mean","lab_a_mean","lab_b_mean",
]

class PortableReferenceModel:
    """Small deterministic reference-only model reconstructed from committed CSV data.

    This keeps the canonical source repository runnable without committing opaque
    joblib binaries. It is not a clinical model and does not output disease risk.
    """
    def __init__(self):
        ref = pd.read_csv(DATA / "features_v0_1.csv")
        self.plate_ids = ref["plate_id"].astype(str).tolist()
        X = np.nan_to_num(ref[ENGINEERED_COLS].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler = RobustScaler().fit(X)
        Xs = self.scaler.transform(X)
        n_components = min(8, len(X)-2, X.shape[1])
        self.pca = PCA(n_components=n_components, random_state=42).fit(Xs)
        self.Z = self.pca.transform(Xs)
        self.detector = IsolationForest(n_estimators=300, contamination="auto", random_state=42).fit(self.Z)
        self.nn = NearestNeighbors(n_neighbors=min(5, len(self.Z)), metric="cosine").fit(self.Z)
        self.ref_raw = -self.detector.score_samples(self.Z)

    @staticmethod
    def _safe(v):
        try:
            x=float(v)
            return x if math.isfinite(x) else 0.0
        except Exception:
            return 0.0

    def analyze(self, features: dict):
        x=np.asarray([[self._safe(features.get(c,0.0)) for c in ENGINEERED_COLS]], dtype=float)
        z=self.pca.transform(self.scaler.transform(x))
        raw=float(-self.detector.score_samples(z)[0])
        percentile=float((self.ref_raw <= raw).mean())
        distances,indices=self.nn.kneighbors(z)
        neighbors=[
            {"plate_id":self.plate_ids[int(i)],"cosine_distance":round(float(d),4)}
            for i,d in zip(indices[0],distances[0])
        ]
        return {
            "reference_anomaly_percentile":round(percentile,6),
            "nearest_reference_plates":neighbors,
        }

reference_model = PortableReferenceModel()
