from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from app.services.native_classifier_runtime import (
    native_classifier_status,
    predict_native_binary,
)
from app.services.reference_model import ENGINEERED_COLS
from app.services.target_feature_builder import feature_schema

ROOT = Path(__file__).resolve().parents[2]
DINO_DIR = ROOT / "artifacts" / "dinov2_reference"
DATA_DIR = ROOT / "data"

BACKBONE_NAME = "dinov2_vits14"
EMBEDDING_DIM = 384
REFERENCE_TLC_PROFILE_ID = "reference-publication-unknown"
CLIENT_TLC_PROFILE_ID = "client-device-tlc-pending"
EXPERIMENTAL_NATIVE_VERSION = "0.1.0-exp8v1-dino"
# Pinned to the official facebookresearch/dinov2 main commit retrieved for this lane.
DINO_HUB_REPOSITORY = (
    "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
)
SCORE_SEMANTICS = (
    "research/reference unusualness only; not a probability, diagnosis, or cancer risk"
)


class DINOv2UnavailableError(RuntimeError):
    """Raised when live DINOv2 inference cannot be initialized or executed."""


def _load_reference_inputs() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    index = pd.read_csv(DINO_DIR / "plate_index.csv")
    embeddings = np.load(DINO_DIR / "dinov2_embeddings.npy", allow_pickle=False)
    features = pd.read_csv(DATA_DIR / "features_v0_1.csv")

    expected_shape = (len(index), EMBEDDING_DIM)
    if embeddings.shape != expected_shape:
        raise RuntimeError(
            "Invalid canonical DINOv2 reference artifact: "
            f"expected shape {expected_shape}, received {embeddings.shape}"
        )
    if embeddings.dtype != np.float32 or not np.isfinite(embeddings).all():
        raise RuntimeError(
            "Invalid canonical DINOv2 reference artifact: expected finite float32 values"
        )
    rows = index["embedding_row"].to_numpy(dtype=int)
    if not np.array_equal(rows, np.arange(len(index))):
        raise RuntimeError("DINOv2 plate index rows must be contiguous and ordered")

    features = index[["plate_id"]].merge(features, on="plate_id", how="left")
    if features[ENGINEERED_COLS].isnull().any().any():
        raise RuntimeError("Every DINOv2 reference plate must have 18 explicit LCT features")
    return index, embeddings.astype(np.float32, copy=False), features


REFERENCE_INDEX, REFERENCE_EMBEDDINGS, REFERENCE_FEATURES = _load_reference_inputs()


def _safe_feature(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _lct_vector(features: dict) -> np.ndarray:
    return np.asarray([_safe_feature(features.get(name)) for name in ENGINEERED_COLS])


def _pair_vector(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate([left, right, np.abs(left - right), left * right])


class _PortableReferenceHead:
    """Deterministically reconstruct an unsupervised head from committed reference data."""

    def __init__(
        self,
        reference: np.ndarray,
        latent_dim: int | None,
        scale: bool = True,
    ):
        reference = np.nan_to_num(
            np.asarray(reference, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        self.input_dim = int(reference.shape[1])
        self.scaler = RobustScaler().fit(reference) if scale else None
        scaled = self.scaler.transform(reference) if self.scaler is not None else reference
        if latent_dim is None:
            self.reducer = None
            latent = scaled
        else:
            components = min(latent_dim, len(reference) - 2, self.input_dim)
            self.reducer = PCA(n_components=components, random_state=42).fit(scaled)
            latent = self.reducer.transform(scaled)
        self.detector = IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        ).fit(latent)
        raw = -self.detector.score_samples(latent)
        self.raw_min = float(raw.min())
        self.raw_max = float(raw.max())

    def score(self, vector: np.ndarray) -> float:
        vector = np.nan_to_num(
            np.asarray(vector, dtype=np.float64).reshape(1, -1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if vector.shape[1] != self.input_dim:
            raise ValueError(
                f"Reference head expected {self.input_dim} values, received {vector.shape[1]}"
            )
        scaled = self.scaler.transform(vector) if self.scaler is not None else vector
        latent = self.reducer.transform(scaled) if self.reducer is not None else scaled
        raw = float(-self.detector.score_samples(latent)[0])
        span = max(self.raw_max - self.raw_min, 1e-12)
        return float(np.clip((raw - self.raw_min) / span, 0.0, 1.0))


_REFERENCE_LCT = REFERENCE_FEATURES[ENGINEERED_COLS].to_numpy(dtype=float)
_DINO_HEAD = _PortableReferenceHead(
    REFERENCE_EMBEDDINGS, latent_dim=None, scale=False
)
_FUSED_HEAD = _PortableReferenceHead(
    np.concatenate([_REFERENCE_LCT, REFERENCE_EMBEDDINGS], axis=1), latent_dim=10
)

_PAIR_LEDGER = pd.read_csv(DINO_DIR / "bilateral_dinov2_reference_scores.csv")
_REFERENCE_BY_ID = {
    plate_id: REFERENCE_EMBEDDINGS[int(row)]
    for plate_id, row in zip(
        REFERENCE_INDEX["plate_id"], REFERENCE_INDEX["embedding_row"]
    )
}
_REFERENCE_PAIRS = np.stack(
    [
        _pair_vector(
            _REFERENCE_BY_ID[row.left_plate_id],
            _REFERENCE_BY_ID[row.right_plate_id],
        )
        for row in _PAIR_LEDGER.itertuples()
    ]
)
_PAIR_HEAD = _PortableReferenceHead(_REFERENCE_PAIRS, latent_dim=6)


class DINOv2Runtime:
    """Lazy, process-cached official DINOv2 encoder for normalized RGB plates."""

    def __init__(
        self,
        model_loader: Callable | None = None,
        repository: str = DINO_HUB_REPOSITORY,
    ):
        self.model = None
        self.transform = None
        self.device = None
        self.repository = repository
        self._model_loader = model_loader
        self._load_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None and self.transform is not None

    def load_official(self, device: str | None = None) -> "DINOv2Runtime":
        if self.ready:
            return self
        with self._load_lock:
            if self.ready:
                return self
            try:
                import torch
                from torchvision import transforms
            except ImportError as exc:
                raise DINOv2UnavailableError(
                    "Live DINOv2 requires requirements-vision.txt"
                ) from exc

            selected_device = device or os.getenv("DINOV2_DEVICE")
            selected_device = selected_device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            loader = self._model_loader or torch.hub.load
            try:
                model = loader(
                    self.repository,
                    BACKBONE_NAME,
                    source="github",
                    pretrained=True,
                    trust_repo=True,
                    force_reload=False,
                )
                model = model.to(selected_device).eval()
            except Exception as exc:
                raise DINOv2UnavailableError(
                    f"Could not load pinned official {BACKBONE_NAME}: {exc}"
                ) from exc

            self.device = selected_device
            self.model = model
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )
        return self

    def encode_rgb(self, rgb: np.ndarray) -> np.ndarray:
        self.load_official()
        try:
            import torch
            from PIL import Image

            image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                embedding = self.model(tensor)
                embedding = torch.nn.functional.normalize(embedding, dim=-1)
            vector = embedding.squeeze(0).float().cpu().numpy()
        except Exception as exc:
            raise DINOv2UnavailableError(f"DINOv2 plate encoding failed: {exc}") from exc

        if vector.shape != (EMBEDDING_DIM,) or not np.isfinite(vector).all():
            raise DINOv2UnavailableError(
                "DINOv2 encoder returned an invalid embedding; expected 384 finite values"
            )
        return vector.astype(np.float32, copy=False)


def _normalized_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.shape != (EMBEDDING_DIM,) or not np.isfinite(vector).all():
        raise ValueError("DINOv2 embedding must contain 384 finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("DINOv2 embedding must have a non-zero norm")
    return vector / norm


def _experimental_native_prediction(vector: np.ndarray, tlc_profile_id: str) -> dict | None:
    if tlc_profile_id != CLIENT_TLC_PROFILE_ID:
        return None
    status = native_classifier_status()
    # This bridge is intentionally version-locked to the current DINO-only
    # 8-vs-1 research artifact. A future fused target model must use the full
    # target-feature builder instead of silently inheriting zero-filled blocks.
    if not status.get("available") or status.get("version") != EXPERIMENTAL_NATIVE_VERSION:
        return status
    schema = feature_schema()
    total = int(schema["total_features"])
    dino_start = int(schema["groups"]["dinov2"]["start"])
    target = np.zeros(total, dtype=np.float32)
    target[dino_start : dino_start + EMBEDDING_DIM] = vector
    return predict_native_binary(target)


def analyze_embedding(
    embedding: np.ndarray,
    lct_features: dict,
    tlc_profile_id: str,
) -> dict:
    vector = _normalized_embedding(embedding)
    reference = REFERENCE_EMBEDDINGS / np.maximum(
        np.linalg.norm(REFERENCE_EMBEDDINGS, axis=1, keepdims=True), 1e-12
    )
    cosine_distances = 1.0 - np.clip(reference @ vector, -1.0, 1.0)
    nearest_rows = np.argsort(cosine_distances)[:5]
    neighbours = [
        {
            "plate_id": str(REFERENCE_INDEX.iloc[int(row)]["plate_id"]),
            "cosine_distance": round(float(cosine_distances[int(row)]), 6),
        }
        for row in nearest_rows
    ]
    lct = _lct_vector(lct_features)
    same_domain = tlc_profile_id == REFERENCE_TLC_PROFILE_ID
    payload = {
        "dinov2_backbone": BACKBONE_NAME,
        "dinov2_embedding_dim": EMBEDDING_DIM,
        "dinov2_reference_anomaly_score_0_1": round(_DINO_HEAD.score(vector), 6),
        "dinov2_nearest_reference_plates": neighbours,
        "dinov2_lct_fused_reference_score_0_1": round(
            _FUSED_HEAD.score(np.concatenate([lct, vector])), 6
        ),
        "dinov2_reference_tlc_profile_id": REFERENCE_TLC_PROFILE_ID,
        "dinov2_reference_domain_match": same_domain,
        "dinov2_domain_notice": (
            "Input and reference TLC profiles match, but reference outputs remain "
            "unlabelled research signals."
            if same_domain
            else "Input TLC profile differs from the publication reference domain; "
            "scores are not formulation-calibrated."
        ),
        "dinov2_score_semantics": SCORE_SEMANTICS,
        "clinical_risk": None,
        "clinical_claim": "NONE",
    }
    native = _experimental_native_prediction(vector, tlc_profile_id)
    if native is not None:
        payload["native_binary_research"] = native
    return payload


def analyze_plate_rgb(rgb: np.ndarray, lct_features: dict, tlc_profile_id: str):
    embedding = runtime.encode_rgb(rgb)
    return analyze_embedding(embedding, lct_features, tlc_profile_id), embedding


def analyze_pair_embeddings(left: np.ndarray, aligned_right: np.ndarray) -> dict:
    left = _normalized_embedding(left)
    aligned_right = _normalized_embedding(aligned_right)
    similarity = float(np.clip(np.dot(left, aligned_right), -1.0, 1.0))
    return {
        "dinov2_cosine_similarity": round(similarity, 6),
        "dinov2_cosine_distance": round(1.0 - similarity, 6),
        "dinov2_pair_reference_anomaly_score_0_1": round(
            _PAIR_HEAD.score(_pair_vector(left, aligned_right)), 6
        ),
        "dinov2_pair_score_semantics": SCORE_SEMANTICS,
        "clinical_risk": None,
        "clinical_claim": "NONE",
    }


runtime = DINOv2Runtime()
