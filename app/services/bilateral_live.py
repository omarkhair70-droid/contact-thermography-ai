from __future__ import annotations
from pathlib import Path
import math, uuid
import cv2
import numpy as np
from app.services.tlc_profiles import TLCProfile

def _phase_register(reference_bgr: np.ndarray, moving_bgr: np.ndarray):
    """
    Translation-only registration using phase correlation.
    Circular plates are already normalized by the ingestion stage.
    """
    ref = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mov = cv2.cvtColor(moving_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref = cv2.GaussianBlur(ref, (5,5), 0)
    mov = cv2.GaussianBlur(mov, (5,5), 0)
    shift, response = cv2.phaseCorrelate(ref, mov)
    dx, dy = shift
    M = np.float32([[1,0,dx],[0,1,dy]])
    aligned = cv2.warpAffine(moving_bgr, M, (moving_bgr.shape[1], moving_bgr.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return aligned, {"dx": float(dx), "dy": float(dy), "response": float(response)}

def _signal_mask(bgr: np.ndarray, profile: TLCProfile):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h,w = hsv.shape[:2]
    yy,xx = np.ogrid[:h,:w]
    disk = ((xx-w/2)**2+(yy-h/2)**2 <= (min(h,w)*0.46)**2)
    active = (
        disk
        & (hsv[:,:,1] > profile.saturation_min)
        & (hsv[:,:,2] > profile.value_min)
    ).astype(np.uint8)
    k = np.ones((3,3), np.uint8)
    active = cv2.morphologyEx(active, cv2.MORPH_OPEN, k)
    active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, k)
    return active.astype(bool), disk, hsv

def compare_pair(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    out_dir: Path,
    pair_id: str,
    profile: TLCProfile,
):
    aligned_right, reg = _phase_register(left_bgr, right_bgr)

    lmask, disk, lhsv = _signal_mask(left_bgr, profile)
    rmask, _, rhsv = _signal_mask(aligned_right, profile)

    valid = disk
    union = (lmask | rmask) & valid
    inter = (lmask & rmask) & valid
    xor = (lmask ^ rmask) & valid

    l_area = float((lmask & valid).sum()/max(valid.sum(),1))
    r_area = float((rmask & valid).sum()/max(valid.sum(),1))
    jaccard = float(inter.sum()/max(union.sum(),1))
    xor_fraction = float(xor.sum()/max(valid.sum(),1))

    dh = np.abs(lhsv[:,:,0].astype(np.int16) - rhsv[:,:,0].astype(np.int16))
    dh = np.minimum(dh, 180-dh)
    common = inter
    mean_hue_distance = float(dh[common].mean()) if common.sum() else None
    p90_hue_distance = float(np.percentile(dh[common],90)) if common.sum() else None

    hue_component = np.zeros_like(dh, dtype=np.float32)
    hue_component[common] = dh[common].astype(np.float32)/90.0
    structural_component = xor.astype(np.float32)
    diff = np.clip(0.65*hue_component + 0.35*structural_component, 0, 1)
    diff[~valid] = 0

    heat = cv2.applyColorMap((diff*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat[~valid] = 0

    xor_vis = np.zeros_like(left_bgr)
    xor_vis[xor] = (255,255,255)

    out_dir.mkdir(parents=True, exist_ok=True)
    right_name = f"{pair_id}_right_aligned.png"
    diff_name = f"{pair_id}_difference.png"
    panel_name = f"{pair_id}_panel.png"
    cv2.imwrite(str(out_dir/right_name), aligned_right)
    cv2.imwrite(str(out_dir/diff_name), heat)

    panel = np.concatenate([left_bgr, aligned_right, xor_vis, heat], axis=1)
    cv2.putText(panel, "LEFT", (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(panel, "RIGHT ALIGNED", (264,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(panel, "STRUCTURAL XOR", (520,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(panel, "FUSED DIFFERENCE", (776,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir/panel_name), panel)

    hue_norm = (mean_hue_distance/90.0) if mean_hue_distance is not None else 0.0
    reference_asymmetry_score = float(np.clip(
        0.45*xor_fraction + 0.25*abs(l_area-r_area) + 0.20*(1-jaccard) + 0.10*hue_norm,
        0, 1
    ))

    return {
        "registration": reg,
        "left_response_area_fraction": round(l_area,6),
        "right_response_area_fraction": round(r_area,6),
        "absolute_area_fraction_delta": round(abs(l_area-r_area),6),
        "response_jaccard_similarity": round(jaccard,6),
        "response_xor_fraction": round(xor_fraction,6),
        "mean_common_response_hue_distance": round(mean_hue_distance,6) if mean_hue_distance is not None else None,
        "p90_common_response_hue_distance": round(p90_hue_distance,6) if p90_hue_distance is not None else None,
        "reference_asymmetry_score_0_1": round(reference_asymmetry_score,6),
        "score_semantics": "reference bilateral asymmetry only; not cancer probability",
        "tlc_profile_id": profile.id,
        "right_aligned_filename": right_name,
        "difference_filename": diff_name,
        "panel_filename": panel_name,
        "clinical_risk": None,
        "clinical_claim": "NONE",
    }
