"""Uncalibrated Contact-LCT local evidence. No disease decision or temperature output.

Run from the repository root: python research/mumguard/local_contrast.py --help
Masks must be in original image coordinates and describe contact, not TLC response.
This independent research implementation does not modify the application's registry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

VERSION = "mumguard-local-contrast-v0.1-prototype"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hist(hue, sat, mask):
    # Hue is circular. Weight by saturation; no hue-to-temperature ordering assumed.
    position = (hue[mask] * 12) % 12
    lower = np.floor(position).astype(int)
    fraction = position - lower
    counts = np.bincount(lower, weights=sat[mask]*(1-fraction), minlength=12).astype(float)
    counts += np.bincount((lower+1) % 12, weights=sat[mask]*fraction, minlength=12)
    counts = .5*counts + .25*np.roll(counts, 1) + .25*np.roll(counts, -1)
    return counts / max(counts.sum(), 1e-12)


def _js(p, q):
    m = (p + q) / 2
    return float(0.5 * sum(np.sum(x * np.log((x + 1e-12) / (m + 1e-12))) for x in (p, q)))


def _box_stats(h, s, v, mask):
    hist = _hist(h, s, mask)
    return hist, float(np.median(s[mask])), float(np.median(v[mask]))


def analyze(rgb, contact, exclusion=None, *, tokens=None, scales=(16, 32, 64),
            minimum_coverage=0.8, minimum_pixels=32):
    """Return two-sided photometric contrasts against a disjoint surrounding ring.

    Inputs are an already oriented/resized RGB image and aligned binary masks.
    Missing observable color is missing evidence, never a healthy-tissue label.
    Optional tokens are a finite Ht x Wt x 384 grid aligned to the complete image.
    Scores are dimensionless engineering evidence, with no learned threshold.
    """
    rgb = np.asarray(rgb)
    contact = np.asarray(contact, bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be H x W x 3 uint8")
    if contact.shape != rgb.shape[:2]:
        raise ValueError("contact mask shape mismatch")
    if exclusion is None:
        exclusion = np.zeros(contact.shape, bool)
    exclusion = np.asarray(exclusion, bool)
    if exclusion.shape != contact.shape:
        raise ValueError("exclusion mask shape mismatch")
    if not 0 < minimum_coverage <= 1 or minimum_pixels < 1:
        raise ValueError("invalid coverage/pixel requirement")
    if not scales or any(int(a) != a or a < 4 or a % 2 for a in scales):
        raise ValueError("scales must be positive even integers >= 4")
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=float) / 255.0
    h, s, v = (hsv[..., i] for i in range(3))
    glare = (v >= 245 / 255) & (s <= 30 / 255)
    valid = contact & ~exclusion & ~glare
    # Provisional acquisition support test; not a thermometric calibration.
    chromatic = valid & (s > 55 / 255) & (v > 85 / 255)
    reasons = []
    if contact.sum() < minimum_pixels:
        reasons.append("INSUFFICIENT_CONTACT_FIELD")
    if np.any(contact) and np.mean(glare[contact]) > 0.04:
        reasons.append("GLARE_REVIEW")
    if np.any(valid) and np.mean(chromatic[valid]) < 0.2:
        reasons.append("LOW_CHROMATIC_SUPPORT")
    token_xy = token_z = None
    if tokens is not None:
        token_z = np.asarray(tokens, float)
        if token_z.ndim != 3 or token_z.shape[2] != 384 or not np.isfinite(token_z).all():
            raise ValueError("tokens must be finite Ht x Wt x 384")
        th, tw = token_z.shape[:2]
        yy, xx = np.meshgrid((np.arange(th) + 0.5) * rgb.shape[0] / th,
                             (np.arange(tw) + 0.5) * rgb.shape[1] / tw, indexing="ij")
        token_xy = (yy.astype(int), xx.astype(int))
        token_z = token_z.reshape(-1, 384)
        norms = np.linalg.norm(token_z, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("zero norm DINO token")
        token_z /= norms
    hh, ww = valid.shape
    records = []
    rejected = 0
    heat = np.zeros((hh, ww), np.float32)
    support = np.zeros((hh, ww), np.uint16)
    for size in scales:
        half = size // 2
        # Outer window side = 3*size. A 0.5*size gap excludes immediate neighbours.
        for cy in range(3 * half, hh - 3 * half + 1, half):
            for cx in range(3 * half, ww - 3 * half + 1, half):
                center = np.zeros((hh, ww), bool)
                center[cy-half:cy+half, cx-half:cx+half] = True
                ring = np.zeros_like(center)
                ring[cy-3*half:cy+3*half, cx-3*half:cx+3*half] = True
                ring[cy-size:cy+size, cx-size:cx+size] = False
                if np.mean(valid[center]) < minimum_coverage or np.mean(valid[ring]) < minimum_coverage:
                    rejected += 1
                    continue
                cm, rm = center & chromatic, ring & chromatic
                if cm.sum() < minimum_pixels or rm.sum() < minimum_pixels:
                    rejected += 1
                    continue
                p, cs, cv = _box_stats(h, s, v, cm)
                q, rs, rv = _box_stats(h, s, v, rm)
                js = _js(p, q)
                # Bounded scale-free blocks; this provisional fusion is not trained.
                hue_distance = js / np.log(2)
                sat_distance = abs(cs - rs)
                value_distance = abs(cv - rv)
                physical = (hue_distance + sat_distance + value_distance) / 3
                dz = None
                if token_z is not None:
                    # Entire patch support, not only its center, must lie in each
                    # disjoint observable region. No boundary/glare token leakage.
                    def patch_support(mask):
                        return np.array([mask[round(ty*hh/th):round((ty+1)*hh/th),
                                              round(tx*ww/tw):round((tx+1)*ww/tw)].mean() >= minimum_coverage
                                         for ty in range(th) for tx in range(tw)])
                    cidx = patch_support(cm)
                    ridx = patch_support(rm)
                    if cidx.sum() >= 2 and ridx.sum() >= 3:
                        similarities = token_z[cidx] @ token_z[ridx].T
                        # Median of nearest three local comparator distances.
                        distances = 1 - np.clip(similarities, -1, 1)
                        dz = float(np.median(np.sort(distances, axis=1)[:, :3]))
                record = {"x": cx, "y": cy, "size": size,
                          "center_pixels": int(cm.sum()), "ring_pixels": int(rm.sum()),
                          "hue_js": js, "saturation_signed_delta": cs-rs,
                          "value_signed_delta": cv-rv, "physical_contrast": float(physical),
                          "dino_local_cosine_distance": dz}
                records.append(record)
                heat[center] += physical
                support[center] += 1
    heat = np.divide(heat, support, out=np.zeros_like(heat), where=support > 0)
    if not records:
        reasons.append("NO_USABLE_LOCAL_COMPARATORS")
    if records:
        # One summary per scale, then median: large numbers of windows do not create votes.
        scale_scores = {str(a): float(np.quantile([r["physical_contrast"] for r in records if r["size"] == a], .9))
                        for a in scales if any(r["size"] == a for r in records)}
        score = float(np.median(list(scale_scores.values())))
    else:
        scale_scores, score = {}, None
    return {"version": VERSION, "clinical_claim": "NONE", "research_binary_class": None,
            "decision_status": "ABSTAIN_UNCALIBRATED", "validation_status": "ENGINEERING_ONLY",
            "absolute_temperature": None, "semantics": "local photometric contrast; not tumor probability or size",
            "local_contrast_score": score, "scale_scores": scale_scores,
            "quality_reasons": reasons, "candidate_count": len(records), "rejected_windows": rejected,
            "analyzed_contact_fraction": float(np.mean(support[contact] > 0)) if contact.any() else 0.,
            "chromatic_fraction_of_contact": float(chromatic.sum() / max(contact.sum(), 1)),
            "candidates": records}, heat, support


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--contact-mask", type=Path, required=True)
    p.add_argument("--exclude-mask", type=Path)
    p.add_argument("--subject-id", required=True)
    p.add_argument("--species", choices=["mouse", "human"], required=True)
    p.add_argument("--tlc-profile-id", required=True)
    p.add_argument("--device-profile-id", required=True)
    p.add_argument("--acquisition-id", required=True)
    p.add_argument("--tokens-npz", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", type=int, default=224)
    args = p.parse_args()
    if args.size < 192 or args.size % 14:
        p.error("size must be a multiple of 14 >= 192")
    # Annotation contract: masks already correspond to EXIF-oriented source pixels.
    source = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    def mask(path):
        m = Image.open(path).convert("L")
        if m.size != source.size:
            raise ValueError("masks must match EXIF-oriented source dimensions")
        return m
    cm = mask(args.contact_mask)
    em = mask(args.exclude_mask) if args.exclude_mask else None
    width, height = source.size
    factor = args.size / max(width, height)
    sz = (max(1, round(width*factor)), max(1, round(height*factor)))
    xy = ((args.size-sz[0])//2, (args.size-sz[1])//2)
    def resize_canvas(im, mode, resample):
        canvas = Image.new(mode, (args.size, args.size))
        canvas.paste(im.resize(sz, resample), xy)
        return np.asarray(canvas)
    rgb = resize_canvas(source, "RGB", Image.Resampling.BILINEAR)
    contact = resize_canvas(cm, "L", Image.Resampling.NEAREST) > 127
    exclusion = resize_canvas(em, "L", Image.Resampling.NEAREST) > 127 if em else None
    tokens = None
    if args.tokens_npz:
        with np.load(args.tokens_npz, allow_pickle=False) as saved:
            if str(saved["backbone"].item()) != "dinov2_vits14" or str(saved["revision"].item()) != "7764ea0f912e53c92e82eb78a2a1631e92725fc8":
                raise ValueError("DINO token provenance mismatch")
            if str(saved["input_sha256"].item()) != hashlib.sha256(rgb.tobytes()).hexdigest():
                raise ValueError("DINO tokens do not match normalized RGB hash")
            tokens = saved["tokens"]
    result, heat, support = analyze(rgb, contact, exclusion, tokens=tokens)
    result["provenance"] = {"subject_id": args.subject_id, "species": args.species,
        "tlc_profile_id": args.tlc_profile_id, "device_profile_id": args.device_profile_id,
        "acquisition_id": args.acquisition_id, "source_sha256": sha256(args.image),
        "contact_mask_sha256": sha256(args.contact_mask),
        "exclude_mask_sha256": sha256(args.exclude_mask) if args.exclude_mask else None,
        "normalized_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "tokens_sha256": sha256(args.tokens_npz) if args.tokens_npz else None,
        "source_dimensions": list(source.size), "normalized_dimensions": list(rgb.shape[:2]),
        "resize_dimensions": list(sz), "padding_xy": list(xy)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "evidence.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(args.out / "maps.npz", local_contrast=heat, support=support)
    Image.fromarray(rgb).save(args.out / "normalized.png")
    Image.fromarray((contact*255).astype(np.uint8)).save(args.out / "contact.png")
    Image.fromarray((np.clip(heat, 0, 1)*255).astype(np.uint8)).save(args.out / "local_contrast.png")
    print(json.dumps({k:v for k,v in result.items() if k != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
