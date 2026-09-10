from __future__ import annotations
import cv2
import numpy as np

def assess_plate_quality(bgr: np.ndarray, response_area_fraction: float | None=None):
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)

    h,w=gray.shape
    yy,xx=np.ogrid[:h,:w]
    disk=((xx-w/2)**2+(yy-h/2)**2 <= (min(h,w)*0.46)**2)

    g=gray[disk]
    s=hsv[:,:,1][disk]
    v=hsv[:,:,2][disk]

    lap_var=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    dark_clip=float((g <= 3).mean())
    bright_clip=float((g >= 252).mean())
    saturation_clip=float((s >= 252).mean())
    low_contrast=float(np.percentile(g,95)-np.percentile(g,5))

    flags=[]
    if lap_var < 20:
        flags.append("LOW_SHARPNESS")
    if bright_clip > 0.08:
        flags.append("BRIGHT_CLIPPING")
    if dark_clip > 0.85:
        flags.append("VERY_DARK_PLATE")
    if saturation_clip > 0.20:
        flags.append("SATURATION_CLIPPING")
    if low_contrast < 12:
        flags.append("LOW_CONTRAST")
    if response_area_fraction is not None:
        if response_area_fraction < 0.005:
            flags.append("VERY_LOW_THERMOCHROMIC_RESPONSE")
        if response_area_fraction > 0.85:
            flags.append("POSSIBLE_OVER_RESPONSE")

    severe={"BRIGHT_CLIPPING","VERY_DARK_PLATE","LOW_CONTRAST"}
    severe_count=sum(1 for f in flags if f in severe)
    if severe_count >= 2:
        status="REJECT_REFERENCE_QC"
    elif flags:
        status="REVIEW"
    else:
        status="PASS_REFERENCE_QC"

    return {
        "status":status,
        "flags":flags,
        "metrics":{
            "laplacian_variance":round(lap_var,4),
            "dark_clip_fraction":round(dark_clip,6),
            "bright_clip_fraction":round(bright_clip,6),
            "saturation_clip_fraction":round(saturation_clip,6),
            "gray_dynamic_range_p95_p5":round(low_contrast,4),
        },
        "semantics":"engineering image-quality gate; not a clinical assessment"
    }
