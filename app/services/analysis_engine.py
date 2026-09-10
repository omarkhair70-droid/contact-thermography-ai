from pathlib import Path
import math
import uuid

import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from app.services.qc import assess_plate_quality
from app.services.reference_model import reference_model

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
STATIC_GENERATED = ROOT / "app" / "static" / "generated"
STATIC_GENERATED.mkdir(parents=True, exist_ok=True)

def _circle_iou(c1, c2):
    x1,y1,r1 = c1; x2,y2,r2 = c2
    d = math.hypot(x1-x2, y1-y2)
    if d >= r1+r2:
        return 0.0
    if d <= abs(r1-r2):
        inter = math.pi * min(r1,r2)**2
    else:
        a1 = math.acos(np.clip((d*d+r1*r1-r2*r2)/(2*d*r1), -1, 1))
        a2 = math.acos(np.clip((d*d+r2*r2-r1*r1)/(2*d*r2), -1, 1))
        inter = (
            r1*r1*a1 + r2*r2*a2
            - 0.5*math.sqrt(max(0,(-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2)))
        )
    union = math.pi*r1*r1 + math.pi*r2*r2 - inter
    return inter/max(union,1e-9)

def _circle_score(img, c):
    x,y,r = c
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h,w = gray.shape
    yy,xx = np.ogrid[:h,:w]
    d2 = (xx-x)**2 + (yy-y)**2
    inside = d2 <= (0.90*r)**2
    ring = (d2 >= (1.02*r)**2) & (d2 <= (1.22*r)**2)
    if inside.sum() < 20 or ring.sum() < 20:
        return None
    mi = float(gray[inside].mean())
    mr = float(gray[ring].mean())
    dark = float((gray[inside] < 160).mean())
    return mi, mr, mr-mi, dark

def detect_circular_plates(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 1.5)
    h,w = gray.shape
    minR = max(15, int(min(h,w)*0.06))
    maxR = max(minR+2, int(min(h,w)*0.48))
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=minR,
        param1=100, param2=30, minRadius=minR, maxRadius=maxR
    )
    candidates=[]
    if circles is not None:
        for raw in np.round(circles[0]).astype(int):
            c=raw.tolist()
            s=_circle_score(img,c)
            if s is None:
                continue
            mi,mr,contrast,dark=s
            if contrast > 100 and dark > 0.75 and mr > 170:
                candidates.append((c,s))
    candidates.sort(key=lambda z:z[1][2], reverse=True)
    keep=[]
    for c,s in candidates:
        if all(_circle_iou(c,k[0]) < 0.35 for k in keep):
            keep.append((c,s))
    keep.sort(key=lambda z:(z[0][1],z[0][0]))
    return keep

def crop_circle(img, x, y, r, out_size=256):
    pad=int(round(r*1.08))
    h,w=img.shape[:2]
    x0,x1=max(0,x-pad),min(w,x+pad)
    y0,y1=max(0,y-pad),min(h,y+pad)
    crop=img[y0:y1,x0:x1].copy()
    ch,cw=crop.shape[:2]
    side=max(ch,cw)
    canvas=np.zeros((side,side,3),dtype=np.uint8)
    oy=(side-ch)//2; ox=(side-cw)//2
    canvas[oy:oy+ch,ox:ox+cw]=crop
    canvas=cv2.resize(canvas,(out_size,out_size),interpolation=cv2.INTER_AREA)
    return canvas

def whole_image_plate(img, out_size=256):
    h,w=img.shape[:2]
    side=max(h,w)
    canvas=np.zeros((side,side,3),dtype=np.uint8)
    oy=(side-h)//2; ox=(side-w)//2
    canvas[oy:oy+h,ox:ox+w]=img
    return cv2.resize(canvas,(out_size,out_size),interpolation=cv2.INTER_AREA)

def _branch_end_points(skel):
    s=skel.astype(np.uint8)
    padded=np.pad(s,1)
    n=np.zeros_like(s,dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            if dx==1 and dy==1:
                continue
            n += padded[dy:dy+s.shape[0], dx:dx+s.shape[1]]
    return int(((s==1)&(n>=3)).sum()), int(((s==1)&(n==1)).sum())

def signal_features(bgr):
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    labc=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB)
    h,w=hsv.shape[:2]
    yy,xx=np.ogrid[:h,:w]
    disk=((xx-w/2)**2+(yy-h/2)**2 <= (min(h,w)*0.46)**2)

    active=(disk & (hsv[:,:,1]>50) & (hsv[:,:,2]>38)).astype(np.uint8)*255
    kernel=np.ones((3,3),np.uint8)
    active=cv2.morphologyEx(active,cv2.MORPH_OPEN,kernel)
    active=cv2.morphologyEx(active,cv2.MORPH_CLOSE,kernel)

    nlab,labs,stats,_=cv2.connectedComponentsWithStats(active,8)
    clean=np.zeros_like(active)
    for i in range(1,nlab):
        if stats[i,cv2.CC_STAT_AREA] >= 18:
            clean[labs==i]=255

    labeled=label(clean>0)
    props=regionprops(labeled)
    total_area=int((clean>0).sum())
    disk_area=int(disk.sum())
    area_frac=total_area/max(disk_area,1)
    largest=max(props,key=lambda r:r.area) if props else None

    largest_area_frac=(largest.area/disk_area) if largest else 0.0
    eccentricity=float(largest.eccentricity) if largest else 0.0
    solidity=float(largest.solidity) if largest else 0.0
    perimeter=float(largest.perimeter) if largest else 0.0
    major=float(getattr(largest,"axis_major_length",0.0)) if largest else 0.0
    minor=float(getattr(largest,"axis_minor_length",0.0)) if largest else 0.0
    elongation=major/max(minor,1e-6) if largest else 0.0

    skel=skeletonize(clean>0)
    branches,endpoints=_branch_end_points(skel)
    skel_len=int(skel.sum())

    if total_area:
        ys,xs=np.where(clean>0)
        cx=float(xs.mean()/w); cy=float(ys.mean()/h)
    else:
        cx=cy=0.0

    def mean_std(arr):
        vals=arr[clean>0]
        if len(vals)==0:
            return 0.0,0.0
        return float(vals.mean()),float(vals.std())

    hue_mean,hue_std=mean_std(hsv[:,:,0])
    sat_mean,_=mean_std(hsv[:,:,1])
    val_mean,_=mean_std(hsv[:,:,2])
    laba_mean,_=mean_std(labc[:,:,1])
    labb_mean,_=mean_std(labc[:,:,2])

    if area_frac < 0.02:
        morph="minimal-response"
    elif len(props) >= 5 and largest_area_frac < 0.12:
        morph="scattered"
    elif elongation >= 3.0 and area_frac < 0.22:
        morph="linear-like"
    elif branches >= 8 and elongation >= 1.6:
        morph="branched/complex"
    elif area_frac >= 0.30:
        morph="diffuse/large-region"
    elif largest_area_frac >= 0.08:
        morph="focal/irregular-region"
    else:
        morph="mixed"

    f={
        "response_area_fraction":area_frac,
        "component_count":len(props),
        "largest_component_fraction":largest_area_frac,
        "largest_eccentricity":eccentricity,
        "largest_solidity":solidity,
        "largest_perimeter_px":perimeter,
        "largest_elongation":elongation,
        "skeleton_length_px":skel_len,
        "branch_pixels":branches,
        "endpoint_pixels":endpoints,
        "response_centroid_x_norm":cx,
        "response_centroid_y_norm":cy,
        "hue_mean":hue_mean,
        "hue_std":hue_std,
        "saturation_mean":sat_mean,
        "value_mean":val_mean,
        "lab_a_mean":laba_mean,
        "lab_b_mean":labb_mean,
    }
    return f,morph,clean

def analyze_plate(bgr, plate_id):
    f,morph,mask=signal_features(bgr)
    ref = reference_model.analyze(f)
    percentile=ref["reference_anomaly_percentile"]
    neighbors=ref["nearest_reference_plates"]

    qc = assess_plate_quality(bgr, float(f["response_area_fraction"]))

    return {
        "plate_id":plate_id,
        "qc":qc,
        "morphology_descriptor":morph,
        "signal_features":{k:round(float(v),6) if isinstance(v,(float,np.floating)) else int(v) for k,v in f.items()},
        "reference_anomaly_percentile":round(percentile,6),
        "reference_anomaly_semantics":"relative unusualness vs 26 supplied reference plates; not a probability",
        "nearest_reference_plates":neighbors,
        "clinical_risk":None,
        "clinical_claim":"NONE",
        "_mask":mask,
    }

def analyze_uploaded_image(file_bytes, source_name, exam_id):
    arr=np.frombuffer(file_bytes,dtype=np.uint8)
    img=cv2.imdecode(arr,cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unsupported or corrupt image")

    dets=detect_circular_plates(img)
    plates=[]
    mode="detected_circles"
    if not dets:
        dets=[([img.shape[1]//2,img.shape[0]//2,min(img.shape[:2])//2],None)]
        mode="whole_image_fallback"

    exam_dir=STATIC_GENERATED/exam_id
    exam_dir.mkdir(parents=True,exist_ok=True)

    for idx,(c,_) in enumerate(dets,1):
        if mode=="whole_image_fallback":
            crop=whole_image_plate(img)
        else:
            x,y,r=c
            crop=crop_circle(img,x,y,r)
        pid=f"{Path(source_name).stem[:24]}-P{idx:02d}".replace(" ","_")
        result=analyze_plate(crop,pid)
        mask=result.pop("_mask")

        img_name=f"{uuid.uuid4().hex[:10]}_{pid}.png"
        mask_name=f"{uuid.uuid4().hex[:10]}_{pid}_mask.png"
        cv2.imwrite(str(exam_dir/img_name),crop)
        cv2.imwrite(str(exam_dir/mask_name),mask)

        result["image_url"]=f"/static/generated/{exam_id}/{img_name}"
        result["response_mask_url"]=f"/static/generated/{exam_id}/{mask_name}"
        result["source_image"]=source_name
        result["_generated_image_path"]=str(exam_dir/img_name)
        plates.append(result)

    return {
        "source_image":source_name,
        "extraction_mode":mode,
        "plates_detected":len(plates),
        "plates":plates,
    }
