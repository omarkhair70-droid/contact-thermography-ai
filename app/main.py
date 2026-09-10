from pathlib import Path
import json
import uuid
from typing import List

import cv2
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.services.reference_store import store
from app.services.analysis_engine import analyze_uploaded_image
from app.services.exam_contracts import DEFAULT_TLC_PROFILE, normalize_side, normalize_tlc_profile
from app.services.bilateral_live import compare_pair
from app.services.db import save_exam, get_exam, list_exams
from app.services.report import build_report_html
from app.services.json_safety import sanitize_for_json
from app.services.tlc_domain import build_profile_provenance, normalize_device_profile

ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(
    title="Contact Thermography Intelligence Platform",
    version="0.5.0",
    description=(
        "Research analysis platform for contact liquid-crystal thermography. "
        "Current bundled models are reference-only and have no clinical diagnostic claim."
    ),
)

app.mount("/static", StaticFiles(directory=str(ROOT / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))


@app.get("/health")
def health():
    return {
        "status":"ok",
        "service":"lct-intelligence",
        "version":"0.5.0",
        "clinical_claim":"NONE",
        "live_bilateral_analysis":True,
    }


@app.get("/api/model")
def model_info():
    return sanitize_for_json(store.manifest)


@app.get("/api/reference/plates")
def reference_plates():
    return sanitize_for_json({"count":len(store.plates),"items":store.plate_records()})


@app.get("/api/reference/dinov2")
def dinov2_reference():
    return sanitize_for_json({
        "backbone":"dinov2_vits14",
        "embedding_dim":384,
        "clinical_claim":"NONE",
        "plates":store.dino_index.to_dict("records"),
        "bilateral_pairs":store.dino_pairs.to_dict("records"),
    })


@app.get("/api/reference/pairs")
def reference_pairs():
    return sanitize_for_json({"count":len(store.pairs),"items":store.pair_records()})


def parse_metadata(metadata_json: str | None):
    if not metadata_json:
        return {}
    try:
        raw = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid metadata_json: {exc}") from exc

    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict) and "files" in raw:
        items = raw["files"]
    else:
        raise HTTPException(status_code=400, detail="metadata_json must be a list or {'files': [...]}")

    if not isinstance(items, list):
        raise HTTPException(status_code=400, detail="metadata_json files must be a list")

    lookup={}
    for item in items:
        if not isinstance(item, dict) or "filename" not in item:
            continue
        lookup[str(item["filename"])] = {
            "side": normalize_side(item.get("side")),
            "position": str(item.get("position")) if item.get("position") is not None else None,
            "sequence_index": item.get("sequence_index"),
            "tlc_profile_id": normalize_tlc_profile(item.get("tlc_profile_id")),
            "device_profile_id": normalize_device_profile(item.get("device_profile_id")),
        }
    return lookup


def _default_file_metadata():
    return {
        "side":"UNKNOWN",
        "position":None,
        "sequence_index":None,
        "tlc_profile_id":DEFAULT_TLC_PROFILE,
        "device_profile_id":None,
    }


@app.post("/api/exams/analyze")
async def analyze_exam(
    files: List[UploadFile] = File(...),
    exam_id: str | None = Form(default=None),
    metadata_json: str | None = Form(default=None),
):
    eid=(exam_id or f"exam-{uuid.uuid4().hex[:10]}").replace("/","_").replace("\\","_")
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required")

    metadata = parse_metadata(metadata_json)
    resolved=[]
    for upload in files:
        name=upload.filename or ""
        resolved.append((upload, metadata.get(name, _default_file_metadata())))

    tlc_profiles=sorted({item["tlc_profile_id"] for _,item in resolved})
    if len(tlc_profiles) != 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "Mixed TLC profiles are not allowed in one examination because colour domains are not "
                f"interchangeable. Submit separate exams per tlc_profile_id: {', '.join(tlc_profiles)}"
            ),
        )
    tlc_profile_id=tlc_profiles[0]
    device_profile_ids=sorted({item["device_profile_id"] for _,item in resolved if item["device_profile_id"]})
    profile_provenance=build_profile_provenance(tlc_profile_id,device_profile_ids)

    sources=[]
    total=0
    flat_plates=[]

    for upload,m in resolved:
        raw=await upload.read()
        if not raw:
            raise HTTPException(status_code=400, detail=f"Empty file: {upload.filename}")
        try:
            result=analyze_uploaded_image(raw,upload.filename or "image",eid)
        except ValueError as exc:
            raise HTTPException(status_code=400,detail=f"{upload.filename}: {exc}") from exc

        result["tlc_profile_id"]=m["tlc_profile_id"]
        result["device_profile_id"]=m["device_profile_id"]
        for idx,p in enumerate(result["plates"], start=1):
            p["side"]=m["side"]
            p["position"]=m["position"] if m["position"] is not None else f"AUTO-{idx:02d}"
            p["sequence_index"]=m["sequence_index"] if m["sequence_index"] is not None else idx
            p["tlc_profile_id"]=m["tlc_profile_id"]
            p["device_profile_id"]=m["device_profile_id"]
            flat_plates.append(p)

        sources.append(result)
        total += result["plates_detected"]

    left = {}
    right = {}
    for p in flat_plates:
        key=(p["position"],p.get("device_profile_id"))
        if p["side"]=="LEFT":
            left.setdefault(key, []).append(p)
        elif p["side"]=="RIGHT":
            right.setdefault(key, []).append(p)

    bilateral=[]
    pair_dir = ROOT/"app"/"static"/"generated"/eid/"bilateral"
    pair_index=0
    for position,device_profile_id in sorted(set(left).intersection(right), key=lambda k:(str(k[0]),str(k[1] or ""))):
        L=sorted(left[(position,device_profile_id)], key=lambda x:(x.get("sequence_index") or 0, x["plate_id"]))
        R=sorted(right[(position,device_profile_id)], key=lambda x:(x.get("sequence_index") or 0, x["plate_id"]))
        for lp,rp in zip(L,R):
            pair_index += 1
            pair_id=f"{eid}-BP-{pair_index:03d}"
            lbgr=cv2.imread(lp["_generated_image_path"])
            rbgr=cv2.imread(rp["_generated_image_path"])
            if lbgr is None or rbgr is None:
                raise HTTPException(status_code=500, detail="Generated plate image missing during bilateral pairing")
            metrics=compare_pair(lbgr,rbgr,pair_dir,pair_id)
            metrics.update({
                "bilateral_pair_id":pair_id,
                "position":position,
                "left_plate_id":lp["plate_id"],
                "right_plate_id":rp["plate_id"],
                "tlc_profile_id":tlc_profile_id,
                "device_profile_id":device_profile_id,
                "panel_url":f"/static/generated/{eid}/bilateral/{metrics.pop('panel_filename')}",
                "difference_url":f"/static/generated/{eid}/bilateral/{metrics.pop('difference_filename')}",
                "right_aligned_url":f"/static/generated/{eid}/bilateral/{metrics.pop('right_aligned_filename')}",
            })
            bilateral.append(metrics)

    pairing_warnings=[]
    left_by_position={p["position"] for p in flat_plates if p["side"]=="LEFT"}
    right_by_position={p["position"] for p in flat_plates if p["side"]=="RIGHT"}
    paired_positions={item["position"] for item in bilateral}
    for position in sorted((left_by_position & right_by_position)-paired_positions, key=str):
        pairing_warnings.append({
            "position":position,
            "reason":"DEVICE_PROFILE_MISMATCH_OR_UNPAIRED_SEQUENCE",
            "message":"LEFT/RIGHT plates were not paired across incompatible or unmatched device provenance.",
        })

    for p in flat_plates:
        p.pop("_generated_image_path",None)

    result = {
        "exam_id":eid,
        "analysis_type":"contact_liquid_crystal_thermography",
        "tlc_profile_id":tlc_profile_id,
        "profile_provenance":profile_provenance,
        "source_images":len(sources),
        "plates_detected":total,
        "metadata_items_supplied":len(metadata),
        "bilateral_pairs_created":len(bilateral),
        "bilateral_pairing_warnings":pairing_warnings,
        "sources":sources,
        "bilateral_analysis":bilateral,
        "clinical_risk":None,
        "clinical_claim":"NONE",
        "model_status":"RESEARCH_REFERENCE_ANALYSIS",
    }
    result=sanitize_for_json(result)
    result["saved_at"] = save_exam(result)
    result["report_url"] = f"/reports/{eid}"
    return result


@app.get("/api/exams")
def exam_history(limit: int=50):
    items=list_exams(limit)
    return sanitize_for_json({"count":len(items),"items":items})


@app.get("/api/exams/{exam_id}")
def exam_detail(exam_id: str):
    item=get_exam(exam_id)
    if item is None:
        raise HTTPException(status_code=404,detail="Exam not found")
    return sanitize_for_json(item)


@app.get("/reports/{exam_id}", response_class=HTMLResponse)
def exam_report(exam_id: str):
    item=get_exam(exam_id)
    if item is None:
        raise HTTPException(status_code=404,detail="Exam not found")
    return HTMLResponse(build_report_html(item["result"]))


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "plates":store.plate_records(),
            "pairs":store.pair_records(),
            "manifest":store.manifest,
        },
    )
