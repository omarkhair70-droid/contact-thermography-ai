from pathlib import Path
import json
import uuid
from typing import List

import cv2
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.services.reference_store import store
from app.services.analysis_engine import analyze_uploaded_image
from app.services.exam_contracts import normalize_side
from app.services.bilateral_live import compare_pair
from app.services.db import save_exam, get_exam, list_exams
from app.services.report import build_report_html

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
    return store.manifest

@app.get("/api/reference/plates")
def reference_plates():
    return {"count":len(store.plates),"items":store.plate_records()}

@app.get("/api/reference/dinov2")
def dinov2_reference():
    def clean_df(df):
        records=[]
        for raw in df.to_dict("records"):
            record={}
            for k,v in raw.items():
                if pd.isna(v):
                    record[k]=None
                elif hasattr(v, "item"):
                    record[k]=v.item()
                else:
                    record[k]=v
            records.append(record)
        return records

    return {
        "backbone":"dinov2_vits14",
        "embedding_dim":384,
        "clinical_claim":"NONE",
        "plates":clean_df(store.dino_index),
        "bilateral_pairs":clean_df(store.dino_pairs),
    }

@app.get("/api/reference/pairs")
def reference_pairs():
    return {"count":len(store.pairs),"items":store.pair_records()}

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

    lookup={}
    for item in items:
        if not isinstance(item, dict) or "filename" not in item:
            continue
        lookup[item["filename"]] = {
            "side": normalize_side(item.get("side")),
            "position": str(item.get("position")) if item.get("position") is not None else None,
            "sequence_index": item.get("sequence_index"),
        }
    return lookup

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
    sources=[]
    total=0
    flat_plates=[]

    for upload in files:
        raw=await upload.read()
        if not raw:
            raise HTTPException(status_code=400, detail=f"Empty file: {upload.filename}")
        try:
            result=analyze_uploaded_image(raw,upload.filename or "image",eid)
        except ValueError as exc:
            raise HTTPException(status_code=400,detail=f"{upload.filename}: {exc}") from exc

        m=metadata.get(upload.filename or "", {"side":"UNKNOWN","position":None,"sequence_index":None})
        for idx,p in enumerate(result["plates"], start=1):
            p["side"]=m["side"]
            p["position"]=m["position"] if m["position"] is not None else f"AUTO-{idx:02d}"
            p["sequence_index"]=m["sequence_index"] if m["sequence_index"] is not None else idx
            flat_plates.append(p)

        sources.append(result)
        total += result["plates_detected"]

    left = {}
    right = {}
    for p in flat_plates:
        if p["side"]=="LEFT":
            left.setdefault(p["position"], []).append(p)
        elif p["side"]=="RIGHT":
            right.setdefault(p["position"], []).append(p)

    bilateral=[]
    pair_dir = ROOT/"app"/"static"/"generated"/eid/"bilateral"
    pair_index=0
    for position in sorted(set(left).intersection(right)):
        L=sorted(left[position], key=lambda x:(x.get("sequence_index") or 0, x["plate_id"]))
        R=sorted(right[position], key=lambda x:(x.get("sequence_index") or 0, x["plate_id"]))
        for lp,rp in zip(L,R):
            pair_index += 1
            pair_id=f"{eid}-BP-{pair_index:03d}"
            lbgr=cv2.imread(lp["_generated_image_path"])
            rbgr=cv2.imread(rp["_generated_image_path"])
            metrics=compare_pair(lbgr,rbgr,pair_dir,pair_id)
            metrics.update({
                "bilateral_pair_id":pair_id,
                "position":position,
                "left_plate_id":lp["plate_id"],
                "right_plate_id":rp["plate_id"],
                "panel_url":f"/static/generated/{eid}/bilateral/{metrics.pop('panel_filename')}",
                "difference_url":f"/static/generated/{eid}/bilateral/{metrics.pop('difference_filename')}",
                "right_aligned_url":f"/static/generated/{eid}/bilateral/{metrics.pop('right_aligned_filename')}",
            })
            bilateral.append(metrics)

    for p in flat_plates:
        p.pop("_generated_image_path",None)

    result = {
        "exam_id":eid,
        "analysis_type":"contact_liquid_crystal_thermography",
        "source_images":len(sources),
        "plates_detected":total,
        "metadata_items_supplied":len(metadata),
        "bilateral_pairs_created":len(bilateral),
        "sources":sources,
        "bilateral_analysis":bilateral,
        "clinical_risk":None,
        "clinical_claim":"NONE",
        "model_status":"RESEARCH_REFERENCE_ANALYSIS",
    }
    result["saved_at"] = save_exam(result)
    result["report_url"] = f"/reports/{eid}"
    return result

@app.get("/api/exams")
def exam_history(limit: int=50):
    items=list_exams(limit)
    return {"count":len(items),"items":items}

@app.get("/api/exams/{exam_id}")
def exam_detail(exam_id: str):
    item=get_exam(exam_id)
    if item is None:
        raise HTTPException(status_code=404,detail="Exam not found")
    return item

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
