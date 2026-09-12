from pathlib import Path
import json
import logging
import uuid
from typing import List

import cv2
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from app.services.reference_store import store
from app.services.analysis_engine import analyze_uploaded_image
from app.services.acquisition_context import AcquisitionContextError, normalize_acquisition_context
from app.services.exam_contracts import (
    DEFAULT_TLC_PROFILE,
    normalize_device_profile,
    normalize_side,
    normalize_tlc_profile,
)
from app.services.bilateral_live import compare_pair
from app.services import dinov2_service
from app.services.dinov2_service import DINOv2UnavailableError
from app.services.tlc_profiles import resolve_tlc_profile
from app.services.db import (
    save_exam,
    get_exam,
    list_exams,
    database_backend,
    database_health,
)
from app.services.report import build_report_html
from app.services.storage import storage
from app.services.json_safety import sanitize_for_json
from app.services.tlc_domain import build_profile_provenance
from app.services.mumguard_session_fusion import (
    SessionFrameInput,
    analyze_bilateral_session,
    persist_session_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "app" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
LOGGER = logging.getLogger("mumguard.human_runtime")

app = FastAPI(
    title="Contact Thermography Intelligence Platform",
    version="0.6.0",
    description=(
        "Research analysis platform for contact liquid-crystal thermography. "
        "Current bundled models are reference-only and have no clinical diagnostic claim."
    ),
)

app.mount(
    "/static/generated",
    StaticFiles(directory=str(storage.generated_root)),
    name="generated",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))


@app.get("/health")
def health():
    db_ok = database_health()
    storage_ok = storage.healthcheck()
    payload = {
        "status": "ok" if db_ok and storage_ok else "degraded",
        "service": "lct-intelligence",
        "version": "0.6.0",
        "clinical_claim": "NONE",
        "live_bilateral_analysis": True,
        "mumguard_session_fusion": True,
        "human_exam_ui": True,
        "human_decision_contract": True,
        "human_runtime_direct": True,
        "live_dinov2_analysis": True,
        "dinov2_backbone": dinov2_service.BACKBONE_NAME,
        "dinov2_runtime_loaded": dinov2_service.runtime.ready,
        "database": database_backend(),
        "storage": storage.backend,
    }
    return JSONResponse(sanitize_for_json(payload), status_code=200 if db_ok and storage_ok else 503)


@app.get("/api/model")
def model_info():
    return sanitize_for_json({
        **store.manifest,
        "mumguard_session_fusion": {
            "architecture": "mumguard_session_fusion_v1",
            "target_species": "human",
            "measurement_mode": "relative_tlc_signal",
            "channels": [
                "core_hyperthermia",
                "bilateral_asymmetry",
                "abnormal_skin_thermal_behavior",
            ],
            "decision_statuses": [
                "INCONCLUSIVE",
                "NOT_CALIBRATED",
                "INDICATION_LOW",
                "INDICATION_INTERMEDIATE",
                "INDICATION_HIGH",
            ],
            "clinical_claim": "NONE",
        },
        "live_dinov2": {
            "backbone": dinov2_service.BACKBONE_NAME,
            "embedding_dim": dinov2_service.EMBEDDING_DIM,
            "source": dinov2_service.DINO_HUB_REPOSITORY,
            "reference_tlc_profile_id": dinov2_service.REFERENCE_TLC_PROFILE_ID,
            "clinical_claim": "NONE",
            "score_semantics": dinov2_service.SCORE_SEMANTICS,
        },
    })


@app.get("/api/reference/plates")
def reference_plates():
    return sanitize_for_json({"count": len(store.plates), "items": store.plate_records()})


@app.get("/api/reference/dinov2")
def dinov2_reference():
    return sanitize_for_json({
        "backbone": "dinov2_vits14",
        "embedding_dim": 384,
        "clinical_claim": "NONE",
        "plates": store.dino_index.to_dict("records"),
        "bilateral_pairs": store.dino_pairs.to_dict("records"),
    })


@app.get("/api/reference/pairs")
def reference_pairs():
    return sanitize_for_json({"count": len(store.pairs), "items": store.pair_records()})


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

    lookup = {}
    for item in items:
        if not isinstance(item, dict) or "filename" not in item:
            continue
        filename = str(item["filename"])
        if filename in lookup:
            raise HTTPException(
                status_code=400,
                detail=f"Duplicate metadata filename is ambiguous: {filename}. Use unique capture filenames.",
            )
        try:
            acquisition_context = normalize_acquisition_context(item)
        except AcquisitionContextError as exc:
            raise HTTPException(status_code=400, detail=f"{filename}: {exc}") from exc
        lookup[filename] = {
            "side": normalize_side(item.get("side")),
            "position": str(item.get("position")) if item.get("position") is not None else None,
            "sequence_index": item.get("sequence_index"),
            "tlc_profile_id": normalize_tlc_profile(item.get("tlc_profile_id")),
            "device_profile_id": normalize_device_profile(item.get("device_profile_id")),
            "species": item.get("species") if item.get("species") in ("mouse", "human") else None,
            "acquisition_type": item.get("acquisition_type") if item.get("acquisition_type") in ("contact-LCT", "radiometric-IR") else None,
            "acquisition_context": acquisition_context,
        }
    return lookup


def _default_file_metadata():
    return {
        "side": "UNKNOWN",
        "position": None,
        "sequence_index": None,
        "tlc_profile_id": DEFAULT_TLC_PROFILE,
        "device_profile_id": None,
        "species": None,
        "acquisition_type": None,
        "acquisition_context": normalize_acquisition_context({}),
    }


def _sequence_index(value, fallback: int) -> int:
    if value is None:
        return int(fallback)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid sequence_index: {value!r}") from exc
    if parsed < 0:
        raise HTTPException(status_code=400, detail="sequence_index must be >= 0")
    return parsed


def _eligible_mumguard_session_frame(tlc_profile_id: str, metadata: dict) -> bool:
    """Keep the human session path profile-aware instead of binding it to one temporary profile id."""
    return (
        tlc_profile_id != "reference-publication-unknown"
        and metadata.get("side") in {"LEFT", "RIGHT"}
        and metadata.get("species") in {None, "human"}
        and metadata.get("acquisition_type") in {None, "contact-LCT"}
    )


@app.post("/api/exams/analyze")
async def analyze_exam(
    files: List[UploadFile] = File(...),
    exam_id: str | None = Form(default=None),
    metadata_json: str | None = Form(default=None),
    tlc_profile_id: str | None = Form(default=None),
    device_profile_id: str | None = Form(default=None),
):
    eid = (exam_id or f"exam-{uuid.uuid4().hex[:10]}").replace("/", "_").replace("\\", "_")
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required")

    metadata = parse_metadata(metadata_json)
    form_tlc_profile_id = normalize_tlc_profile(tlc_profile_id) if tlc_profile_id is not None else None
    form_device_profile_id = normalize_device_profile(device_profile_id) if device_profile_id is not None else None

    upload_names = [upload.filename or "" for upload in files]
    if len(upload_names) != len(set(upload_names)):
        raise HTTPException(
            status_code=400,
            detail="Duplicate uploaded filenames are ambiguous for per-capture metadata. Rename captures uniquely.",
        )

    resolved = []
    for upload in files:
        name = upload.filename or ""
        item = dict(metadata.get(name, _default_file_metadata()))
        if form_tlc_profile_id is not None:
            item["tlc_profile_id"] = form_tlc_profile_id
        if device_profile_id is not None:
            item["device_profile_id"] = form_device_profile_id
        resolved.append((upload, item))

    tlc_profiles = sorted({item["tlc_profile_id"] for _, item in resolved})
    if len(tlc_profiles) != 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "Mixed TLC profiles are not allowed in one examination because colour domains are not "
                f"interchangeable. Submit separate exams per tlc_profile_id: {', '.join(tlc_profiles)}"
            ),
        )
    selected_tlc_profile_id = tlc_profiles[0]
    device_profile_ids = sorted({
        item["device_profile_id"] for _, item in resolved if item["device_profile_id"]
    })
    profile_provenance = build_profile_provenance(selected_tlc_profile_id, device_profile_ids)

    sources = []
    total = 0
    flat_plates = []
    session_frames: list[SessionFrameInput] = []

    for upload_index, (upload, m) in enumerate(resolved, start=1):
        raw = await upload.read()
        if not raw:
            raise HTTPException(status_code=400, detail=f"Empty file: {upload.filename}")

        sequence_index = _sequence_index(m.get("sequence_index"), upload_index)
        if _eligible_mumguard_session_frame(selected_tlc_profile_id, m):
            session_frames.append(
                SessionFrameInput(
                    source=raw,
                    source_name=upload.filename or f"image-{upload_index}",
                    side=m["side"],
                    sequence_index=sequence_index,
                    acquisition_context=m.get("acquisition_context"),
                )
            )

        try:
            result = analyze_uploaded_image(
                raw,
                upload.filename or "image",
                eid,
                m["tlc_profile_id"],
                m["device_profile_id"],
                m.get("acquisition_type"),
                m.get("species"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{upload.filename}: {exc}") from exc
        except DINOv2UnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        for idx, plate in enumerate(result["plates"], start=1):
            plate["side"] = m["side"]
            plate["position"] = m["position"] if m["position"] is not None else f"AUTO-{idx:02d}"
            plate["sequence_index"] = sequence_index
            flat_plates.append(plate)

        sources.append(result)
        total += result["plates_detected"]

    left = {}
    right = {}
    for plate in flat_plates:
        key = (plate["position"], plate["tlc_profile_id"], plate.get("device_profile_id"))
        if plate["side"] == "LEFT":
            left.setdefault(key, []).append(plate)
        elif plate["side"] == "RIGHT":
            right.setdefault(key, []).append(plate)

    bilateral = []
    pair_dir = storage.generated_exam_dir(eid) / "bilateral"
    pair_index = 0
    pair_keys = sorted(
        set(left).intersection(right),
        key=lambda key: tuple("" if value is None else str(value) for value in key),
    )
    for position, pair_tlc_profile_id, pair_device_profile_id in pair_keys:
        key = (position, pair_tlc_profile_id, pair_device_profile_id)
        left_plates = sorted(left[key], key=lambda x: (x.get("sequence_index") or 0, x["plate_id"]))
        right_plates = sorted(right[key], key=lambda x: (x.get("sequence_index") or 0, x["plate_id"]))
        for lp, rp in zip(left_plates, right_plates):
            pair_index += 1
            pair_id = f"{eid}-BP-{pair_index:03d}"
            lbgr = cv2.imread(lp["_generated_image_path"])
            rbgr = cv2.imread(rp["_generated_image_path"])
            if lbgr is None or rbgr is None:
                raise HTTPException(status_code=500, detail="Generated plate image missing during bilateral pairing")

            profile = resolve_tlc_profile(pair_tlc_profile_id)
            metrics = compare_pair(lbgr, rbgr, pair_dir, pair_id, profile)
            aligned_bgr = cv2.imread(str(pair_dir / metrics["right_aligned_filename"]))
            if aligned_bgr is None:
                raise HTTPException(status_code=500, detail=f"Could not read aligned plate for {pair_id}")
            try:
                aligned_embedding = dinov2_service.runtime.encode_rgb(
                    cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
                )
            except DINOv2UnavailableError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            metrics.update(dinov2_service.analyze_pair_embeddings(lp["_dinov2_embedding"], aligned_embedding))
            metrics.update({
                "bilateral_pair_id": pair_id,
                "position": position,
                "tlc_profile_id": pair_tlc_profile_id,
                "device_profile_id": pair_device_profile_id,
                "left_plate_id": lp["plate_id"],
                "right_plate_id": rp["plate_id"],
                "panel_url": storage.generated_url(eid, "bilateral", metrics.pop("panel_filename")),
                "difference_url": storage.generated_url(eid, "bilateral", metrics.pop("difference_filename")),
                "right_aligned_url": storage.generated_url(eid, "bilateral", metrics.pop("right_aligned_filename")),
            })
            bilateral.append(metrics)

    pairing_warnings = []
    left_by_position = {p["position"] for p in flat_plates if p["side"] == "LEFT"}
    right_by_position = {p["position"] for p in flat_plates if p["side"] == "RIGHT"}
    paired_positions = {item["position"] for item in bilateral}
    for position in sorted((left_by_position & right_by_position) - paired_positions, key=str):
        pairing_warnings.append({
            "position": position,
            "reason": "DEVICE_PROFILE_MISMATCH_OR_UNPAIRED_SEQUENCE",
            "message": "LEFT/RIGHT plates were not paired across incompatible or unmatched device provenance.",
        })

    mumguard_session = None
    session_sides = {frame.side for frame in session_frames}
    if session_sides == {"LEFT", "RIGHT"}:
        try:
            mumguard_session, session_arrays = analyze_bilateral_session(
                session_frames,
                tlc_profile_id=selected_tlc_profile_id,
                include_dino=bool(dinov2_service.runtime.ready),
            )
        except ValueError as exc:
            pairing_warnings.append({
                "position": "SESSION",
                "reason": "MUMGUARD_SESSION_FUSION_FAILED",
                "message": str(exc),
            })
        else:
            session_folder = "mumguard-session"
            persisted = persist_session_evidence(
                mumguard_session,
                session_arrays,
                storage.generated_exam_dir(eid) / session_folder,
            )
            mumguard_session.update({
                "evidence_url": storage.generated_url(eid, session_folder, persisted["evidence_filename"]),
                "maps_url": storage.generated_url(eid, session_folder, persisted["maps_filename"]),
            })
    elif session_frames:
        pairing_warnings.append({
            "position": "SESSION",
            "reason": "MUMGUARD_BILATERAL_SESSION_INCOMPLETE",
            "message": "Session fusion requires at least one LEFT and one RIGHT MumGuard capture.",
        })

    for plate in flat_plates:
        plate.pop("_generated_image_path", None)
        plate.pop("_dinov2_embedding", None)

    tlc_profile_ids = sorted({p["tlc_profile_id"] for p in flat_plates})
    device_profile_ids = sorted({
        p["device_profile_id"] for p in flat_plates if p.get("device_profile_id")
    })

    result = {
        "exam_id": eid,
        "analysis_type": "contact_liquid_crystal_thermography",
        "tlc_profile_id": selected_tlc_profile_id,
        "profile_provenance": profile_provenance,
        "source_images": len(sources),
        "plates_detected": total,
        "metadata_items_supplied": len(metadata),
        "bilateral_pairs_created": len(bilateral),
        "bilateral_pairing_warnings": pairing_warnings,
        "sources": sources,
        "bilateral_analysis": bilateral,
        "mumguard_session_evidence": mumguard_session,
        "tlc_profile_ids": tlc_profile_ids,
        "device_profile_ids": device_profile_ids,
        "tlc_profile_provenance": [
            resolve_tlc_profile(profile_id).provenance() for profile_id in tlc_profile_ids
        ],
        "clinical_risk": None,
        "clinical_claim": "NONE",
        "model_status": (
            "MUMGUARD_SESSION_EVIDENCE_RESEARCH"
            if mumguard_session is not None
            else "RESEARCH_REFERENCE_ANALYSIS"
        ),
    }
    result = sanitize_for_json(result)
    result["saved_at"] = save_exam(result)
    result["report_url"] = f"/reports/{eid}"
    return result


@app.post("/api/human-exams/analyze")
async def analyze_human_exam(
    files: List[UploadFile] = File(...),
    exam_id: str | None = Form(default=None),
    metadata_json: str | None = Form(default=None),
    tlc_profile_id: str | None = Form(default=None),
    device_profile_id: str | None = Form(default=None),
    include_dino: bool = Form(default=True),
):
    """Direct human-session runtime.

    This route intentionally bypasses legacy per-image reference analysis and
    legacy LEFT/RIGHT plate pairing. Human captures are persisted once, then the
    complete bilateral session is reconstructed and analyzed as one examination.
    """
    eid = (exam_id or f"exam-{uuid.uuid4().hex[:10]}").replace("/", "_").replace("\\", "_")
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required")

    metadata = parse_metadata(metadata_json)
    form_tlc_profile_id = normalize_tlc_profile(tlc_profile_id) if tlc_profile_id is not None else None
    form_device_profile_id = normalize_device_profile(device_profile_id) if device_profile_id is not None else None

    upload_names = [upload.filename or "" for upload in files]
    if len(upload_names) != len(set(upload_names)):
        raise HTTPException(
            status_code=400,
            detail="Duplicate uploaded filenames are ambiguous for per-capture metadata. Rename captures uniquely.",
        )

    resolved = []
    for upload in files:
        name = upload.filename or ""
        item = dict(metadata.get(name, _default_file_metadata()))
        if form_tlc_profile_id is not None:
            item["tlc_profile_id"] = form_tlc_profile_id
        if device_profile_id is not None:
            item["device_profile_id"] = form_device_profile_id
        resolved.append((upload, item))

    tlc_profiles = sorted({item["tlc_profile_id"] for _, item in resolved})
    if len(tlc_profiles) != 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "Mixed TLC profiles are not allowed in one human examination. "
                f"Submit separate exams per tlc_profile_id: {', '.join(tlc_profiles)}"
            ),
        )
    selected_tlc_profile_id = tlc_profiles[0]
    if selected_tlc_profile_id == "reference-publication-unknown":
        raise HTTPException(
            status_code=400,
            detail="The publication-reference TLC profile is not valid for a live MumGuard human session.",
        )

    device_profile_ids = sorted({
        item["device_profile_id"] for _, item in resolved if item["device_profile_id"]
    })
    profile_provenance = build_profile_provenance(selected_tlc_profile_id, device_profile_ids)
    session_frames: list[SessionFrameInput] = []

    for upload_index, (upload, item) in enumerate(resolved, start=1):
        name = upload.filename or f"image-{upload_index}"
        if item.get("side") not in {"LEFT", "RIGHT"}:
            raise HTTPException(status_code=400, detail=f"{name}: human session captures require LEFT or RIGHT side")
        if item.get("species") not in {None, "human"}:
            raise HTTPException(status_code=400, detail=f"{name}: human session captures must use species=human")
        if item.get("acquisition_type") not in {None, "contact-LCT"}:
            raise HTTPException(status_code=400, detail=f"{name}: human session captures must use acquisition_type=contact-LCT")

        raw = await upload.read()
        if not raw:
            raise HTTPException(status_code=400, detail=f"Empty file: {name}")
        storage.persist_upload(eid, name, raw)
        session_frames.append(
            SessionFrameInput(
                source=raw,
                source_name=name,
                side=item["side"],
                sequence_index=_sequence_index(item.get("sequence_index"), upload_index),
                acquisition_context=item.get("acquisition_context"),
            )
        )

    sides = {frame.side for frame in session_frames}
    if sides != {"LEFT", "RIGHT"}:
        raise HTTPException(
            status_code=400,
            detail="A complete MumGuard human examination requires at least one LEFT and one RIGHT capture.",
        )

    LOGGER.info(
        "HUMAN_EXAM_RECEIVED exam_id=%s frames=%d left=%d right=%d include_dino=%s",
        eid,
        len(session_frames),
        sum(frame.side == "LEFT" for frame in session_frames),
        sum(frame.side == "RIGHT" for frame in session_frames),
        include_dino,
    )
    try:
        mumguard_session, session_arrays = await run_in_threadpool(
            analyze_bilateral_session,
            session_frames,
            tlc_profile_id=selected_tlc_profile_id,
            include_dino=include_dino,
        )
    except ValueError as exc:
        LOGGER.warning("HUMAN_EXAM_REJECTED exam_id=%s reason=%s", eid, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    LOGGER.info("HUMAN_EXAM_FUSION_DONE exam_id=%s status=%s", eid, mumguard_session.get("status"))
    session_folder = "mumguard-session"
    persisted = await run_in_threadpool(
        persist_session_evidence,
        mumguard_session,
        session_arrays,
        storage.generated_exam_dir(eid) / session_folder,
    )
    mumguard_session.update({
        "evidence_url": storage.generated_url(eid, session_folder, persisted["evidence_filename"]),
        "maps_url": storage.generated_url(eid, session_folder, persisted["maps_filename"]),
    })

    result = {
        "exam_id": eid,
        "analysis_type": "contact_liquid_crystal_thermography",
        "human_runtime": "DIRECT_SESSION_ONLY",
        "legacy_reference_analysis_executed": False,
        "tlc_profile_id": selected_tlc_profile_id,
        "profile_provenance": profile_provenance,
        "source_images": len(session_frames),
        "metadata_items_supplied": len(metadata),
        "bilateral_pairs_created": 0,
        "bilateral_pairing_warnings": [],
        "sources": [],
        "bilateral_analysis": [],
        "mumguard_session_evidence": mumguard_session,
        "tlc_profile_ids": [selected_tlc_profile_id],
        "device_profile_ids": device_profile_ids,
        "tlc_profile_provenance": [resolve_tlc_profile(selected_tlc_profile_id).provenance()],
        "clinical_risk": None,
        "clinical_claim": "NONE",
        "model_status": "MUMGUARD_SESSION_EVIDENCE_RESEARCH",
    }
    result = sanitize_for_json(result)
    result["saved_at"] = save_exam(result)
    result["report_url"] = f"/reports/{eid}"
    LOGGER.info("HUMAN_EXAM_SAVED exam_id=%s", eid)
    return result


@app.get("/api/exams")
def exam_history(limit: int = 50):
    items = list_exams(limit)
    return sanitize_for_json({"count": len(items), "items": items})


@app.get("/api/exams/{exam_id}")
def exam_detail(exam_id: str):
    item = get_exam(exam_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Exam not found")
    return sanitize_for_json(item)


@app.get("/reports/{exam_id}", response_class=HTMLResponse)
def exam_report(exam_id: str):
    item = get_exam(exam_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Exam not found")
    return HTMLResponse(build_report_html(item["result"]))


@app.get("/human-exam", response_class=HTMLResponse)
def human_exam_ui():
    page = STATIC_DIR / "human-exam.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Human examination UI not available")
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "plates": store.plate_records(),
            "pairs": store.pair_records(),
            "manifest": store.manifest,
        },
    )
