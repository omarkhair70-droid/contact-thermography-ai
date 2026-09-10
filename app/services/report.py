from __future__ import annotations
from html import escape
import math


def _fmt(value, digits=3):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def _score_line(label: str, value) -> str:
    if value is None:
        return ""
    return f"<p><b>{escape(label)}:</b> {_fmt(value)}</p>"


def build_report_html(result: dict):
    exam_id = escape(str(result["exam_id"]))
    profile = result.get("profile_provenance") or {}
    tlc_profile_id = escape(str(result.get("tlc_profile_id") or profile.get("tlc_profile_id") or "unknown"))
    device_profiles = profile.get("device_profile_ids") or result.get("device_profile_ids") or []
    device_text = escape(", ".join(str(item) for item in device_profiles) or "not supplied")
    domain_status = escape(str(profile.get("domain_status") or "UNKNOWN"))

    plates = []
    for src in result.get("sources", []):
        for plate in src.get("plates", []):
            plates.append(plate)

    body = []
    for plate in plates:
        flags = ", ".join(plate.get("qc", {}).get("flags", [])) or "None"
        body.append(f"""
        <article class="card">
          <img src="{escape(str(plate.get('image_url','')))}" />
          <div>
            <h3>{escape(str(plate.get('plate_id', 'plate')))}</h3>
            <p><b>Side / position:</b> {escape(str(plate.get('side')))} / {escape(str(plate.get('position')))}</p>
            <p><b>TLC profile:</b> {escape(str(plate.get('tlc_profile_id') or tlc_profile_id))}</p>
            <p><b>Device profile:</b> {escape(str(plate.get('device_profile_id') or 'not supplied'))}</p>
            <p><b>QC:</b> {escape(str(plate.get('qc', {}).get('status', 'unknown')))}</p>
            <p><b>QC flags:</b> {escape(flags)}</p>
            <p><b>Morphology descriptor:</b> {escape(str(plate.get('morphology_descriptor', 'unknown')))}</p>
            <p><b>Reference anomaly percentile:</b> {_fmt(plate.get('reference_anomaly_percentile'))}</p>
            {_score_line('DINOv2 reference unusualness score', plate.get('dinov2_reference_anomaly_score_0_1'))}
            {_score_line('Fused LCT + DINOv2 reference score', plate.get('dinov2_lct_fused_reference_score_0_1'))}
            <p class="muted">{escape(str(plate.get('dinov2_domain_notice', 'Reference unusualness only; not cancer probability.')))}</p>
          </div>
        </article>
        """)

    pair_blocks = []
    for pair in result.get("bilateral_analysis", []):
        pair_blocks.append(f"""
        <article class="pair">
          <img src="{escape(str(pair.get('panel_url','')))}" />
          <h3>{escape(str(pair.get('bilateral_pair_id', 'pair')))}</h3>
          <p><b>Position:</b> {escape(str(pair.get('position')))}</p>
          <p><b>TLC profile:</b> {escape(str(pair.get('tlc_profile_id') or tlc_profile_id))}</p>
          <p><b>Device profile:</b> {escape(str(pair.get('device_profile_id') or 'not supplied'))}</p>
          <p><b>Reference bilateral asymmetry score:</b> {_fmt(pair.get('reference_asymmetry_score_0_1'))}</p>
          {_score_line('DINOv2 pair reference unusualness score', pair.get('dinov2_pair_reference_anomaly_score_0_1'))}
          <p><b>Response-area delta:</b> {_fmt(pair.get('absolute_area_fraction_delta'))}</p>
          <p><b>Jaccard similarity:</b> {_fmt(pair.get('response_jaccard_similarity'))}</p>
        </article>
        """)

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>LCT Report — {exam_id}</title>
<style>
body{{font-family:Arial,sans-serif;margin:32px;color:#17202a}}
.banner{{padding:16px;border:1px solid #bbb;border-radius:10px;background:#f8f9fa}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}}
.card,.pair{{border:1px solid #ddd;border-radius:10px;padding:14px}}
.card img{{width:100%;max-height:260px;object-fit:contain;background:#111}}
.pair img{{width:100%;object-fit:contain;background:#111}}
.muted{{color:#6c757d}}
</style></head><body>
<h1>Contact Thermography Analysis Report</h1>
<div class="banner">
<p><b>Exam:</b> {exam_id}</p>
<p><b>TLC profile:</b> {tlc_profile_id}</p>
<p><b>Device profile(s):</b> {device_text}</p>
<p><b>Profile/domain status:</b> {domain_status}</p>
<p><b>Source images:</b> {result.get('source_images', 0)} &nbsp; <b>Plates:</b> {result.get('plates_detected', 0)} &nbsp; <b>Bilateral pairs:</b> {result.get('bilateral_pairs_created', 0)}</p>
<p><b>Clinical claim:</b> NONE</p>
<p>This report contains engineering/research analysis of liquid-crystal contact thermograms. Current model scores are not cancer probabilities.</p>
<p>Colour interpretation is profile-dependent. Absolute temperature interpretation remains disabled until the selected TLC formulation is calibrated, and clinical inference remains disabled until a validated clinical model exists.</p>
</div>
<h2>Plate analysis</h2><div class="grid">{''.join(body)}</div>
<h2>Bilateral analysis</h2><div>{''.join(pair_blocks) if pair_blocks else '<p>No true bilateral pairs were created for this exam.</p>'}</div>
</body></html>"""
