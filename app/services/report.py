from __future__ import annotations
from pathlib import Path
from html import escape

def _score_line(label: str, value) -> str:
    if value is None:
        return ""
    return f"<p><b>{escape(label)}:</b> {float(value):.3f}</p>"

def build_report_html(result: dict):
    exam_id=escape(result["exam_id"])
    plates=[]
    for src in result.get("sources",[]):
        for p in src.get("plates",[]):
            plates.append(p)

    body=[]
    for p in plates:
        flags=", ".join(p["qc"].get("flags",[])) or "None"
        body.append(f"""
        <article class="card">
          <img src="{escape(p.get('image_url',''))}" />
          <div>
            <h3>{escape(p['plate_id'])}</h3>
            <p><b>Side / position:</b> {escape(str(p.get('side')))} / {escape(str(p.get('position')))}</p>
            <p><b>TLC profile:</b> {escape(str(p.get('tlc_profile_id', 'unknown')))}</p>
            <p><b>Device profile:</b> {escape(str(p.get('device_profile_id') or 'not supplied'))}</p>
            <p><b>QC:</b> {escape(p['qc']['status'])}</p>
            <p><b>QC flags:</b> {escape(flags)}</p>
            <p><b>Morphology descriptor:</b> {escape(p['morphology_descriptor'])}</p>
            <p><b>Reference anomaly percentile:</b> {p['reference_anomaly_percentile']:.3f}</p>
            {_score_line('DINOv2 reference unusualness score', p.get('dinov2_reference_anomaly_score_0_1'))}
            {_score_line('Fused LCT + DINOv2 reference score', p.get('dinov2_lct_fused_reference_score_0_1'))}
            <p class="muted">{escape(str(p.get('dinov2_domain_notice', 'Reference unusualness only; not cancer probability.')))}</p>
          </div>
        </article>
        """)

    pair_blocks=[]
    for p in result.get("bilateral_analysis",[]):
        pair_blocks.append(f"""
        <article class="pair">
          <img src="{escape(p.get('panel_url',''))}" />
          <h3>{escape(p['bilateral_pair_id'])}</h3>
          <p><b>Position:</b> {escape(str(p.get('position')))}</p>
          <p><b>TLC profile:</b> {escape(str(p.get('tlc_profile_id', 'unknown')))}</p>
          <p><b>Device profile:</b> {escape(str(p.get('device_profile_id') or 'not supplied'))}</p>
          <p><b>Reference bilateral asymmetry score:</b> {p['reference_asymmetry_score_0_1']:.3f}</p>
          {_score_line('DINOv2 pair reference unusualness score', p.get('dinov2_pair_reference_anomaly_score_0_1'))}
          <p><b>Response-area delta:</b> {p['absolute_area_fraction_delta']:.3f}</p>
          <p><b>Jaccard similarity:</b> {p['response_jaccard_similarity']:.3f}</p>
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
<p><b>Source images:</b> {result['source_images']} &nbsp; <b>Plates:</b> {result['plates_detected']} &nbsp; <b>Bilateral pairs:</b> {result['bilateral_pairs_created']}</p>
<p><b>TLC profiles:</b> {escape(', '.join(result.get('tlc_profile_ids', [])) or 'not recorded')}</p>
<p><b>Device profiles:</b> {escape(', '.join(result.get('device_profile_ids', [])) or 'not supplied')}</p>
<p><b>Clinical claim:</b> NONE</p>
<p>This report contains engineering/research analysis of liquid-crystal contact thermograms. Current model scores are not cancer probabilities.</p>
</div>
<h2>Plate analysis</h2><div class="grid">{''.join(body)}</div>
<h2>Bilateral analysis</h2><div>{''.join(pair_blocks) if pair_blocks else '<p>No true bilateral pairs were created for this exam.</p>'}</div>
</body></html>"""
