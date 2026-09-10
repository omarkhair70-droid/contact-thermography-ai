from __future__ import annotations
from html import escape


def _safe_score(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def build_report_html(result: dict):
    exam_id=escape(str(result["exam_id"]))
    tlc_profile=escape(str(result.get("tlc_profile_id") or "profile not recorded"))
    device_profile=escape(str(result.get("device_profile_id") or "not specified"))
    plates=[]
    for src in result.get("sources",[]):
        for p in src.get("plates",[]):
            plates.append(p)

    body=[]
    for p in plates:
        flags=", ".join(p.get("qc",{}).get("flags",[])) or "No acquisition flags reported"
        body.append(f"""
        <article class="plate-card">
          <div class="visuals">
            <figure><img src="{escape(p.get('image_url',''))}" /><figcaption>Original</figcaption></figure>
            <figure><img src="{escape(p.get('response_mask_url',''))}" /><figcaption>Response mask</figcaption></figure>
          </div>
          <div class="plate-copy">
            <div class="title-row"><h3>{escape(str(p.get('plate_id','Plate')))}</h3><span class="badge">{escape(str(p.get('qc',{}).get('status','REVIEW')))}</span></div>
            <dl>
              <div><dt>Side / position</dt><dd>{escape(str(p.get('side','UNKNOWN')))} · {escape(str(p.get('position','—')))}</dd></div>
              <div><dt>Morphology</dt><dd>{escape(str(p.get('morphology_descriptor','—')))}</dd></div>
              <div><dt>Response area</dt><dd>{_safe_score(p.get('signal_features',{}).get('response_area_fraction'))}</dd></div>
              <div><dt>Reference unusualness</dt><dd>{_safe_score(p.get('reference_anomaly_percentile'))}</dd></div>
            </dl>
            <p class="note"><strong>QC:</strong> {escape(flags)}.</p>
            <p class="boundary">Reference unusualness only; not cancer probability.</p>
          </div>
        </article>
        """)

    pair_blocks=[]
    for p in result.get("bilateral_analysis",[]):
        pair_blocks.append(f"""
        <article class="pair-card">
          <img class="pair-hero" src="{escape(p.get('panel_url',''))}" />
          <div class="pair-copy">
            <div class="title-row"><h3>{escape(str(p.get('bilateral_pair_id','Bilateral pair')))}</h3><span class="badge">{escape(str(p.get('position','Matched')))}</span></div>
            <dl>
              <div><dt>LEFT ↔ RIGHT</dt><dd>{escape(str(p.get('left_plate_id','—')))} ↔ {escape(str(p.get('right_plate_id','—')))}</dd></div>
              <div><dt>Reference asymmetry</dt><dd>{_safe_score(p.get('reference_asymmetry_score_0_1'))}</dd></div>
              <div><dt>Response-area Δ</dt><dd>{_safe_score(p.get('absolute_area_fraction_delta'))}</dd></div>
              <div><dt>Jaccard similarity</dt><dd>{_safe_score(p.get('response_jaccard_similarity'))}</dd></div>
            </dl>
            <p class="boundary">Matched-position research comparison; not a diagnostic conclusion.</p>
          </div>
        </article>
        """)

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LCT Analysis Report — {exam_id}</title>
<style>
:root{{font-family:Inter,Arial,sans-serif;color:#13231f;background:#eef3f1}}*{{box-sizing:border-box}}body{{margin:0;background:#eef3f1;color:#13231f}}.page{{max-width:1180px;margin:0 auto;padding:42px 28px 70px}}header{{display:flex;justify-content:space-between;gap:24px;align-items:flex-start;margin-bottom:20px}}.brand{{font-size:12px;letter-spacing:.13em;text-transform:uppercase;color:#367363;font-weight:800}}h1{{font-size:30px;line-height:1.15;margin:8px 0 0}}.mode{{border:1px solid #b7ccc6;border-radius:999px;padding:7px 10px;background:#f8fbfa;color:#45645c;font-size:11px;white-space:nowrap}}.summary{{display:grid;grid-template-columns:1.2fr repeat(3,.7fr);gap:10px;margin:18px 0}}.summary>div{{background:#fff;border:1px solid #d4e0dc;border-radius:13px;padding:14px}}.summary small{{font-size:10px;color:#738780;text-transform:uppercase;letter-spacing:.08em}}.summary b{{display:block;font-size:17px;margin-top:5px;overflow-wrap:anywhere}}.profiles{{background:#f8fbfa;border:1px solid #d3dfdb;border-radius:14px;padding:15px 17px;display:flex;gap:28px;flex-wrap:wrap;margin-bottom:14px;font-size:11px}}.profiles span{{color:#70837d}}.boundary-banner{{border-left:4px solid #4a9f87;background:#f8fbfa;border-top:1px solid #d4e0dc;border-right:1px solid #d4e0dc;border-bottom:1px solid #d4e0dc;border-radius:12px;padding:14px 16px;color:#526b64;font-size:12px;line-height:1.55;margin-bottom:28px}}h2{{font-size:18px;margin:31px 0 12px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:13px}}.plate-card,.pair-card{{background:#fff;border:1px solid #d4e0dc;border-radius:15px;overflow:hidden;break-inside:avoid}}.visuals{{display:grid;grid-template-columns:1fr 1fr;background:#07100f}}figure{{margin:0;position:relative}}figure+figure{{border-left:1px solid #23332f}}figure img{{width:100%;aspect-ratio:1/1;object-fit:contain;display:block}}figcaption{{position:absolute;left:8px;bottom:8px;background:#07100fd9;color:#d6e4e0;border:1px solid #324640;border-radius:999px;padding:4px 7px;font-size:9px}}.plate-copy,.pair-copy{{padding:14px}}.title-row{{display:flex;justify-content:space-between;gap:12px;align-items:center}}h3{{font-size:13px;margin:0}}.badge{{border:1px solid #c6d6d1;border-radius:999px;padding:4px 7px;font-size:9px;color:#58736b}}dl{{margin:12px 0 0;display:grid;gap:7px}}dl div{{display:flex;justify-content:space-between;gap:14px;border-bottom:1px solid #edf2f0;padding-bottom:6px}}dt{{font-size:10px;color:#71847e}}dd{{font-size:10px;margin:0;text-align:right;max-width:60%}}.note,.boundary{{font-size:10px;line-height:1.45;margin:10px 0 0;color:#6b7f78}}.boundary{{color:#487064}}.pair-wrap{{display:grid;gap:13px}}.pair-hero{{width:100%;aspect-ratio:4/1.15;object-fit:contain;background:#07100f;display:block}}footer{{margin-top:40px;padding-top:18px;border-top:1px solid #cedbd7;color:#74867f;font-size:10px;line-height:1.5}}@media(max-width:720px){{.summary{{grid-template-columns:1fr 1fr}}header{{display:block}}.mode{{display:inline-block;margin-top:14px}}.page{{padding:26px 16px 55px}}}}@media print{{body{{background:#fff}}.page{{max-width:none;padding:20px}}.plate-card,.pair-card,.summary>div{{box-shadow:none}}}}
</style></head><body><div class="page">
<header><div><div class="brand">Contact Thermography AI</div><h1>Contact LCT Analysis Report</h1></div><span class="mode">Research / reference mode</span></header>
<div class="summary"><div><small>Examination</small><b>{exam_id}</b></div><div><small>Source images</small><b>{result.get('source_images',0)}</b></div><div><small>Plates</small><b>{result.get('plates_detected',0)}</b></div><div><small>Bilateral pairs</small><b>{result.get('bilateral_pairs_created',0)}</b></div></div>
<div class="profiles"><div><span>TLC profile</span><br><strong>{tlc_profile}</strong></div><div><span>Device profile</span><br><strong>{device_profile}</strong></div><div><span>Analysis type</span><br><strong>Contact liquid crystal thermography</strong></div></div>
<div class="boundary-banner"><strong>No clinical diagnostic claim.</strong> This report contains engineering/research analysis of contact liquid-crystal thermograms. Current model scores describe reference unusualness and bilateral image asymmetry; they are not cancer probabilities. Colour response from one TLC formulation must not be assumed to share an absolute temperature meaning with another formulation.</div>
<h2>Plate analysis</h2><div class="grid">{''.join(body) if body else '<p>No plate results were recorded.</p>'}</div>
<h2>Bilateral analysis</h2><div class="pair-wrap">{''.join(pair_blocks) if pair_blocks else '<div class="boundary-banner">No true LEFT / RIGHT pairs were created for this examination.</div>'}</div>
<footer>Generated by Contact Thermography AI · Research/reference analysis only · Examination profile provenance is retained with the saved result.</footer>
</div></body></html>"""
