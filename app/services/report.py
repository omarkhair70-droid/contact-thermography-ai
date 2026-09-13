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


def _native_research_block(plate: dict) -> str:
    native = plate.get("native_binary_research")
    if not isinstance(native, dict):
        return ""
    if not native.get("available") or not native.get("research_binary_class"):
        return ""
    cls = escape(str(native.get("research_binary_class")))
    score = _fmt(native.get("model_score"))
    validation = escape(str(native.get("validation_status") or "UNVALIDATED_RESEARCH"))
    warning = escape(
        str(
            native.get("warning")
            or "Experimental research model; current training cohort contains one no-tumor subject."
        )
    )
    return f"""
            <div class="native-research">
              <p><b>Experimental native class:</b> {cls}</p>
              <p><b>Research model score:</b> {score}</p>
              <p><b>Validation status:</b> {validation}</p>
              <p class="muted">{warning}</p>
              <p class="muted">This score is not a cancer probability, diagnosis, or tumor-size estimate.</p>
            </div>
    """


def _research_finding_block(session: dict) -> str:
    finding = session.get("research_finding")
    if not isinstance(finding, dict):
        source = session.get("human_transfer_source_support")
        if isinstance(source, dict):
            finding = {
                "status": source.get("research_decision", "INCONCLUSIVE"),
                "research_concern_score": source.get("research_concern_score"),
                "domain_status": source.get("status"),
                "reason": source.get("reason"),
                "score_semantics": source.get("score_semantics"),
            }
        else:
            return ""

    status = escape(str(finding.get("status") or "INCONCLUSIVE"))
    concern = finding.get("research_concern_score")
    concern_html = f"{_fmt(concern, 1)}/100" if concern is not None else "not available"
    domain = escape(str(finding.get("domain_status") or "UNKNOWN"))
    side = escape(str(finding.get("likely_side") or "not localized"))
    reason = escape(str(finding.get("reason") or ""))
    semantics = escape(
        str(
            finding.get("score_semantics")
            or "Research concern only; not a cancer probability or clinical diagnosis."
        )
    )
    channels = finding.get("dominant_channels") or []
    channel_html = ""
    if isinstance(channels, list) and channels:
        items = []
        for item in channels[:3]:
            if isinstance(item, dict):
                items.append(
                    f"<li>{escape(str(item.get('channel', 'evidence')))}: {_fmt(item.get('evidence'))}</li>"
                )
        if items:
            channel_html = "<ul>" + "".join(items) + "</ul>"

    return f"""
      <div class="research-finding">
        <div class="eyebrow">Primary research finding</div>
        <h2>{status}</h2>
        <div class="finding-grid">
          <div><b>Research Concern Score</b><br><span>{concern_html}</span></div>
          <div><b>Transfer/domain status</b><br><span>{domain}</span></div>
          <div><b>Likely evidence side</b><br><span>{side}</span></div>
        </div>
        {channel_html}
        <p>{reason}</p>
        <p class="muted">{semantics}</p>
        <p class="muted"><b>Research result — not a clinical diagnosis.</b> Clinical risk remains unavailable.</p>
      </div>
    """


def _decision_block(session: dict) -> str:
    decision = session.get("human_decision")
    if not isinstance(decision, dict):
        return ""
    status = escape(str(decision.get("decision_status") or decision.get("status") or "NOT_CALIBRATED"))
    indication = decision.get("indication")
    risk = decision.get("risk_score")
    reason = escape(str(decision.get("reason") or ""))
    indication_html = escape(str(indication)) if indication is not None else "not available"
    risk_html = _fmt(risk) if risk is not None else "not available"
    calibrated = "yes" if decision.get("calibrated") else "no"
    return f"""
      <div class="decision-block">
        <h3>Human decision layer — clinical calibration (secondary)</h3>
        <p><b>Decision status:</b> {status}</p>
        <p><b>Indication:</b> {indication_html}</p>
        <p><b>Risk score:</b> {risk_html}</p>
        <p><b>Human calibration active:</b> {calibrated}</p>
        <p class="muted">{reason}</p>
      </div>
    """


def _session_preview_block(session: dict) -> str:
    previews = session.get("preview_filenames")
    maps_url = str(session.get("maps_url") or "")
    if not isinstance(previews, dict) or not previews or "/" not in maps_url:
        return ""
    base = maps_url.rsplit("/", 1)[0]
    labels = {
        "left_thermal_evidence": "LEFT thermal evidence",
        "right_thermal_evidence": "RIGHT thermal evidence",
        "left_fused_evidence": "LEFT thermal + AI fused evidence",
        "right_fused_evidence": "RIGHT thermal + AI fused evidence",
        "bilateral_asymmetry": "Bilateral asymmetry",
    }
    cards = []
    for key, label in labels.items():
        filename = previews.get(key)
        if not filename:
            continue
        url = escape(f"{base}/{filename}")
        cards.append(
            f'<figure class="map-card"><img src="{url}" alt="{escape(label)}">'
            f'<figcaption>{escape(label)}</figcaption></figure>'
        )
    if not cards:
        return ""
    return '<h3>Explainable session maps</h3><div class="map-grid">' + "".join(cards) + "</div>"


def _performance_block(session: dict) -> str:
    perf = session.get("performance")
    if not isinstance(perf, dict):
        return ""
    total = perf.get("total_analysis_s")
    if total is None:
        return ""
    return f"""
      <div class="performance">
        <b>Runtime:</b> {_fmt(total, 1)} s total &nbsp;·&nbsp;
        preparation {_fmt(perf.get('frame_preparation_s'), 1)} s &nbsp;·&nbsp;
        DINO {_fmt(perf.get('dino_batch_s'), 1)} s &nbsp;·&nbsp;
        reconstruction {_fmt(perf.get('side_reconstruction_s'), 1)} s
      </div>
    """


def _session_block(result: dict) -> str:
    session = result.get("mumguard_session_evidence")
    if not isinstance(session, dict):
        return '<p>No complete MumGuard bilateral session was available for session-level fusion.</p>'
    scores = session.get("three_channel_scores") or {}
    left = session.get("left") or {}
    right = session.get("right") or {}
    bilateral = session.get("bilateral") or {}
    evidence_url = escape(str(session.get("evidence_url") or ""))
    maps_url = escape(str(session.get("maps_url") or ""))
    ai_text = "available" if session.get("ai_evidence_available") else "not included in session fusion"
    links = []
    if evidence_url:
        links.append(f'<a href="{evidence_url}">session evidence JSON</a>')
    if maps_url:
        links.append(f'<a href="{maps_url}">session numerical maps bundle</a>')
    links_html = " &nbsp; ".join(links)
    if links_html:
        links_html = f"<p>{links_html}</p>"
    return f"""
    <article class="session">
      {_research_finding_block(session)}
      <h2>MumGuard measurement evidence</h2>
      <p><b>Architecture:</b> {escape(str(session.get('architecture', 'unknown')))}</p>
      <p><b>Status:</b> {escape(str(session.get('status', 'unknown')))}</p>
      <p><b>Measurement mode:</b> {escape(str(session.get('measurement_mode', 'unknown')))}</p>
      <p><b>Target species:</b> {escape(str(session.get('target_species', 'human')))}</p>
      <div class="score-grid">
        <div><b>Core hyperthermia evidence</b><br>{_fmt(scores.get('core_hyperthermia_score'))}</div>
        <div><b>Bilateral asymmetry evidence</b><br>{_fmt(scores.get('bilateral_asymmetry_score'))}</div>
        <div><b>Abnormal skin behaviour</b><br>{_fmt(scores.get('abnormal_skin_behavior_score'))}</div>
        <div><b>Overall measurement evidence</b><br>{_fmt(scores.get('overall_measurement_evidence_score'))}</div>
      </div>
      {_decision_block(session)}
      <p><b>LEFT observable field:</b> {_fmt(left.get('observable_fraction'))} &nbsp; <b>RIGHT observable field:</b> {_fmt(right.get('observable_fraction'))}</p>
      <p><b>Bilateral joint coverage:</b> {_fmt((bilateral.get('features') or {}).get('joint_fraction'))}</p>
      <p><b>AI evidence:</b> {escape(ai_text)}</p>
      {_performance_block(session)}
      <p class="muted">{escape(str(session.get('response_support_semantics', '')))}</p>
      <p class="muted">These are research measurement-evidence scores, not cancer probabilities or a diagnosis.</p>
      {_session_preview_block(session)}
      {links_html}
    </article>
    """


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
        local = plate.get("local_research_evidence")
        local_html = ""
        if isinstance(local, dict):
            local_html = ('<p><b>MumGuard local research:</b> '
                + escape(str(local.get('status', 'UNAVAILABLE')))
                + '</p><p>Contact certainty: unknown. Disease decision: abstain.</p><p>'
                + escape(', '.join(str(r) for r in local.get('quality_reasons', []))) + '</p>')
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
            {_native_research_block(plate)}
            {local_html}
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

    legacy_html = ""
    if result.get("human_runtime") != "DIRECT_SESSION_ONLY":
        legacy_html = (
            f"<h2>Plate analysis</h2><div class=\"grid\">{''.join(body)}</div>"
            f"<h2>Legacy pairwise bilateral analysis</h2><div>{''.join(pair_blocks) if pair_blocks else '<p>No legacy pairwise bilateral pairs were created for this exam.</p>'}</div>"
        )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>MumGuard Examination Report — {exam_id}</title>
<style>
body{{font-family:Arial,sans-serif;margin:32px;color:#17202a;background:#fbfdfc}}
.banner{{padding:16px;border:1px solid #bbb;border-radius:10px;background:#f8f9fa}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}}
.card,.pair,.session{{border:1px solid #ddd;border-radius:10px;padding:14px;margin-bottom:16px;background:white}}
.card img{{width:100%;max-height:260px;object-fit:contain;background:#111}}
.pair img{{width:100%;object-fit:contain;background:#111}}
.native-research{{margin:12px 0;padding:10px;border:1px solid #d1a84d;border-radius:8px;background:#fff9e9}}
.research-finding{{margin:0 0 18px;padding:18px;border:2px solid #3d8b72;border-radius:12px;background:#eefaf6}}
.research-finding h2{{font-size:28px;margin:5px 0 14px}}
.eyebrow{{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:#47675d}}
.finding-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:12px 0}}
.finding-grid>div{{padding:10px;border:1px solid #b9d8ce;border-radius:8px;background:white}}
.decision-block{{margin:14px 0;padding:12px;border:1px solid #b9c9c4;border-radius:8px;background:#f8faf9}}
.performance{{margin:12px 0;padding:10px;border:1px solid #ddd;border-radius:8px;background:#fafafa}}
.score-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:12px 0}}
.score-grid>div{{border:1px solid #ddd;border-radius:8px;padding:10px;background:#fafafa}}
.map-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:12px 0}}
.map-card{{margin:0;border:1px solid #ddd;border-radius:8px;overflow:hidden;background:#111}}
.map-card img{{width:100%;height:180px;object-fit:contain;display:block}}
.map-card figcaption{{padding:8px;background:#fafafa;font-size:12px}}
.muted{{color:#6c757d}}
</style></head><body>
<h1>MumGuard Examination Report</h1>
<div class="banner">
<p><b>Exam:</b> {exam_id}</p>
<p><b>TLC profile:</b> {tlc_profile_id}</p>
<p><b>Device profile(s):</b> {device_text}</p>
<p><b>Profile/domain status:</b> {domain_status}</p>
<p><b>Source images:</b> {result.get('source_images', 0)} &nbsp; <b>Plates:</b> {result.get('plates_detected', 0)} &nbsp; <b>Legacy bilateral pairs:</b> {result.get('bilateral_pairs_created', 0)}</p>
<p><b>Clinical claim:</b> NONE</p>
<p>This report contains engineering/research analysis of liquid-crystal contact thermograms. Current model scores are not cancer probabilities.</p>
<p>Absolute temperature interpretation remains disabled until the selected TLC formulation is calibrated.</p>
</div>
{_session_block(result)}
{legacy_html}
</body></html>"""
