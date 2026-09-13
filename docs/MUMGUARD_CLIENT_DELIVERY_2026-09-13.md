# MumGuard — Client Delivery Guide

**Date:** 2026-09-13  
**Canonical target branch:** `integration`  
**Purpose:** freeze scope, run the final demo safely, and explain the product to the client in normal language.

## 1. What is being delivered

MumGuard now has a browser-based human Contact-TLC examination flow:

`LEFT + RIGHT MumGuard captures -> bilateral reconstruction -> relative thermal evidence -> focused/core thermal response -> bilateral asymmetry -> thermal-distribution evidence -> local visual AI evidence -> MumGuard research decision -> result + maps + report`

The primary user-facing result is one of:

- `Tumor-like pattern detected` (`SUSPICIOUS_RESEARCH` internally)
- `No tumor-like pattern detected` (`NOT_SUSPICIOUS_RESEARCH` internally)
- `Inconclusive`

The screen also shows:

- Research Concern Score 0–100 when available;
- side with strongest evidence;
- focused thermal-response evidence;
- LEFT/RIGHT asymmetry evidence;
- thermal-distribution evidence;
- analysis runtime;
- explainable evidence maps;
- full report.

Technical provenance remains available inside **Technical details** instead of dominating the client-facing screen.

## 2. UI design decision

The final operator UI is **brand-aligned, not a pixel-for-pixel copy of mumguard.net**.

It uses the public MumGuard positioning as the design brief:

- early-detection / breast-health product;
- simple guided examination;
- approachable home/clinical technology rather than a developer dashboard;
- clean, light medical interface;
- MumGuard name and a warm breast-health accent system;
- result-first hierarchy.

The public marketing website and the examination console serve different jobs. The website sells/explains the company; this console must make an examination obvious and fast. Therefore the console keeps the same product personality while using a purpose-built workflow.

No remote logo/image dependency is required for the application to work. The current header uses a local CSS mark + MumGuard wordmark. If the client supplies the canonical SVG/PNG logo asset, it can be swapped into the header without changing the workflow.

## 3. Exact client demo flow

1. Open `/human-exam`.
2. Add all LEFT captures to **LEFT breast** in scan order.
3. Add all RIGHT captures to **RIGHT breast** in scan order.
4. Leave advanced acquisition fields at defaults unless real metadata are known.
5. Click **Analyze examination**.
6. Read the large primary finding first.
7. Review Research Concern Score and the strongest-evidence side.
8. Review evidence maps.
9. Open **Full report**.
10. Only open **Technical details** if the client asks about provenance, DINO, transfer reference, calibration state, or raw evidence files.

## 4. How to explain what was built

Use this architecture story instead of listing PRs or model names:

1. MumGuard images are Contact-TLC colour responses, not ordinary RGB photos.
2. The application first extracts a relative thermal/TLC signal from the visible response.
3. Multiple captures from each side are reconstructed while preserving overlap/coverage.
4. It measures three major physical/evidence channels: focused/core thermal response, LEFT-vs-RIGHT asymmetry, and abnormal thermal distribution.
5. A frozen visual AI backbone adds local morphology/novelty evidence; it is not the sole decision-maker.
6. A MumGuard-specific research decision layer fuses the available evidence and can abstain instead of guessing.
7. The result is explainable through the score, side, channel values, maps and report.

If asked specifically about DINOv2:

> DINOv2 is a Meta visual foundation model used as the visual feature backbone. We did not claim to invent DINOv2. The MumGuard-specific work is the Contact-TLC signal pipeline, bilateral reconstruction, thermal evidence, evidence fusion, decision contract, APIs, UI, persistence, report and deployment/runtime integration around it.

## 5. What the score means

**Research Concern Score is not a cancer probability.**

It summarizes the strength/concordance of the current MumGuard research evidence. Do not say `73/100 = 73% chance of cancer`.

Current product language:

- Tumor-like pattern detected
- No tumor-like pattern detected
- Inconclusive

Clinical sensitivity/specificity and a calibrated probability require outcome-linked target-device human validation and are a later validation layer, not a missing web-app feature.

## 6. Scope freeze for this delivery

Do not start another research lane before delivery unless final QA finds a blocking defect.

Not blockers for this delivery:

- another public-dataset hunt;
- another model family;
- async job architecture while current runtime remains acceptable;
- new mouse experiments;
- a clinical cancer-probability model;
- prospective clinical validation.

The remaining delivery work is only:

- merge the client-facing UI after CI is green;
- ensure the deployed server is running the merged `integration` revision;
- run final A/B/C engineering smoke cases if available;
- capture one clean live result/report as delivery evidence;
- send the link and a short voice-note explanation.

## 7. Suggested client voice-note script

> بصي يا مريم، أنا خلصت النسخة اللي كنا بنتكلم عليها من MumGuard، وحبيت أشرحلك ببساطة أنا عملت فيها إيه بدل ما أبعتلك لينك وخلاص. الصور اللي بتطلع من الـLiquid Crystal عندكم ما تعاملتش معاها كصور عادية؛ بنيت processing مخصوص للـContact-TLC، والسيستم بياخد صور الشمال واليمين ويركب كل ناحية ويحسب الاستجابة الحرارية النسبية. بعد كده بيقيس الـfocused hotspot-like response، الفرق بين الشمال واليمين، وشكل توزيع الحرارة. وضفت visual AI علشان يلقط patterns محلية تكمل التحليل الحراري، وبعدها decision layer مخصوص لـMumGuard بيجمع الأدلة ويطلع Tumor-like أو No tumor-like، ولو القياس نفسه مش كفاية بيقول Inconclusive بدل ما يخمن. النتيجة بتطلع معاها Research Concern Score، الناحية اللي فيها أقوى evidence، maps بتوضح اللي السيستم شافه، وتقرير كامل. الجزء السريري بمعنى نسبة سرطان معتمدة محتاج validation بشرية لاحقًا، لكن وظيفة التحليل والقرار البحثي نفسها شغالة end-to-end دلوقتي.

## 8. Delivery acceptance checklist

Before sending the client link, verify:

- page loads from the public server;
- LEFT and RIGHT files can be added;
- Analyze examination completes;
- primary finding renders in plain language;
- Research Concern Score renders correctly or shows unavailable without error;
- side + three evidence channels render;
- maps load;
- full report opens;
- Technical details remain collapsed by default;
- no raw `ABSTAIN_OOD`, DMR, DINO or calibration jargon dominates the primary result;
- page works on a narrow/mobile viewport;
- `/health` is healthy;
- deployed revision matches the intended merged `integration` commit.

Once those checks pass, this phase is ready to hand over as the MumGuard research-analysis product implementation.