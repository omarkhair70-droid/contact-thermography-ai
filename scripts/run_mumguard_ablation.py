"""Execute bounded 8/1 development experiments; fail closed on missing contact."""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from app.services.contact_field import normalize_field
from app.services.client_device_domain import analyze_client_image
from app.services.local_contrast_features import extract, schema
from app.services.mumguard_research_runtime import analyze_research, persist_evidence
from research.mumguard.fit_subject_heads import FEATURES, features, fit, score
from scripts.validate_mumguard_acquisitions import validate
from scripts.stress_mumguard_evidence import conservative_support, variants


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def folds(data, y, ids):
    rows = []
    for test in np.flatnonzero(y == 1):
        train = np.arange(len(y)) != test
        model = fit(data[train], y[train])
        held = float(score(model, data[test:test+1])[0])
        neg = float(score(model, data[y == 0])[0])
        rows.append({'held_out_subject': ids[test], 'train_subjects': list(np.array(ids)[train]),
            'held_out_score': held, 'negative_training_score': neg, 'margin': held-neg,
            'independent_negative_tests': 0, 'model': model})
    return rows


def local_vector(result):
    base = features(result)
    records = sorted(result['candidates'], key=lambda r: r['physical_contrast'], reverse=True)
    selected = []
    for row in records:
        x1, y1, x2, y2 = row['x']-row['size']/2, row['y']-row['size']/2, row['x']+row['size']/2, row['y']+row['size']/2
        if all(x2 <= a[0] or a[2] <= x1 or y2 <= a[1] or a[3] <= y1 for a in selected):
            selected.append((x1, y1, x2, y2, row['physical_contrast']))
        if len(selected) == 3:
            break
    physical = np.asarray([r['physical_contrast'] for r in records])
    return np.concatenate([base, [physical.mean(), physical.max(), np.mean([r[4] for r in selected])]])


def run(args):
    started = time.perf_counter()
    ledger = validate(args.manifest, args.image_root)
    ledger.sort(key=lambda r: r['subject_id'])
    ids = [r['subject_id'] for r in ledger]
    y = np.array([int(r['lesion_presence'] == 'present') for r in ledger])
    baseline = np.load(args.baseline, allow_pickle=False)
    if baseline.shape != (9, 384) or baseline.dtype != np.float32 or not np.isfinite(baseline).all():
        raise ValueError('baseline requires finite 9 x 384 float32')
    # Row identity comes from the supplied artifact summary, never an assumed sort.
    summary = list(csv.DictReader(args.baseline.with_name('client_device_dinov2_summary.csv').open()))
    filenames = [r.get('image', r.get('source_image', '')) for r in summary]
    if filenames != [r['filename'] for r in ledger]:
        raise ValueError(f'baseline row provenance mismatch: {filenames}')
    if (np.linalg.norm(baseline, axis=1) < 1e-9).any():
        raise ValueError('zero baseline vector')
    baseline = baseline / np.linalg.norm(baseline, axis=1, keepdims=True)
    args.out.mkdir(parents=True, exist_ok=True)
    args.private_out.mkdir(parents=True, exist_ok=True)
    protocol = {'version': 'mumguard-local-v1-exploratory-2', 'supersedes': 'protocol_v1_initial.json',
        'revision_reason': 'Initial conservative optical support yielded no usable comparator in five subjects; add an explicitly unknown-contact full-field diagnostic without changing clinical eligibility.',
        'development_subjects': ids,
        'contact_certainty': 'UNKNOWN', 'E1': 'BLOCKED_MISSING_INDEPENDENT_CONTACT_ANNOTATIONS',
        'scales': [16, 32, 64], 'minimum_coverage': .8, 'minimum_pixels': 32,
        'head': 'class-balanced subject logistic; ridge=1; fold-local scaler',
        'normal_anchor_arm': '0038 global cosine distance, provenance-visible; compared with label-free self-comparison and never treated as multiple negatives',
        'selection': 'provenance/contact eligibility, shortcuts, stability/coverage, then >=6/8 improved margins and positive median; simpler otherwise',
        'support_arms': {
            'conservative_chromatic': 'label-free conservative chromatic optical support; NOT confirmed tissue contact or healthy tissue',
            'unknown_full_field': 'complete captured field diagnostic; contact remains UNKNOWN and setup/background can contribute'},
        'threshold': None, 'clinical_claim': 'NONE'}
    write(args.out/'protocol.json', protocol)  # frozen before inspecting any candidate outcomes
    write(args.out/'feature_schema.json', schema())
    local = {name: [] for name in protocol['support_arms']}
    global_new, morphology, stress, receipt, local_rows = [], [], [], [], []
    global_perturbations = {}
    token_provenance = []
    for row in ledger:
        sid = row['subject_id']
        source = (args.image_root/row['filename']).read_bytes()
        rgb, _, em, frame, geometry = normalize_field(source)
        metadata = {**row, 'acquisition_type': 'contact-LCT', 'calibration_status': 'UNCALIBRATED'}
        missing, arrays = analyze_research(source, metadata)
        persist_evidence(missing, arrays, args.private_out/sid/'missing-contact')
        tokens = tp = None
        if args.dino:
            import torch
            torch.set_num_threads(4)
            from app.services.dinov2_patch_encoder import encode_patches
            from app.services.dinov2_service import runtime
            tokens, tp = encode_patches(rgb)
            np.savez_compressed(args.private_out/sid/'tokens.npz', tokens=tokens, **tp)
            global_new.append(runtime.encode_rgb(rgb))
            token_provenance.append({'subject_id': sid, **tp})
        support_arms = {'conservative_chromatic': conservative_support(rgb) & frame,
                        'unknown_full_field': frame}
        arm_results = {}
        for arm, support in support_arms.items():
            result, heat, counts = extract(rgb, support, em, tokens)
            result.update({'provenance': {**metadata, **geometry, 'tokens': tp},
                'contact_certainty': 'UNKNOWN', 'support_semantics': protocol['support_arms'][arm],
                'support_arm': arm, 'disease_eligible': False,
                'validation_status': 'UNKNOWN_CONTACT_DIAGNOSTIC_ONLY'})
            persist_evidence(result, {'rgb': rgb, 'contact': np.zeros_like(support),
                'provisional_optical_support': support, 'exclusion': em,
                'local_contrast': heat, 'support': counts}, args.private_out/sid/arm)
            vector = local_vector(result) if result['candidates'] else np.full(9, np.nan)
            local[arm].append(vector)
            arm_results[arm] = result
            local_rows.append({'subject_id': sid, 'support_arm': arm,
                'candidate_count': result['candidate_count'], 'support_fraction': float(support.mean()),
                'analyzed_support_fraction': result['analyzed_contact_fraction'],
                **{name: (float(value) if np.isfinite(value) else None) for name, value in zip(
                    FEATURES+['physical_mean', 'physical_max', 'physical_top3_nonoverlap_mean'], vector)}})
        client_features, _, _ = analyze_client_image(args.image_root/row['filename'])
        valid_area = max(1, geometry['resize_dimensions'][0]*geometry['resize_dimensions'][1])
        morphology.append([client_features['response_area_fraction'], client_features['component_count'],
            client_features['largest_component_fraction'], client_features['largest_elongation'],
            client_features['skeleton_length_px']/valid_area, client_features['branch_pixels']/valid_area])
        one = {'subject_id': sid, 'source_sha256': row['source_sha256'],
            'contact_certainty': 'UNKNOWN', 'missing_contact_status': missing['quality_reasons'],
            'arms': {name: {'candidate_count': result['candidate_count'],
                'local_contrast_score': result['local_contrast_score'],
                'analyzed_support_fraction': result['analyzed_contact_fraction'],
                'dino_candidate_count': sum(r['dino_local_cosine_distance'] is not None for r in result['candidates'])}
                for name, result in arm_results.items()}}
        receipt.append(one)
        for name, perturbed in variants(rgb).items():
            for arm, mask in {'conservative_chromatic': conservative_support(perturbed) & frame,
                              'unknown_full_field': frame}.items():
                e, _, _ = extract(perturbed, mask, em)
                stress.append({'subject_id': sid, 'support_arm': arm, 'variant': name,
                    'score': e['local_contrast_score'], 'candidate_count': e['candidate_count'],
                    'analyzed_support_fraction': e['analyzed_contact_fraction'], 'independent_contact': False})
            if args.dino:
                global_perturbations.setdefault(name, []).append(runtime.encode_rgb(perturbed))
        if args.dino:
            # Matched area occlusions probe sensitivity, not lesion causality.
            for label, x, yy in [('center_occlusion', 84, 84), ('corner_occlusion', 28, 28)]:
                occluded = rgb.copy(); occluded[yy:yy+56, x:x+56] = 127
                global_perturbations.setdefault(label, []).append(runtime.encode_rgb(occluded))
        print(json.dumps({'subject': sid, 'candidates': {k: v['candidate_count'] for k, v in arm_results.items()}, 'dino': tokens is not None}), flush=True)
    negative_anchor_distance = (1-np.clip(baseline @ baseline[-1], -1, 1)).reshape(-1, 1)
    blocks = {'A_saved_global': baseline, 'M_response_morphology': np.asarray(morphology),
              'D_0038_global_anchor': negative_anchor_distance}
    for arm, values in local.items():
        X = np.asarray(values)
        blocks.update({f'B_physical_{arm}': X[:, :1], f'C_fixed_pool_{arm}': X[:, :2],
            f'C_mean_{arm}': X[:, 6:7], f'C_max_{arm}': X[:, 7:8],
            f'C_top3_{arm}': X[:, 8:9], f'F_photometric_{arm}': X[:, :5],
            f'B_dino_{arm}': X[:, 5:6], f'F_local_fusion_{arm}': X[:, :6]})
        blocks[f'F_photometric_0038_anchor_{arm}'] = np.concatenate([X[:, :5], negative_anchor_distance], axis=1)
    if global_new:
        blocks['A_reencoded_global'] = np.array(global_new)
    fold_results, models, skipped = {}, {}, {}
    for name, data in blocks.items():
        if not np.isfinite(data).all():
            missing_ids = [ids[i] for i in range(9) if not np.isfinite(data[i]).all()]
            skipped[name] = {'reason': 'MISSING_USABLE_COMPARATOR_OR_DINO; no imputation', 'subjects': missing_ids}
            continue
        fold_results[name] = folds(data, y, ids)
        models[name] = fit(data, y)
    permutations = []
    for neg in range(9):
        permuted = np.ones(9, int); permuted[neg] = 0
        for name, data in blocks.items():
            if name not in models:
                continue
            margins = [r['margin'] for r in folds(data, permuted, ids)]
            permutations.append({'sole_negative_assignment': ids[neg], 'candidate': name,
                'positive_margins': sum(v > 0 for v in margins), 'median_margin': float(np.median(margins))})
    global_stress = []
    if global_new:
        for name, encoded in global_perturbations.items():
            for i, r in enumerate(fold_results['A_reencoded_global']):
                value = float(score(r['model'], np.array(encoded)[i:i+1])[0])
                global_stress.append({'subject_id': ids[i], 'variant': name, 'held_out_score': value,
                    'delta_from_identity': value-r['held_out_score'], 'contact_certainty': 'UNKNOWN'})
    # Reproduce the actual saved linear artifact on its existing embeddings.
    legacy_path = ROOT/'artifacts/lct_native_exp8v1_dino.json'
    legacy = json.loads(legacy_path.read_text())
    legacy_scores = 1/(1+np.exp(-(baseline @ np.array(legacy['coefficients'])+legacy['intercept'])))
    write(args.out/'saved_baseline_reproduction.json', {
        'artifact_sha256': hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
        'role': 'resubstitution only; no independent validation',
        'rows': [{'subject_id': sid, 'computed_score': float(v), 'stored_score': saved['score'],
                  'absolute_difference': abs(float(v)-saved['score'])}
                 for sid, v, saved in zip(ids, legacy_scores, legacy['training']['resubstitution_scores'])]})
    comparisons = {}
    base = np.array([r['margin'] for r in fold_results['A_saved_global']])
    for name, rows in fold_results.items():
        diff = np.array([r['margin'] for r in rows])-base
        comparisons[name] = {'improved_paired_margins': int((diff > 0).sum()), 'median_improvement': float(np.median(diff)),
            'contact_eligibility_pass': None if name.startswith('A_') else False,
            'selection_eligible': False,
            'shortcut_clearance': 'NOT_ESTABLISHED_WITHOUT_CONTACT_TRUTH' if not name.startswith('A_') else 'NOT_CLEARED; SETUP_ARTIFACT_PROBES_REPORTED'}
    card = {'name': 'mumguard-local-evidence', 'feature_contract_version': 'mumguard-local-v1',
        'status': 'INACTIVE_RESEARCH_CANDIDATE', 'clinical_claim': 'NONE', 'clinical_use': False,
        'selected_architecture': 'label-free local sensing with explicit abstention', 'selected_disease_head': None,
        'selection_reason': 'No disease candidate clears contact/provenance and shortcut gates; retain reusable sensing per program kill criteria.',
        'validation_status': 'EXPLORATORY_DEVELOPMENT_ONLY_SINGLE_NEGATIVE',
        'target_species': 'human', 'development_challenge_species': 'mouse',
        'human_binary_inference': 'DISABLED_PENDING_HUMAN_HEAD_AND_CALIBRATION', 'threshold': None,
        'subjects': ids, 'positive_subjects': 8, 'negative_subjects': 1, 'comparisons': comparisons,
        'skipped_candidates': skipped, 'negative_holdout': 'SKIPPED_BINARY_FIT_NO_NEGATIVE_IN_TRAIN',
        'E1': protocol['E1'],
        'E2': 'EXECUTED_WHERE_COMPARATORS_EXIST; COMPLETE-COHORT_HEADS_SKIP_MISSING_CHANNELS',
        'E3': 'GLOBAL_OCCLUSION_STRESS_EXECUTED; TRUE_CONTACT_VS_BACKGROUND_BLOCKED',
        'E4': 'FIXED_Q90_MEDIAN_MEAN_MAX_TOP3_POOLING_EXECUTED_ON_UNKNOWN_FULL_FIELD',
        'E5': '0038 ANCHOR AND LABEL_FREE INTERNAL COMPARISON EXECUTED SEPARATELY; ANCHOR DELETION REMOVES THAT CANDIDATE',
        'E6': 'LABEL_FREE_SELF_COMPARISON_EXECUTED',
        'E7': 'RESPONSE_MORPHOLOGY, DINO, LOCAL_PHOTOMETRY, AND AVAILABLE_FUSION COMPARED',
        'E8': 'NOT_EXECUTED_NO_GENUINE_REPEAT_PAIRS; ADAPTER REMAINS IDENTITY',
        'E9': 'HUMAN_FIRST_INTAKE_ADAPTATION_HEAD_CALIBRATION CONTRACTS READY; NO TARGET HUMAN OUTCOMES',
        'local_crop_ablation': 'NOT_EXECUTED_NO_CREDIBLE_CONTACT_ROI_RING_PAIRS',
        'baseline_sha256': hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'private_assets_committed': False, 'runtime_seconds': time.perf_counter()-started,
        'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'device': 'cpu'}}
    write(args.out/'folds.json', fold_results)
    write(args.out/'development_heads.json', {'activation_status': 'INACTIVE_DIAGNOSTIC_ONLY', 'models': models, 'threshold': None})
    write(args.out/'acquisition_receipts.json', receipt)
    with (args.out/'local_evidence_by_subject.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(local_rows[0]))
        writer.writeheader(); writer.writerows(local_rows)
    write(args.out/'perturbation_report.json', {'semantics': 'engineering stress; not confidence intervals, causal localization, or genuine device generalization', 'rows': stress, 'global_held_out_probes': global_stress})
    if args.dino:
        checkpoint = Path(os.environ['TORCH_HOME'])/'hub/checkpoints/dinov2_vits14_pretrain.pth'
        write(args.out/'weight_provenance.json', {'filename': checkpoint.name,
            'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'size_bytes': checkpoint.stat().st_size,
            'backbone': 'dinov2_vits14', 'revision': token_provenance[0]['revision']})
    write(args.out/'permutation_report.json', {'assignments': 9, 'rows': permutations, 'minimum_unrandomized_p': 1/9, 'population_significance_claim': None})
    write(args.out/'token_provenance.json', token_provenance)
    write(args.out/'model_card.json', card)
    label_free = {}
    for arm, values in local.items():
        values = np.asarray(values)
        label_free[arm] = {
            'available_subjects': [ids[i] for i in range(9) if np.isfinite(values[i, 0])],
            'missing_subjects': [ids[i] for i in range(9) if not np.isfinite(values[i, 0])],
            'physical_q90': {ids[i]: (float(values[i, 0]) if np.isfinite(values[i, 0]) else None) for i in range(9)},
            'dino_q90': {ids[i]: (float(values[i, 5]) if np.isfinite(values[i, 5]) else None) for i in range(9)},
            'normal_bank_used': False, 'disease_labels_used_to_extract': False,
            'contact_certainty': 'UNKNOWN'}
    write(args.out/'label_free_summary.json', label_free)
    ablation_rows = []
    for name in blocks:
        rows = fold_results.get(name)
        ablation_rows.append({'candidate': name, 'execution_status': 'FITTED_DIAGNOSTIC' if rows else 'SKIPPED_MISSING_CHANNEL',
            'held_out_positive_folds': len(rows or []),
            'positive_margins': sum(r['margin'] > 0 for r in (rows or [])),
            'median_margin': float(np.median([r['margin'] for r in rows])) if rows else None,
            'improved_vs_saved_A': comparisons.get(name, {}).get('improved_paired_margins'),
            'contact_certainty': 'NOT_APPLICABLE_GLOBAL' if name.startswith('A_') else 'UNKNOWN',
            'selection_eligible': False, 'independent_negative_tests': 0,
            'clinical_claim': 'NONE'})
    with (args.out/'ablation_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ablation_rows[0]))
        writer.writeheader(); writer.writerows(ablation_rows)
    with (args.out/'subject_scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['subject_id', 'candidate', 'role', 'score', 'negative_training_score', 'margin'])
        writer.writeheader()
        for name, rows in fold_results.items():
            for r in rows:
                writer.writerow({'subject_id': r['held_out_subject'], 'candidate': name, 'role': 'exploratory_held_out_positive',
                    'score': r['held_out_score'], 'negative_training_score': r['negative_training_score'], 'margin': r['margin']})
            writer.writerow({'subject_id': ids[-1], 'candidate': name, 'role': 'all_subject_fit_negative_NOT_TEST',
                'score': float(score(models[name], blocks[name][-1:])[0])})
    report = [
        '# MumGuard executable experiment result', '',
        'Target architecture: human-first local evidence sensing. Mouse labels are used only for the bounded development challenge.', '',
        f'- Originals verified: {len(ids)} (8 tumor-bearing, 1 healthy)',
        f'- E1: {protocol["E1"]}',
        f'- Fitted diagnostic candidates: {len(models)}; skipped incomplete candidates: {len(skipped)}',
        '- Contact certainty: UNKNOWN for every source; no whole-frame or optical-support pixel was relabeled as confirmed tissue.',
        '- Selected disease head: none. Runtime decision remains ABSTAIN and registry status remains inactive.', '',
        'See `subject_scores.csv`, `local_evidence_by_subject.csv`, `folds.json`, `perturbation_report.json`, and `model_card.json` for exact rows and provenance.'
    ]
    (args.out/'EXECUTION_SUMMARY.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(json.dumps(card, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT/'data/mumguard_acquisitions_v1.jsonl')
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--out', type=Path, default=ROOT/'artifacts/mumguard-local-v1')
    p.add_argument('--private-out', type=Path, required=True)
    p.add_argument('--dino', action='store_true')
    p.add_argument('--acknowledge-development-only', action='store_true', required=True)
    run(p.parse_args())
