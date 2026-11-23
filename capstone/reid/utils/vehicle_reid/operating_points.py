"""Calibrate open-set thresholds once; measure held-out recovery and false accepts."""
import csv
from pathlib import Path
import numpy as np
from scipy.stats import beta
import torch
import yaml
from utils.tracking.common import ROOT, load
from utils.reid_teacher.data import read, partitions
from utils.teacher.assets import sha256


def scores(query, gallery, query_ids, gallery_ids):
    identities = sorted(set(gallery_ids))
    unknown = set(identities[::5])
    known = [identity for identity in identities if identity not in unknown]
    references = torch.stack([gallery[gallery_ids.index(identity)] for identity in known])
    values, indices = (query @ references.T).topk(2, dim=1)
    return dict(similarity=values[:, 0].numpy(), margin=(values[:, 0]-values[:, 1]).numpy(),
        correct=np.array([known[index] == identity for index, identity in zip(indices[:, 0].tolist(), query_ids)]),
        known=np.array([identity not in unknown for identity in query_ids]), identities=np.array(query_ids))


def statistics(data, threshold, margin, confidence=False):
    accept = (data['similarity'] >= threshold) & (data['margin'] >= margin)
    known, correct = data['known'], data['correct']
    unknown = ~known
    accepted_correct = int((accept & known & correct).sum())
    accepted_wrong = int((accept & known & ~correct).sum())
    false_unknown = int((accept & unknown).sum())
    result = dict(threshold=float(threshold), margin=float(margin), known_queries=int(known.sum()),
        unknown_queries=int(unknown.sum()), correct_known=accepted_correct, wrong_known=accepted_wrong,
        rejected_known=int((~accept & known).sum()), false_unknown=false_unknown,
        known_recovery=float(accepted_correct/max(1, known.sum())), unknown_far=float(false_unknown/max(1, unknown.sum())),
        wrong_known_rate=float(accepted_wrong/max(1, known.sum())))
    if confidence:
        # Resample identities, preserving within-identity correlated queries.
        rng = np.random.default_rng(42)
        known_ids = np.unique(data['identities'][known])
        counts = np.array([(int(((data['identities']==identity)&accept&correct).sum()),
                            int((data['identities']==identity).sum())) for identity in known_ids])
        draws = rng.integers(0, len(counts), size=(10000, len(counts)))
        sampled = counts[draws].sum(axis=1)
        low, high = np.quantile(sampled[:, 0]/sampled[:, 1], [.025, .975])
        unknown_ids = np.unique(data['identities'][unknown])
        false_ids = sum(bool((accept[data['identities']==identity]).any()) for identity in unknown_ids)
        # This conservative identity-level risk asks whether ANY query for an
        # unknown identity is accepted, not whether a randomly sampled crop is.
        upper = 1. if false_ids == len(unknown_ids) else beta.ppf(.95, false_ids+1, len(unknown_ids)-false_ids)
        result.update(known_recovery_lower95=float(low), known_recovery_upper95=float(high),
            unknown_identities=len(unknown_ids), falsely_accepted_unknown_identities=false_ids,
            unknown_identity_any_accept_upper95=float(upper))
    return result


def calibrate(data, budget):
    best = None
    for threshold in np.arange(.2, .981, .01):
        for margin in np.arange(0., .201, .01):
            result = statistics(data, threshold, margin)
            # Control both unknown false accepts and wrong known reconnections.
            if result['unknown_far'] > budget or result['wrong_known_rate'] > budget:
                continue
            key = (result['known_recovery'], -result['wrong_known_rate'], -result['unknown_far'], float(threshold), float(margin))
            if best is None or key > best[0]:
                best = key, result
    if best is None:
        return dict(threshold=1.01, margin=1., known_recovery=0., unknown_far=0., wrong_known_rate=0.)
    return best[1]


def run(base=ROOT / 'metrics/vehicle_reid/cross_camera'):
    settings = read()
    data = partitions(settings)
    extracted = {}
    for stage, names in [('calibration', ['calibration_query', 'calibration_gallery']), ('final', ['query', 'gallery'])]:
        state = load(base / stage / 'state.pt')
        for name in names:
            if state['signature']['manifests'][name] != sha256(ROOT / settings['data'][name]):
                raise ValueError('Evaluation manifests changed')
        if stage == 'final' and state['signature']['model'] != model_signature:
            raise ValueError('Calibration and final use different model weights')
        model_signature = state['signature']['model']
        query, gallery = [torch.cat(chunks) for chunks in state['chunks']]
        if len(query) != len(data[names[0]]) or len(gallery) != len(data[names[1]]):
            raise ValueError('Operating point analysis requires full completed extraction, no smoke subsets')
        extracted[stage] = scores(query, gallery, [row['id'] for row in data[names[0]]], [row['id'] for row in data[names[1]]])
    results = []
    for budget in [.01, .05]:
        chosen = calibrate(extracted['calibration'], budget)
        final = statistics(extracted['final'], chosen['threshold'], chosen['margin'], confidence=True)
        results.append(dict(empirical_calibration_budget=budget, calibration=chosen, final=final))
    report = dict(protocol='single reference cross-camera open-set; thresholds chosen on calibration IDs only',
        model=model_signature,
        selection='maximize correct known acceptance with unknown FAR and wrong-known rate within empirical budget',
        warning='Empirical calibration budgets are not statistical guarantees. Identity-level risk upper bounds are reported on final IDs.',
        default_final=statistics(extracted['final'], .82, .08, confidence=True), operating_points=results,
        reference_rule='first gallery crop in manifest per known identity; every fifth sorted ID withheld as unknown')
    (base / 'operating_points.yaml').write_text(yaml.safe_dump(report))
    print(yaml.safe_dump(report), flush=True)
    return report


if __name__ == '__main__':
    run()
