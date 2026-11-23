"""Build an auditable scorecard from executed comparisons and saved retrieval runs."""
import csv
from collections import defaultdict
from pathlib import Path
import yaml
from utils.tracking.common import ROOT


LABELS = dict(tracklet_specialist='Local tracklets + global TransReID memory', tracklet_joint='Local tracklets + global joint memory', bytetrack='ByteTrack (official association)', ocsort='OC-SORT (official association)',
    legacy_joint='Existing tracker + joint features', memory_joint='Persistent memory + joint features',
    legacy_specialist='Existing tracker + TransReID', memory_specialist='Persistent memory + TransReID',
    memory_specialist_unguarded='Persistent memory + TransReID, guards disabled')


def rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def generate(config):
    output = ROOT / config['output']
    scores = rows(output / 'summary.csv')
    recovery = rows(output / 'recovery.csv')
    totals = defaultdict(lambda: defaultdict(float))
    count_fields = ['visibility_returns', 'eligible_returns', 'recovered', 'failed', 'censored',
                    'correct_reconnections', 'false_reconnections', 'returns_without_prior_match', 'mask_adapter_drops']
    for row in recovery:
        for key in count_fields:
            totals[row['method']][key] += float(row.get(key) or 0)
    text = ['# Executed vehicle benchmark', '',
        'All nine conventional KITTI MOTS validation sequences, 2,981 frames. Identical fixed Co-DINO candidate detections/masks. '
        'This evaluates association and ReID under a common detector; it is not the Mask R-CNN demo or a hidden-test leaderboard result. '
        'Every exported identity decision is causal. Thresholds were frozen before the run. '
        'Architecture revisions motivated by earlier validation results are development comparisons, not untouched-test claims.', '',
        '| Method | HOTA | AssA | IDF1 | ID switches |', '|---|---:|---:|---:|---:|']
    for row in scores:
        name = row['model']
        text.append(f'| {LABELS[name]} | {float(row["HOTA"]):.2f} | {float(row["AssA"]):.2f} | {float(row["IDF1"]):.2f} | {row["IDSW"]} |')
    lookup = {row['model']: row for row in scores}
    candidate = config.get('candidate', 'memory_specialist')
    change = float(lookup[candidate]['HOTA'])-float(lookup['legacy_specialist']['HOTA'])
    rejected = sum(float(row.get('matched_unknown_observations') or 0) for row in recovery if row['method']==candidate)
    text.extend(['', f'**Measured candidate:** {LABELS[candidate]} has a {change:+.2f} pooled HOTA-point difference from the existing TransReID tracker. '
        f'{int(rejected):,} GT-matched observations have unresolved global identity. '
        'For v2, those observations retain stable local tracklet IDs; for v1 they become separate one-frame IDs.', ''])
    text.extend(['', 'Scores are percentages; lower ID switches is better. Masks for ByteTrack/OC-SORT are assigned to their current '
        'source detections with one-to-one IoU matching. ByteTrack measurements are captured before Kalman smoothing without changing its association. Adapter drops must be zero. These are official association implementations with a common detector and mask adapter, '
        'not reproductions of their original published system scores. V1 memory unknowns use one-frame output IDs. '
        'V2 unresolved global matches retain stable local tracklet IDs in MOT and continuity-recovery scoring. '
        'Global rejection remains explicitly counted; a local ID is not a claimed cross-view match.', '',
        '## Disappearance recovery', '',
        'A return is a gap in ground-truth mask visibility. Recovery requires the previously assigned ID within ten visible observations '
        'after return. Conditional recovery excludes events without a prior matched reference; all-return recovery counts those as misses. '
        'Stream-end incomplete events are censored. Reconnection precision checks the physical GT identity against the ID\'s first matched GT identity after a gap in matched observations.', '',
        '| Method | Recovered / all returns | Prior-reference events | Conditional recovery | Correct / false reconnections | Adapter drops |',
        '|---|---:|---:|---:|---:|---:|'])
    for row in scores:
        name = row['model']; value = totals[name]
        eligible, recovered, returned = (int(value[key]) for key in ('eligible_returns', 'recovered', 'visibility_returns'))
        rate = f'{100*recovered/eligible:.1f}%' if eligible else 'N/A'
        text.append(f'| {LABELS[name]} | {recovered}/{returned} | {eligible} | {rate} | '
            f'{int(value["correct_reconnections"])}/{int(value["false_reconnections"])} | {int(value["mask_adapter_drops"])} |')
    text.extend(['', 'Gap-bucket counts (1–5, 6–15, 16–60, and 61+ frames), per-sequence metrics, and censoring are in `recovery.csv`. '
        'Natural visibility gaps in KITTI are limited; these metrics do not establish unrestricted long-term or cross-camera tracking.', '',
        '## Paired uncertainty', '',
        'The intervals below resample nine whole sequences, pairing candidate and baseline. They describe mean per-sequence HOTA differences '
        'rather than the pooled HOTA above. Nine sequences give limited uncertainty estimates.', '',
        f'| {LABELS[candidate]} versus | Mean sequence HOTA delta | 95% interval |', '|---|---:|---:|'])
    for row in rows(output / 'paired_intervals.csv'):
        text.append(f'| {LABELS[row["baseline"]]} | {float(row["mean_sequence_HOTA_delta"]):+.2f} | '
                    f'[{float(row["lower95"]):+.2f}, {float(row["upper95"]):+.2f}] |')
    text.extend(['', '## Cross-camera crop retrieval — separate protocol', '',
        'The following results use the same identity-disjoint local VRAI split, not official VRAI test. '
        'They provide measured comparisons with vehicle-ReID methods; differing pretraining data and adaptation are part of each model condition.', '',
        '| Model | Rank-1 | mAP |', '|---|---:|---:|'])
    cross_path = output / 'cross_camera.yaml'
    if cross_path.exists():
        cross = yaml.safe_load(cross_path.read_text())
        for model in cross['models']:
            text.append(f'| {model["model"]} | {100*model["rank1"]:.2f} | {100*model["map"]:.2f} |')
        text.extend(['', 'These four models were re-executed in this run. Each uses its required preprocessing, identical query/gallery manifests, and no final-ID threshold tuning.', '',
            '| Model | Known acceptance, 1% calibration budget | Unknown FAR on final IDs | Known acceptance, 5% budget | Unknown FAR on final IDs |',
            '|---|---:|---:|---:|---:|'])
        for model in cross['models']:
            a, b = [point['final'] for point in model['operating_points']]
            text.append(f'| {model["model"]} | {100*a["known_recovery"]:.2f}% | {100*a["unknown_far"]:.2f}% | {100*b["known_recovery"]:.2f}% | {100*b["unknown_far"]:.2f}% |')
    else:
        for row in rows(ROOT / 'metrics/reid_teacher/final.csv'):
            text.append(f'| {row["model"]} | {float(row["Rank1"]):.2f} | {float(row["mAP"]):.2f} |')
    operating = ROOT / 'metrics/vehicle_reid/cross_camera/operating_points.yaml'
    if operating.exists():
        measured = yaml.safe_load(operating.read_text())
        text.extend(['', 'The full new specialist run reproduces its closed-set scores. The open-set check instead enrolls '
            'one gallery reference per known identity and withholds 126 identities as unknown. Final queries are never used for threshold selection.', '',
            '| Operating point | Correct known acceptance | Wrong known acceptance | Unknown false acceptance |', '|---|---:|---:|---:|'])
        default = measured['default_final']
        entries = [('Original defaults', default)] + [(f'{100*item["empirical_calibration_budget"]:.0f}% empirical calibration budget', item['final'])
                                                       for item in measured['operating_points']]
        for label, value in entries:
            text.append(f'| {label} | {100*value["known_recovery"]:.2f}% | {100*value["wrong_known_rate"]:.2f}% | {100*value["unknown_far"]:.2f}% |')
        text.extend(['', 'These budgets constrain empirical calibration error, not guaranteed final risk. At original defaults, '
            'zero accepted unknown queries still leaves a 95% one-sided upper bound of '
            f'{100*default["unknown_identity_any_accept_upper95"]:.2f}% on accepting at least one query from a new unknown identity. '
            'Identity-cluster bootstrap intervals for known acceptance and identity-level unknown risk bounds are in `operating_points.yaml`. '
            'This single-reference scoring rule omits temporal confirmation and recent-memory weighting, so its selected thresholds do not overwrite video settings.'])
    cost_path = output / 'online_cost.yaml'
    if cost_path.exists():
        cost = yaml.safe_load(cost_path.read_text())
        text.extend(['', '## Live inference cost', '',
            f'Live Co-DINO + RefineMask + TransReID + association + rendering/encoding/checkpoint I/O: '
            f'**{cost["end_to_end_fps"]:.2f} FPS** over {cost["frames"]} frames of sequence {cost["sequence"]} '
            f'on {cost["hardware"]}. Model loading takes {cost["initialization_seconds"]:.2f}s separately; '
            f'peak allocated CUDA memory is {cost["peak_cuda_gib"]:.2f} GiB. '
            'The detector checkpoint matches the fixed comparison cache. This is a single-sequence cost check, '
            'not dataset-wide throughput; it does not establish tracking quality. Details are in `online_cost.yaml`.'])
    text.extend(['', '## Comparison boundaries', '',
        '- Official source revisions and all runtime/evaluation settings are in `run.yaml`.',
        '- Existing adapted McByte++/EdgeTAM scores are in `metrics/tracking/summary.csv`; its previous export can finalize IDs after the sequence, '
        'so it is excluded from the causal headline table.',
        '- Association timing is offline replay cost, not online FPS. Detector extraction and specialist extraction are not added across devices to fabricate throughput.',
        '- The local VRAI split uses official training images with disjoint identities. Published official-test mAP cannot be ranked against it.',
        '- ByteTrack: https://github.com/FoundationVision/ByteTrack ; OC-SORT: https://github.com/noahcao/OC_SORT ; '
        'KITTI MOTS protocol: https://www.cvlibs.net/datasets/kitti/eval_mots.php ; evaluator: https://github.com/JonathonLuiten/TrackEval', '',
        '## Next experiment dictated by these measurements', '',
        'Use `legacy_specialist` versus `legacy_joint` to isolate the specialist, and `memory_specialist` versus `legacy_specialist` to isolate '
        'the new association/memory. A larger backbone or propagation network is justified only after this isolates the failing component. '
        'Improve identity association/recovery at a fixed reconnection error budget, while checking DetA and IDF1 so rejection cannot appear to improve '
        'precision by dropping all difficult objects. Any subsequent threshold/model selection requires a separate development partition; '
        'do not tune repeatedly on these nine reported validation sequences.', ''])
    (output / 'scorecard.md').write_text('\n'.join(text))
    return output / 'scorecard.md'
