"""Full fixed-detector vehicle comparison using upstream trackers and TrackEval."""
import argparse
import csv
from pathlib import Path
import time
import numpy as np
from pycocotools import mask as rle
import torch
import yaml
from utils.tracking.common import ROOT, load, save, decode
from utils.teacher.settings import read as teacher_settings
from utils.full_frame.protocol import VALIDATION
from utils.full_frame.mots_evaluation import evaluate, TRACKEVAL_REVISION
from .baselines import prepare_sources, imports, Method
from .models import observations as quality_observations
from .recovery import Recovery
from .extraction import extract
from .tracklets import tracking_identity


def emitted_masks(raw, emitted, frame, minimum):
    occupied = torch.zeros(raw['masks'].shape[-2:], dtype=torch.bool)
    masks, decisions, rows = [], [], []
    used = set()
    for index, decision in sorted(emitted, key=lambda item: -float(raw['scores'][item[0]])):
        mask = raw['masks'][index, 0].bool() & ~occupied
        if int(mask.sum()) < minimum:
            continue
        identity = decision['identity']
        if identity is None:
            if 'tracklet_id' in decision:
                identity = tracking_identity(decision)
                decision = {**decision, 'identity': identity, 'global_identity': None}
            else:
                identity = 1_000_000_000+frame*1000+index
        if identity in used:
            raise ValueError('Duplicate identity emitted within a frame')
        used.add(identity)
        occupied |= mask
        array = mask.numpy()
        masks.append(array)
        decisions.append(decision)
        encoded = rle.encode(np.asfortranarray(array.astype(np.uint8)))
        height, width = array.shape
        rows.append(f'{frame} {identity} 1 {height} {width} {encoded["counts"].decode()}\n')
    return masks, decisions, ''.join(rows)


def tracking_settings(pipeline):
    # Live renderer/output/backend settings do not change fixed-cache association.
    return {key: pipeline[key] for key in ('reid', 'quality', 'association')}


def sequence(config, pipeline, sequence_name, signature):
    frames = sorted((ROOT / config['data_root'] / 'training/image_02' / sequence_name).glob('*.png'))
    features = load(ROOT / config['features'] / sequence_name / 'state.pt')
    if not features['complete'] or len(features['observations']) != len(frames):
        raise ValueError(f'Incomplete specialist extraction for {sequence_name}')
    legacy = teacher_settings()[0]['tracking']
    methods = {name: Method(name, config, pipeline, legacy) for name in config['methods']}
    truth = ROOT / config['data_root'] / 'instances_txt' / f'{sequence_name}.txt'
    recovery = {name: Recovery(truth, config['recovery_window']) for name in methods}
    checkpoint = ROOT / config['predictions'] / 'state' / f'{sequence_name}.pt'
    seconds, begin = {}, 0
    paths = {name: ROOT / config['predictions'] / name / 'data' / f'{sequence_name}.txt' for name in methods}
    offsets = {name: 0 for name in methods}
    if checkpoint.exists():
        previous = load(checkpoint)
        previous_signature = dict(previous['signature'])
        previous_signature['pipeline'] = tracking_settings(previous_signature['pipeline'])
        previous_signature['config'] = dict(previous_signature['config'])
        previous_signature['config'].pop('pipeline_config',None)
        if previous_signature != signature or previous['feature_signature'] != features['signature']:
            raise ValueError(f'Benchmark inputs/settings changed: {sequence_name}; choose new output paths')
        first_method = config['methods'][0]
        if previous['recovery'][first_method].truth != recovery[first_method].truth:
            raise ValueError(f'Ground truth changed for {sequence_name}; use a new prediction directory')
        if previous['complete']:
            print(f'{sequence_name}: reusing completed metrics predictions', flush=True)
            return len(frames), previous['recovery_summary'], previous['seconds']
        methods, recovery, offsets, begin = (previous[key] for key in ('methods', 'recovery', 'offsets', 'next_frame'))
        seconds = previous['seconds']
        from yolox.tracker.basetrack import BaseTrack
        from trackers.ocsort_tracker.ocsort import KalmanBoxTracker
        BaseTrack._count = previous['upstream_counters']['byte']
        KalmanBoxTracker.count = previous['upstream_counters']['oc']
    handles = {}
    try:
        for name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if begin:
                if not path.exists() or path.stat().st_size < offsets[name]:
                    raise ValueError(f'Committed prediction data missing: {path}')
                handle = path.open('r+b')
                handle.truncate(offsets[name]); handle.seek(offsets[name])
            else:
                handle = path.open('wb')
            handles[name] = handle
        for frame in range(begin, len(frames)):
            raw = decode(load(ROOT / config['cache'] / sequence_name / f'{frame:06d}.pt'))
            observation = features['observations'][frame]
            indices = observation['indices']
            height, width = raw['masks'].shape[-2:]
            _, _, _, reliable = quality_observations(np.empty((height, width, 0), dtype=np.uint8),
                raw['boxes'][indices].numpy(), raw['scores'][indices].numpy(), raw['masks'][indices, 0].bool().numpy(), pipeline['quality'])
            for name, method in methods.items():
                start = time.perf_counter()
                emitted = method.update(raw, observation, reliable, frame)
                seconds[name] = seconds.get(name, 0.) + time.perf_counter()-start
                masks, decisions, text = emitted_masks(raw, emitted, frame, config['min_mask_pixels'])
                handles[name].write(text.encode('ascii'))
                recovery[name].update(frame, masks, decisions)
            if (frame+1) % config['checkpoint_every'] == 0 or frame+1 == len(frames):
                for handle in handles.values():
                    handle.flush()
                summary = {name: {**value.summary(), 'mask_adapter_drops': methods[name].mapping_drops}
                           for name, value in recovery.items()}
                from yolox.tracker.basetrack import BaseTrack
                from trackers.ocsort_tracker.ocsort import KalmanBoxTracker
                save(dict(signature=signature, feature_signature=features['signature'], methods=methods,
                    upstream_counters=dict(byte=BaseTrack._count, oc=KalmanBoxTracker.count),
                    recovery=recovery, recovery_summary=summary, offsets={name: handle.tell() for name, handle in handles.items()},
                    next_frame=frame+1, seconds=seconds, complete=frame+1 == len(frames)), checkpoint)
                print(f'Association {sequence_name}: {frame+1}/{len(frames)} across {len(methods)} methods', flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    return len(frames), summary, seconds


def paired_intervals(config):
    """Paired sequence bootstrap, explicitly different from pooled TrackEval HOTA."""
    names = config['methods']
    output = ROOT / config['output']
    scores = {}
    for name in names:
        with (output / name / 'car_detailed.csv').open() as file:
            rows = list(csv.DictReader(file))
        scores[name] = {row['seq']: float(row['HOTA___AUC'])*100 for row in rows if row['seq'] != 'COMBINED'}
    selected = scores[names[0]].keys()
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(selected), size=(10000, len(selected)))
    intervals = []
    candidate = config.get('candidate', 'memory_specialist')
    for name in names:
        if name == candidate:
            continue
        deltas = np.array([scores[candidate][seq]-scores[name][seq] for seq in selected])
        bootstrap = deltas[draws].mean(1)
        low, high = np.quantile(bootstrap, [.025, .975])
        intervals.append(dict(candidate=candidate, baseline=name,
            mean_sequence_HOTA_delta=float(deltas.mean()), lower95=float(low), upper95=float(high),
            unit='percentage points; paired sequence bootstrap; mean sequence HOTA, not pooled HOTA', sequences=len(selected)))
    with (output / 'paired_intervals.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(intervals[0]))
        writer.writeheader(); writer.writerows(intervals)
    return intervals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/vehicle_benchmark_tracklets.yaml')
    args = parser.parse_args(argv)
    config = yaml.safe_load((ROOT / args.config).read_text())
    if config['protocol'] != 'kitti_mots_standard_val_v1' or sorted(config['sequences']) != list(VALIDATION):
        raise ValueError('Full benchmark requires all nine conventional validation sequences')
    torch.set_num_threads(config['cpu_threads'])
    pipeline = yaml.safe_load((ROOT / config['pipeline_config']).read_text())
    prepare_sources(config)
    imports(config)  # Ensure dependencies and pickle module paths exist before resuming.
    extract(config)
    association_config = {key: value for key, value in config.items() if key not in ('cross_camera','pipeline_config')}
    signature = dict(config=association_config, pipeline=tracking_settings(pipeline),
        detector=yaml.safe_load((ROOT / config['cache'] / 'source.yaml').read_text()),
        evaluator=TRACKEVAL_REVISION)
    lengths, recovery_rows, timing_rows = {}, [], []
    for value in config['sequences']:
        name = f'{value:04d}'
        count, recovery, seconds = sequence(config, pipeline, name, signature)
        lengths[name] = count
        for method in config['methods']:
            recovery_rows.append(dict(method=method, sequence=name, **recovery[method]))
            timing_rows.append(dict(method=method, sequence=name, frames=count, association_seconds=seconds[method],
                scope='association only on fixed offline observations; NOT online FPS'))
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    scores = evaluate(ROOT / config['data_root'], ROOT / config['predictions'], config['methods'], lengths, output)
    for filename, rows in [('recovery.csv', recovery_rows), ('association_timing.csv', timing_rows)]:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / filename).open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    intervals = paired_intervals(config)
    from utils.teacher.assets import sha256
    summary = dict(protocol=config['protocol'], frames=sum(lengths.values()), sequences=lengths,
        pipeline_config=config['pipeline_config'],
        specialist_selection=yaml.safe_load((ROOT / pipeline['reid']['selection']).read_text()),
        truth_sha256={seq: sha256(ROOT / config['data_root'] / 'instances_txt' / f'{seq}.txt') for seq in lengths},
        implementation_sha256={name: sha256(Path(__file__).parent / name) for name in
            ('association.py', 'tracklets.py', 'baselines.py', 'benchmark.py', 'recovery.py', 'crosscamera_benchmark.py')},
        conditions=signature, scores=scores, paired_intervals=intervals,
        caveats=['Fixed Co-DINO detection replay; not the Mask R-CNN demo backend',
            'Every decision is causal; IDs are never retrospectively merged',
            'ByteTrack and OC-SORT use upstream association with common mask mapping, not their published detector',
            'V1 memory unknowns use one-frame IDs; v2 unresolved identities retain stable local tracklet IDs',
            'Global identity rejection is counted separately from local tracking continuity',
            'This architecture revision is a development comparison motivated by the previous validation results',
            'All thresholds were frozen before this run; these results do not tune thresholds',
            'Local conventional validation scores are not KITTI hidden-test leaderboard scores',
            'Replay/association/extraction costs are not live end-to-end speed'])
    (output / 'run.yaml').write_text(yaml.safe_dump(summary))
    if config.get('cross_camera', True):
        from .crosscamera_benchmark import run as crosscamera
        crosscamera(config)
    from .scorecard import generate
    print(f'Full comparison: {generate(config)}', flush=True)
