"""Replay identical cached detections through tracking variants and TrackEval."""

import csv
from pathlib import Path
import statistics
import tempfile
import time
import cv2
import torch
import yaml
from pycocotools import mask as rle

from utils.full_frame.tracker import VehicleTracker, disjoint_predictions
from utils.full_frame.mots_evaluation import evaluate
from utils.teacher.settings import read as teacher_settings
from .common import ROOT, read, load, decode
from .mcbyte import McByte


def sequence(settings, teacher, variant, frames, records, destination, check=False):
    if not frames or len(frames) != len(records):
        raise ValueError('Incomplete frame/prediction records')
    if variant not in ('baseline', 'baseline_motion', 'mcbyte_no_masks', 'mcbyte_edgetam'):
        raise ValueError(f'Unknown tracker variant {variant}')
    tracking = teacher['tracking'].copy()
    if variant == 'baseline_motion':
        tracking['appearance_weight'] = 0
    tracker = VehicleTracker(tracking) if variant.startswith('baseline') else McByte(settings, frames, variant.endswith('edgetam'))
    rows, times, teacher_times = [], [], []
    for frame, (path, cached) in enumerate(zip(frames, records)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        record = load(cached)
        prediction = decode(record)
        if check and frame == 2:
            prediction = {key: value[:0] for key, value in prediction.items()}
        teacher_times.append(record['model_ms'])
        if variant.startswith('baseline'):
            prediction = disjoint_predictions(prediction, tracking)
            identities = tracker.update(prediction, frame)
            frame_rows = [(identity, rle.encode(mask.numpy().astype('uint8', order='F')))
                          for identity, mask in zip(identities, prediction['masks'])]
        else:
            indices, identities = tracker.update(prediction, image, frame)
            occupied = torch.zeros(image.shape[:2], dtype=torch.bool)
            frame_rows = []
            for position in sorted(range(len(indices)), key=lambda p: -float(prediction['scores'][indices[p]])):
                mask = prediction['masks'][indices[position], 0].bool() & ~occupied
                if mask.sum() < tracking['min_mask_pixels']:
                    continue
                occupied |= mask
                frame_rows.append((identities[position], rle.encode(mask.numpy().astype('uint8', order='F'))))
        rows.append(frame_rows)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        if frame % 100 == 0:
            print(f'{variant} {path.parent.name}: {frame+1}/{len(frames)}', flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.partial.txt')
    with temporary.open('w') as file:
        for frame, frame_rows in enumerate(rows):
            exported = set()
            for identity, mask in frame_rows:
                identity = identity if variant.startswith('baseline') else tracker.final_identity(identity)
                if identity in exported:
                    raise RuntimeError('Reconnection produced duplicate exported identities in a frame')
                exported.add(identity)
                height, width = mask['size']
                file.write(f'{frame} {identity} 1 {height} {width} {mask["counts"].decode("ascii")}\n')
    temporary.replace(destination)
    if not variant.startswith('baseline') and tracker.manager is not None:
        if tracker.manager.current_iter is not None:
            tracker.manager.current_iter.close()
    return {'frames': len(frames), 'tracking_replay_ms': statistics.mean(times),
            'cached_teacher_ms': statistics.mean(teacher_times),
            'estimated_total_ms': statistics.mean(times) + statistics.mean(teacher_times),
            'hardware': torch.cuda.get_device_name(),
            'timing_note': 'Replay includes image/cache load, RLE decode and association; estimate adds cached teacher time, not a live end-to-end measurement'}


def run(check=False):
    from utils.teacher.assets import sha256
    settings = read()
    teacher, _ = teacher_settings()
    if not torch.cuda.is_available():
        raise RuntimeError('Tracking experiments require CUDA')
    sequences = teacher['data']['val_sequences']
    if check:
        first = int(settings['check_sequence'])
        sequences = [first, next(seq for seq in sequences if seq != first)]
    source = ROOT / teacher['data']['kitti_root']
    manifest = ROOT / settings['cache'] / 'source.yaml'
    benchmark_settings = {key: value for key, value in settings.items()
                          if key not in ('selected', 'cutie_source', 'cutie_revision', 'cutie_weights',
                                         'check_frames', 'check_sequence')}
    provenance = {'cache': yaml.safe_load(manifest.read_text()), 'experiment': benchmark_settings,
                  'edgetam_sha256': sha256(ROOT / settings['edgetam_weights']),
                  'teacher_tracking': teacher['tracking'], 'format': 1}
    with tempfile.TemporaryDirectory(prefix='tracker_check_') as scratch:
        predictions = Path(scratch) / 'predictions' if check else ROOT / settings['predictions']
        output = Path(scratch) / 'metrics' if check else ROOT / settings['metrics']
        predictions.mkdir(parents=True, exist_ok=True)
        marker = predictions / 'run.yaml'
        if marker.is_file() and yaml.safe_load(marker.read_text()) != provenance:
            raise ValueError('Tracking outputs use another config/cache; choose a new predictions folder')
        marker.write_text(yaml.safe_dump(provenance))
        sequence_lengths, timings = {}, []
        for seq in sequences:
            name = f'{seq:04d}'
            frames = sorted((source / 'training/image_02' / name).glob('*.png'))
            if check:
                frames = frames[:settings['check_frames']]
            if not frames or [int(path.stem) for path in frames] != list(range(len(frames))):
                raise ValueError(f'Incomplete frames for {name}')
            records = [ROOT / settings['cache'] / name / f'{path.stem}.pt' for path in frames]
            if not all(path.is_file() for path in records):
                raise FileNotFoundError('Missing fixed teacher predictions; run `python benchmark.py trackers`')
            sequence_lengths[name] = len(frames)
            for variant in settings['variants']:
                destination = predictions / variant / 'data' / f'{name}.txt'
                timing_path = destination.with_suffix('.timing.yaml')
                if destination.is_file() and timing_path.is_file():
                    timing = yaml.safe_load(timing_path.read_text())
                    if timing['frames'] != len(frames):
                        raise ValueError('Cached tracking output has wrong length')
                    print(f'{variant} {name}: reusing completed sequence', flush=True)
                else:
                    timing = sequence(settings, teacher, variant, frames, records, destination, check=check)
                    timing_path.write_text(yaml.safe_dump(timing))
                timings.append({'variant': variant, 'sequence': name, **timing})
        truth_root = source
        if check:
            truth_root = Path(scratch) / 'truth'
            (truth_root / 'instances_txt').mkdir(parents=True)
            for name, count in sequence_lengths.items():
                lines = (source / 'instances_txt' / f'{name}.txt').read_text().splitlines()
                (truth_root / 'instances_txt' / f'{name}.txt').write_text('\n'.join(line for line in lines if int(line.split()[0]) < count) + '\n')
        scores = evaluate(truth_root, predictions, settings['variants'], sequence_lengths, output, print_results=not check)
        if check:
            print(f'Tracker check passed on {len(sequences)} sequences of {settings["check_frames"]} real frames, '
                  'including repeated propagator initialization, an injected empty-detection frame, '
                  'mask refresh, and official mask evaluation. These are not benchmark results.', flush=True)
        else:
            with (output / 'timing.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(timings[0]))
                writer.writeheader(); writer.writerows(timings)
            report = output / 'summary.html'
            report.write_text(report.read_text() +
                '<h2>Experiment conditions</h2><p>Identical teacher checkpoint and raw candidate cache. '
                'baseline_motion disables appearance. McByte variants use upstream association, camera-motion '
                'compensation and cached vehicle embeddings instead of the person ReID network. '
                'mcbyte_no_masks disables temporal mask cues; mcbyte_edgetam enables the released EdgeTAM. '
                'All exported masks come from the teacher, not propagated masks. McByte can recover candidates '
                'above 0.1 confidence; baseline uses its original 0.3 threshold. '
                'McByte export IDs are finalized after the sequence, as in its upstream demo. '
                'This is an adapted system, not a reproduction of published sports benchmark scores.</p>'
                '<p>timing.csv separates replay cost from cached teacher time. Their sum is an estimate '
                'and cannot be claimed as live pipeline latency. Startup and final file export are excluded.</p>')
            print(f'Report: {report}', flush=True)
