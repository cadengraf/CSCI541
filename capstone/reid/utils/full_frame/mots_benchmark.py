"""Export every validation frame, track vehicles, and run official TrackEval."""

import hashlib
import csv
from pathlib import Path
import re
import statistics
import time

import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor
import yaml

from .model import FullFrameVehicleNet
from .mots_evaluation import evaluate, TRACKEVAL_REVISION
from .protocol import CONFIG, check_checkpoint, load
from .tracker import VehicleTracker, disjoint_predictions


def write_frame(file, frame, prediction, identities):
    from pycocotools import mask as mask_tools
    for mask, identity in zip(prediction['masks'], identities):
        array = np.asfortranarray(mask.numpy().astype(np.uint8))
        encoded = mask_tools.encode(array)
        height, width = array.shape
        file.write(f'{frame} {identity} 1 {height} {width} {encoded["counts"].decode("ascii")}\n')


@torch.inference_mode()
def infer_sequence(model, folder, destination, settings, device, frame_limit=None):
    frames = sorted(Path(folder).glob('*.png'))
    if frame_limit is not None:
        frames = frames[:frame_limit]
    if not frames or [int(frame.stem) for frame in frames] != list(range(len(frames))):
        raise ValueError(f'Missing frames or nonzero starting frame: {folder}')
    tracker = VehicleTracker(settings)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.partial.txt')
    times = []
    with temporary.open('w') as file:
        for frame in frames:
            start = time.perf_counter()
            with Image.open(frame) as raw:
                image = pil_to_tensor(raw.convert('RGB')).float().to(device) / 255
            prediction = model([image])[0]
            prediction = disjoint_predictions(prediction, settings)
            identities = tracker.update(prediction, int(frame.stem))
            write_frame(file, int(frame.stem), prediction, identities)
            times.append((time.perf_counter() - start) * 1000)
    temporary.replace(destination)
    return len(frames), times


def run(score_only=False):
    config, root = load()
    settings, tracking = config['full_frame'], config['tracking']
    checkpoint = root / settings['checkpoint']
    if not checkpoint.is_file():
        raise FileNotFoundError('Run `python train.py kitti-mots` to train the clean checkpoint first')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    check_checkpoint(saved)
    if saved['settings']['architecture'] != settings['architecture'] or saved['settings']['image_size'] != settings['image_size']:
        raise ValueError('Checkpoint architecture/image size differs from benchmark config')
    data_root = root / settings['data_root']
    sequences = {f'{seq:04d}': len(list((data_root / 'training/image_02' / f'{seq:04d}').glob('*.png')))
                 for seq in settings['val_sequences']}
    if not all(sequences.values()):
        raise FileNotFoundError('Missing validation images; run `python prepare.py kitti`')
    names = [tracking['name']]
    if not re.fullmatch(r'[A-Za-z0-9_-]+', names[0]):
        raise ValueError('tracking.name must be a single folder name')
    predictions_root = root / tracking['predictions_root']
    run_folder = predictions_root / names[0]
    run_folder.mkdir(parents=True, exist_ok=True)
    with checkpoint.open('rb') as file:
        sha = hashlib.file_digest(file, 'sha256').hexdigest()
    provenance = {'checkpoint_sha256': sha, 'checkpoint': str(checkpoint),
                  'settings': saved['settings'],
                  'tracking': {key: value for key, value in tracking.items() if key != 'external_trackers'},
                  'trackeval_revision': TRACKEVAL_REVISION}
    manifest = run_folder / 'run.yaml'
    if manifest.is_file() and yaml.safe_load(manifest.read_text()) != provenance:
        raise ValueError('Predictions belong to different weights/settings. Choose a new tracking.name in configs/kitti_mots.yaml')
    manifest.write_text(yaml.safe_dump(provenance))
    if not score_only:
        model = FullFrameVehicleNet(num_identities=saved['model']['id_classifier.weight'].shape[0],
                                    pretrained=False, image_size=settings['image_size'],
                                    architecture=settings['architecture'])
        model.load_state_dict(saved['model'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device).eval()
        timings = []
        timing_rows = []
        device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
        for sequence in sequences:
            destination = run_folder / 'data' / f'{sequence}.txt'
            timing_path = run_folder / 'data' / f'{sequence}.timing.yaml'
            if destination.is_file() and timing_path.is_file():
                cached = yaml.safe_load(timing_path.read_text())
                times = cached['ms_per_frame']
                hardware = cached['device_name']
                print(f'{sequence}: reusing complete predictions', flush=True)
            else:
                count, times = infer_sequence(model, data_root / 'training/image_02' / sequence,
                                               destination, tracking, device)
                hardware = device_name
                timing_path.write_text(yaml.safe_dump({'device': str(device), 'device_name': hardware,
                                                      'ms_per_frame': times}))
                print(f'{sequence}: tracked all {count} frames ({device})', flush=True)
            timings.extend(times)
            timing_rows.append({'sequence': sequence, 'device': hardware, 'frames': len(times),
                                'median_ms': statistics.median(times), 'mean_ms': statistics.mean(times),
                                'fps': 1000 / statistics.mean(times)})
        timing_output = root / tracking['metrics_root'] / 'timing.csv'
        timing_output.parent.mkdir(parents=True, exist_ok=True)
        with timing_output.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(timing_rows[0]))
            writer.writeheader()
            writer.writerows(timing_rows)
        if len({row['device'] for row in timing_rows}) == 1:
            print(f'Median end-to-end time: {statistics.median(timings):.1f} ms/frame '
                  f'({timing_rows[0]["device"]}, batch 1, includes PNG decode, network, tracking, and mask export)', flush=True)
        else:
            print('Cached timings used different devices; compare per-device timings in timing.csv', flush=True)
    for name, external in tracking.get('external_trackers', {}).items():
        if name in names or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError(f'Invalid/duplicate external tracker name: {name}')
        from .protocol import check_settings
        check_settings(external['protocol'])
        source = (root / external['predictions']).resolve()
        destination = predictions_root / name / 'data'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.symlink_to(source, target_is_directory=True)
        elif destination.resolve() != source:
            raise ValueError(f'External tracker folder conflict: {destination}')
        names.append(name)
    rows = evaluate(data_root, predictions_root, names, sequences, root / tracking['metrics_root'])
    for row in rows:
        print(f'{row["model"]}: sMOTSA={row["sMOTSA"]:.2f}, HOTA={row["HOTA"]:.2f}, '
              f'IDF1={row["IDF1"]:.2f}, IDSW={row["IDSW"]}', flush=True)
    print(f'Report: {root / tracking["metrics_root"] / "summary.html"}', flush=True)


def check():
    """Verify loss, inference, export, and official evaluation without training."""
    import tempfile
    from .training import check_dataset
    config, root = load()
    check_dataset(CONFIG)
    settings = config['full_frame']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FullFrameVehicleNet(num_identities=1, pretrained=False,
                                image_size=settings['image_size'], architecture=settings['architecture']).to(device).eval()
    sequence = f'{settings["val_sequences"][0]:04d}'
    with tempfile.TemporaryDirectory(prefix='mots_pipeline_') as directory:
        folder = Path(directory)
        truth_folder = folder / 'gt/instances_txt'
        truth_folder.mkdir(parents=True)
        lines = (root / settings['data_root'] / 'instances_txt' / f'{sequence}.txt').read_text().splitlines()
        (truth_folder / f'{sequence}.txt').write_text('\n'.join(line for line in lines if int(line.split()[0]) < 3) + '\n')
        infer_sequence(model, root / settings['data_root'] / 'training/image_02' / sequence,
                       folder / 'trackers/smoke/data' / f'{sequence}.txt', config['tracking'], device, frame_limit=3)
        evaluate(folder / 'gt', folder / 'trackers', ['smoke'], {sequence: 3}, folder / 'metrics', print_results=False)
    print('Pipeline check passed: three real frames, disjoint mask export, official TrackEval. '
          'Smoke predictions used untrained weights and are not benchmark results.', flush=True)
