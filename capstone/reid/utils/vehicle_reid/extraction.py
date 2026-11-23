"""Resumable common specialist observations; no tracking or GT enters extraction."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import time
import cv2
import numpy as np
import torch
import yaml
from utils.tracking.common import ROOT, load, save
from utils.teacher.assets import sha256
from .models import Embeddings


def extract_sequence(task):
    config, sequence = task
    torch.set_num_threads(config['cpu_threads'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frames = sorted((ROOT / config['data_root'] / 'training/image_02' / sequence).glob('*.png'))
    records = [ROOT / config['cache'] / sequence / f'{path.stem}.pt' for path in frames]
    if not frames or any(int(path.stem) != index for index, path in enumerate(frames)) or not all(p.is_file() for p in records):
        raise ValueError(f'Incomplete original frames or detector cache for {sequence}')
    pipeline = yaml.safe_load((ROOT / config['pipeline_config']).read_text())
    selection = yaml.safe_load((ROOT / pipeline['reid']['selection']).read_text())
    signature = dict(selection=selection, cache=yaml.safe_load((ROOT / config['cache'] / 'source.yaml').read_text()),
        score_threshold=config['score_threshold'],
        images=[(path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in frames],
        records=[(path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in records])
    destination = ROOT / config['features'] / sequence / 'state.pt'
    observations, seconds = [], 0.
    if destination.exists():
        previous = load(destination)
        if previous['signature'] != signature:
            raise ValueError(f'Feature extraction inputs changed: {sequence}; use a new features directory')
        observations, seconds = previous['observations'], previous['seconds']
        if len(observations) == len(frames):
            print(f'{sequence}: all {len(frames)} specialist observations already complete', flush=True)
            return sequence
    encoder = Embeddings(pipeline['reid'], device)
    print(f'{sequence}: specialist extraction resumes at {len(observations)}/{len(frames)} ({device})', flush=True)
    begin = time.perf_counter()
    for index in range(len(observations), len(frames)):
        raw = load(records[index])
        image = cv2.imread(str(frames[index]))
        if image is None:
            raise ValueError(f'Cannot decode {frames[index]}')
        height, width = image.shape[:2]
        indices = torch.where(raw['scores'] >= config['score_threshold'])[0].numpy()
        boxes = raw['boxes'][indices].numpy().copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        boxes = np.round(boxes)
        keep = np.isfinite(boxes).all(1) & ((boxes[:, 2:]-boxes[:, :2]).min(1) >= 2)
        indices, boxes = indices[keep], boxes[keep]
        vectors = encoder(image, boxes)
        observations.append(dict(indices=torch.from_numpy(indices), boxes=torch.from_numpy(boxes),
                                 embeddings=torch.from_numpy(vectors)))
        if (index+1) % config['checkpoint_every'] == 0 or index+1 == len(frames):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            seconds += time.perf_counter()-begin
            save(dict(format=1, signature=signature, observations=observations, seconds=seconds,
                      device=str(device), frames=len(frames), complete=index+1 == len(frames)), destination)
            print(f'{sequence}: specialist {index+1}/{len(frames)} frames committed ({seconds:.1f}s extraction)', flush=True)
            begin = time.perf_counter()
    return sequence


def extract(config):
    tasks = [(config, f'{int(sequence):04d}') for sequence in config['sequences']]
    workers = 1 if torch.cuda.is_available() else config['cpu_workers']
    if workers == 1:
        for task in tasks:
            extract_sequence(task)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            list(pool.map(extract_sequence, tasks))


if __name__ == '__main__':
    extract(yaml.safe_load((ROOT / 'configs/vehicle_benchmark.yaml').read_text()))
