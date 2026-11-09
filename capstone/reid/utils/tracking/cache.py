"""Export the fixed teacher once, in its isolated environment, with resume by frame."""

import time
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor
import yaml

from .common import ROOT, read, encode, save


@torch.inference_mode()
def export(check=False, selected=False):
    from utils.teacher.settings import read as teacher_settings
    from utils.teacher.assets import sha256
    from utils.teacher.model import CoDINOJoint
    from utils.teacher.training import validate_checkpoint
    settings = read()
    config, root = teacher_settings()
    checkpoint = ROOT / config['training']['best_checkpoint']
    saved = torch.load(checkpoint, map_location='cpu')
    validate_checkpoint(saved, config)
    last = torch.load(ROOT / config['training']['checkpoint'], map_location='cpu')
    if not last['finished'] or last['run_id'] != saved['run_id']:
        raise ValueError('Use the selected checkpoint from a completed teacher run')
    del last
    model = CoDINOJoint(config, root, saved['model']['reid.classifier'].shape[0], pretrained=False)
    model.load_state_dict(saved['model'])
    model.cuda().eval()
    del saved
    cache = ROOT / settings['cache']
    provenance = {'checkpoint_sha256': sha256(checkpoint), 'model': config['model'],
                  'source': config['source'], 'min_score': settings['cache_min_score'], 'format': 1,
                  'hardware': torch.cuda.get_device_name()}
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / 'source.yaml'
    if manifest.is_file() and yaml.safe_load(manifest.read_text()) != provenance:
        raise ValueError('Cached predictions use different weights/settings; choose another cache folder')
    manifest.write_text(yaml.safe_dump(provenance))
    sequences = config['data']['val_sequences']
    if check:
        first = int(settings['check_sequence'])
        sequences = [first, next(seq for seq in sequences if seq != first)]
    elif selected:
        sequences = [int(settings['selected']['sequence'])]
    for seq in sequences:
        folder = ROOT / config['data']['kitti_root'] / 'training/image_02' / f'{seq:04d}'
        frames = sorted(folder.glob('*.png'))
        if not frames or [int(path.stem) for path in frames] != list(range(len(frames))):
            raise ValueError(f'Incomplete source sequence {folder}')
        if check:
            frames = frames[:settings['check_frames']]
        elif selected:
            spec = settings['selected']
            frames = frames[spec['start_frame']:spec['start_frame'] + spec['frames']]
        for index, path in enumerate(frames):
            destination = cache / f'{seq:04d}' / f'{path.stem}.pt'
            if destination.is_file():
                continue
            torch.cuda.synchronize()
            start = time.perf_counter()
            with Image.open(path) as raw:
                image = pil_to_tensor(raw.convert('RGB')).float().cuda() / 255
            prediction = model([image])[0]
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            chosen = prediction['scores'] >= settings['cache_min_score']
            prediction = {key: value[chosen] for key, value in prediction.items()}
            record = encode(prediction, elapsed)
            record['size'] = list(image.shape[-2:])
            save(record, destination)
            if index % 50 == 0:
                print(f'Fixed teacher cache: {seq:04d} {index+1}/{len(frames)} frames', flush=True)
        print(f'Cache ready: {seq:04d}, {len(frames)} frames', flush=True)
