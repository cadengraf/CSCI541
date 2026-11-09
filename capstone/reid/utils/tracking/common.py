"""Configuration and small prediction records shared across Python environments."""

from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import yaml
from pycocotools import mask as rle

ROOT = Path(__file__).resolve().parents[2]


def read():
    return yaml.safe_load((ROOT / 'configs/tracking.yaml').read_text())


def vendor(settings, name):
    path = ROOT / settings[f'{name}_source']
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != settings[f'{name}_revision']:
        raise ValueError(f'{name} source revision differs from configs/tracking.yaml')
    sys.path.insert(0, str(path))
    return path


def encode(prediction, elapsed):
    masks = [rle.encode(np.asfortranarray(mask[0].cpu().numpy().astype(np.uint8)))
             for mask in prediction['masks']]
    return {**{key: prediction[key].detach().cpu().float() for key in ('boxes', 'scores', 'embeddings')},
            'rles': masks, 'model_ms': elapsed}


def decode(record):
    masks = np.stack([rle.decode(mask) for mask in record['rles']]) if record['rles'] else None
    height, width = record['size']
    return {**{key: record[key] for key in ('boxes', 'scores', 'embeddings')},
            'masks': torch.from_numpy(masks)[:, None].float() if masks is not None
                     else torch.zeros((0, 1, height, width))}


def load(path):
    # We only read our locally generated records. Explicit for both Torch versions.
    return torch.load(path, map_location='cpu', weights_only=False)


def save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    torch.save(payload, temporary)
    temporary.replace(path)
