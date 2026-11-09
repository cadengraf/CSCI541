"""VRAI cross-camera crop retrieval for the KITTI-trained full-frame network."""

import csv
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from utils.benchmarking.models import retrieval_metrics
from .model import FullFrameVehicleNet


def _rows(path):
    with Path(path).open(newline='') as file:
        reader = csv.DictReader(file)
        if not {'path', 'id', 'camera'} <= set(reader.fieldnames or ()):
            raise ValueError(f'{path} needs path, id, camera columns')
        return list(reader)


@torch.inference_mode()
def _vectors(model, rows, device):
    mask_vectors, global_vectors = [], []
    for index, row in enumerate(rows, 1):
        with Image.open(row['path']) as raw:
            image = pil_to_tensor(raw.convert('RGB')).float().to(device) / 255
        masked, global_pooled = model.crop_embeddings([image], both=True)
        mask_vectors.append(masked[0].cpu())
        global_vectors.append(global_pooled[0].cpu())
        if index % 250 == 0 or index == len(rows):
            print(f'Embedded {index}/{len(rows)} VRAI crops', flush=True)
    return torch.stack(mask_vectors), torch.stack(global_vectors)


def evaluate(config_path, multiview=False):
    """Evaluate an oracle crop box with predicted mask versus global ROI pooling.

    VRAI's public test labels are unavailable, so this uses the existing
    identity-disjoint local holdout of VRAI's officially labeled train split.
    """
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    settings = config['full_frame']
    manifests = config['vrai']['vehicle_benchmark']

    def path(value):
        result = Path(value).expanduser()
        return result if result.is_absolute() else config_path.parent / result

    train_rows = _rows(path(manifests['train_manifest']))
    query_rows = _rows(path(manifests['query_manifest']))
    gallery_rows = _rows(path(manifests['gallery_manifest']))
    train_ids = {row['id'] for row in train_rows}
    query_ids = {row['id'] for row in query_rows}
    gallery_ids = {row['id'] for row in gallery_rows}
    if train_ids & (query_ids | gallery_ids) or query_ids != gallery_ids:
        raise ValueError('VRAI train/query/gallery identities are not disjoint as expected')
    if {row['path'] for row in query_rows} & {row['path'] for row in gallery_rows}:
        raise ValueError('VRAI query/gallery share image paths')
    checkpoint = path(settings['multiview']['checkpoint'] if multiview else settings['checkpoint'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    model = FullFrameVehicleNet(num_identities=saved['model']['id_classifier.weight'].shape[0],
                                pretrained=False, image_size=saved['settings']['image_size'],
                                architecture=saved['settings']['architecture'],
                                crop_image_size=settings['multiview']['crop_image_size'])
    model.load_state_dict(saved['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    print(f'VRAI local cross-camera holdout: {len(query_rows)} query / {len(gallery_rows)} gallery '
          f'/ {len(query_ids)} IDs; crop size {settings["multiview"]["crop_image_size"]}; '
          f'checkpoint epoch {saved["epoch"]}; device={device}', flush=True)
    query_vectors = _vectors(model, query_rows, device)
    gallery_vectors = _vectors(model, gallery_rows, device)
    results = []
    label = 'Full-frame KITTI + VRAI' if multiview else 'Full-frame KITTI teacher'
    for name, q_vectors, g_vectors in (
            (f'{label} (mask pool)', query_vectors[0], gallery_vectors[0]),
            (f'{label} (global pool)', query_vectors[1], gallery_vectors[1])):
        metrics = retrieval_metrics(q_vectors, [row['id'] for row in query_rows],
                                    [row['camera'] for row in query_rows],
                                    g_vectors, [row['id'] for row in gallery_rows],
                                    [row['camera'] for row in gallery_rows])
        print(f'{name}: Rank-1={metrics["rank1"]:.2%}, mAP={metrics["map"]:.2%}', flush=True)
        results.append({'model': name, 'queries': metrics['queries'], 'gallery': metrics['gallery'],
                        'rank1': metrics['rank1'], 'rank5': metrics['rank5'],
                        'rank10': metrics['rank10'], 'map': metrics['map']})
    output = path('metrics/full_frame_multiview_vrai_local.csv' if multiview
                  else 'metrics/full_frame_vrai_local.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial.csv')
    with temporary.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=('model', 'queries', 'gallery', 'rank1',
                                                  'rank5', 'rank10', 'map'))
        writer.writeheader()
        writer.writerows(results)
    temporary.replace(output)
    print(f'Saved {output}. Oracle object crops; not full-frame detection or official VRAI test.', flush=True)
