"""Cross-camera embedding evaluation; calibration never uses final-test IDs."""

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from utils.core.metrics import cross_view_metrics


@torch.no_grad()
def extract(model, records, device, global_pool=False, mixed_precision=False):
    vectors = []
    for start in range(0, len(records), 2):
        images = []
        for row in records[start:start + 2]:
            with Image.open(row['path']) as raw:
                images.append(pil_to_tensor(raw.convert('RGB')).float().to(device) / 255)
        with torch.cuda.amp.autocast(enabled=mixed_precision and device.type == 'cuda'):
            vectors.append(model.crop_embeddings(images, global_pool=global_pool).float().cpu())
    return torch.cat(vectors)


def evaluate(model, partition, device, global_pool=False, mixed_precision=False):
    model.eval()
    query, gallery = partition
    q = extract(model, query, device, global_pool, mixed_precision)
    g = extract(model, gallery, device, global_pool, mixed_precision)
    correct, count, mean_ap = cross_view_metrics(q, [row['id'] for row in query], [row['camera'] for row in query],
                                                g, [row['id'] for row in gallery], [row['camera'] for row in gallery])
    return {'rank1': correct / count, 'map': mean_ap, 'queries': count, 'gallery': len(gallery)}
