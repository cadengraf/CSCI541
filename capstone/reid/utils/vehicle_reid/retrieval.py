"""Separate cross-camera open-set evaluation, with resumable specialist inference."""
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
import yaml
from utils.tracking.common import ROOT, save
from utils.teacher.assets import sha256
from utils.reid_teacher.data import read, partitions
from utils.core.metrics import cross_view_metrics
from .models import Embeddings


def open_set(query, gallery, query_ids, gallery_ids, threshold, margin):
    """One immutable reference per known identity; withhold every fifth identity."""
    identities = sorted(set(gallery_ids))
    unknown = set(identities[::5])
    known = [identity for identity in identities if identity not in unknown]
    if len(known) < 2 or not unknown:
        raise ValueError('Open-set evaluation needs at least three identities')
    references = torch.stack([gallery[gallery_ids.index(identity)] for identity in known])
    scores = query @ references.T
    top, indices = scores.topk(2, dim=1)
    accepted = (top[:, 0] >= threshold) & (top[:, 0]-top[:, 1] >= margin)
    predicted = [known[index] for index in indices[:, 0].tolist()]
    is_known = np.array([identity not in unknown for identity in query_ids])
    correct = np.array([a == b for a, b in zip(predicted, query_ids)])
    accept = accepted.numpy()
    return dict(protocol='single frozen gallery reference per identity; every fifth sorted ID withheld as unknown',
        threshold=threshold, margin=margin, known_queries=int(is_known.sum()), unknown_queries=int((~is_known).sum()),
        correct_matches=int((accept & correct & is_known).sum()),
        wrong_known_matches=int((accept & ~correct & is_known).sum()),
        rejected_known=int((~accept & is_known).sum()),
        false_unknown_accepts=int((accept & ~is_known).sum()),
        correct_unknown_rejections=int((~accept & ~is_known).sum()),
        known_recovery_rate=float((accept & correct & is_known).sum()/max(1, is_known.sum())),
        unknown_false_accept_rate=float((accept & ~is_known).sum()/max(1, (~is_known).sum())))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/vehicle_reid.yaml')
    parser.add_argument('--stage', choices=['calibration', 'final'], default='calibration')
    parser.add_argument('--limit-identities', type=int, help='Smoke subset only; omit for reporting')
    parser.add_argument('--output', default='metrics/vehicle_reid/cross_camera')
    parser.add_argument('--device', choices=['auto'], default='auto')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    config = yaml.safe_load((ROOT / args.config).read_text())
    torch.set_num_threads(config['cpu_threads'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    settings = read()
    data = partitions(settings)  # Always audit full identity/camera/image separation.
    names = ['calibration_query', 'calibration_gallery'] if args.stage == 'calibration' else ['query', 'gallery']
    query, gallery = (data[name] for name in names)
    if args.limit_identities is not None:
        if args.limit_identities < 3:
            raise ValueError('limit-identities must be at least three')
        ids = set(sorted({row['id'] for row in query})[:args.limit_identities])
        query = [row for row in query if row['id'] in ids]
        gallery = [row for row in gallery if row['id'] in ids]
    output = ROOT / args.output / (args.stage + (f'_smoke{args.limit_identities}' if args.limit_identities else ''))
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'state.pt'
    if destination.exists() and not args.resume:
        raise ValueError('Existing evaluation state; use --resume or another output')
    encoder = Embeddings(config['reid'], device)
    signature = dict(model=encoder.signature, manifests={name: sha256(ROOT / settings['data'][name]) for name in names},
                     limit=args.limit_identities, stage=args.stage)
    chunks = [[], []]
    elapsed, processed = 0., 0
    if args.resume:
        state = torch.load(destination, map_location='cpu', weights_only=False)
        if state['signature'] != signature:
            raise ValueError('Evaluation models or data changed')
        chunks, elapsed, processed = state['chunks'], state['seconds'], state['processed']
    batch = encoder.batch_size
    started = time.perf_counter()
    since_save = 0
    for group, rows in enumerate((query, gallery)):
        offset = sum(len(chunk) for chunk in chunks[group])
        for start in range(offset, len(rows), batch):
            images = []
            for row in rows[start:start+batch]:
                with Image.open(row['path']) as image:
                    images.append(encoder.transform(image.convert('RGB')))
            with torch.inference_mode():
                vectors = encoder.model(torch.stack(images).to(device)).cpu()
            chunks[group].append(vectors)
            processed += len(images)
            since_save += 1
            if since_save >= 10 or start+batch >= len(rows):
                elapsed += time.perf_counter()-started
                save(dict(signature=signature, chunks=chunks, seconds=elapsed, processed=processed), destination)
                print(f'{args.stage}: {processed}/{len(query)+len(gallery)} crop embeddings; checkpoint saved', flush=True)
                started, since_save = time.perf_counter(), 0
    q, g = [torch.cat(group) for group in chunks]
    query_ids, gallery_ids = [[row['id'] for row in rows] for rows in (query, gallery)]
    correct, count, mean_ap = cross_view_metrics(q, query_ids, [row['camera'] for row in query],
                                                g, gallery_ids, [row['camera'] for row in gallery])
    association = config['association']
    result = dict(stage=args.stage, smoke_subset=args.limit_identities is not None, signature=signature,
        rank1=correct/count, map=mean_ap, queries=count, gallery=len(gallery),
        open_set=open_set(q, g, query_ids, gallery_ids, association['recovery_similarity'], association['recovery_margin']),
        extraction_seconds=elapsed, crops_per_second=processed/max(elapsed, 1e-8), device=str(device),
        timing_scope='crop decode, preprocessing, specialist, periodic extraction checkpoints; excludes video detector and association',
        note='Cross-camera retrieval only. Thresholds supplied by YAML; final IDs never tune thresholds.')
    (output / 'summary.yaml').write_text(yaml.safe_dump(result))
    print(yaml.safe_dump({key: value for key, value in result.items() if key != 'signature'}), flush=True)
