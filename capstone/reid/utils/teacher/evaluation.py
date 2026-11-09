"""Final held-out retrieval and conventional KITTI MOTS teacher evaluation."""

import csv
from html import escape
import statistics

import torch
import yaml

from utils.full_frame.mots_benchmark import infer_sequence
from utils.full_frame.mots_evaluation import evaluate as evaluate_mots, TRACKEVAL_REVISION
from .assets import sha256
from .data import build
from .model import CoDINOJoint
from .retrieval import evaluate as evaluate_retrieval
from .settings import read, resolve
from .training import validate_checkpoint


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.csv')
    with temporary.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run():
    settings, root = read()
    spec, tracking = settings['training'], settings['tracking']
    last = torch.load(resolve(root, spec['checkpoint']), map_location='cpu')
    if not last['finished']:
        raise ValueError('Complete teacher training before evaluating final held-out identities')
    run_id = last['run_id']
    del last
    path = resolve(root, spec['best_checkpoint'])
    saved = torch.load(path, map_location='cpu')
    kitti, vrai, _, final = build(settings, root)
    validate_checkpoint(saved, settings, kitti, vrai)
    if saved['run_id'] != run_id:
        raise ValueError('Best and last teacher checkpoints are from different runs')
    if not torch.cuda.is_available():
        raise RuntimeError('Run teacher evaluation on the CUDA GPU host')
    model = CoDINOJoint(settings, root, len(kitti.identity_to_label) + len(vrai.identities), pretrained=False)
    model.load_state_dict(saved['model'])
    device = torch.device('cuda')
    model.to(device).eval()
    epoch = saved['epoch']
    del saved
    output = resolve(root, tracking['metrics_root']).parent
    output.mkdir(parents=True, exist_ok=True)
    print(f'Evaluating teacher selected at epoch {epoch}; final VRAI query/gallery', flush=True)
    retrieval_rows = []
    for pooling in ('mask', 'global'):
        scores = evaluate_retrieval(model, final, device, global_pool=pooling == 'global',
                                    mixed_precision=spec['mixed_precision'])
        retrieval_rows.append({'model': tracking['name'], 'pooling': pooling,
                               'Rank1': scores['rank1'] * 100, 'mAP': scores['map'] * 100,
                               'queries': scores['queries'], 'gallery': scores['gallery']})
        print(f'VRAI {pooling}: Rank-1 {scores["rank1"]:.2%}, mAP {scores["map"]:.2%}', flush=True)
    write_csv(output / 'vrai.csv', retrieval_rows)
    data_root = resolve(root, settings['data']['kitti_root'])
    sequences = {f'{seq:04d}': len(list((data_root / 'training/image_02' / f'{seq:04d}').glob('*.png')))
                 for seq in settings['data']['val_sequences']}
    if not all(sequences.values()):
        raise FileNotFoundError('Incomplete KITTI validation frames')
    name = tracking['name']
    if not name or any(character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for character in name):
        raise ValueError('tracking.name must be a single folder name')
    predictions_root = resolve(root, tracking['predictions_root'])
    folder = predictions_root / name
    folder.mkdir(parents=True, exist_ok=True)
    provenance = {'checkpoint_sha256': sha256(path), 'run_id': run_id, 'settings': settings,
                  'trackeval_revision': TRACKEVAL_REVISION}
    manifest = folder / 'run.yaml'
    if manifest.is_file() and yaml.safe_load(manifest.read_text()) != provenance:
        raise ValueError('Existing predictions use other weights/settings; choose a new tracking.name')
    manifest.write_text(yaml.safe_dump(provenance))
    timings = []
    for sequence in sequences:
        destination = folder / 'data' / f'{sequence}.txt'
        timing_file = destination.with_suffix('.timing.yaml')
        if destination.is_file() and timing_file.is_file():
            cached = yaml.safe_load(timing_file.read_text())
            times, hardware = cached['times'], cached['hardware']
            if len(times) != sequences[sequence]:
                raise ValueError(f'Incomplete cached sequence {sequence}')
            print(f'{sequence}: reusing complete predictions', flush=True)
        else:
            _, times = infer_sequence(model, data_root / 'training/image_02' / sequence,
                                       destination, tracking, device)
            hardware = torch.cuda.get_device_name(device)
            timing_file.write_text(yaml.safe_dump({'times': times, 'hardware': hardware}))
            print(f'{sequence}: all {len(times)} frames tracked', flush=True)
        timings.append({'sequence': sequence, 'hardware': hardware, 'frames': len(times),
                        'mean_ms': statistics.mean(times), 'median_ms': statistics.median(times),
                        'scope': 'batch 1; PNG decode, AMP backbone/FP32 heads, tracking, RLE export'})
    write_csv(output / 'timing.csv', timings)
    mots_rows = evaluate_mots(data_root, predictions_root, [name], sequences,
                              resolve(root, tracking['metrics_root']))
    report(output, retrieval_rows, mots_rows, epoch, model)
    print(f'Report: {output / "summary.html"}', flush=True)


def report(output, retrieval, mots, epoch, model):
    def table(rows):
        heading = ''.join(f'<th>{escape(key)}</th>' for key in rows[0])
        body = ''.join('<tr>' + ''.join(f'<td>{value:.2f}</td>' if isinstance(value, float)
                                      else f'<td>{escape(str(value))}</td>' for value in row.values())
                       + '</tr>' for row in rows)
        return f'<table><tr>{heading}</tr>{body}</table>'
    parameters = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    (output / 'summary.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Vehicle teacher results</title>'
        '<style>body{font:16px system-ui;margin:40px;color:#172033}table{border-collapse:collapse}'
        'td,th{padding:12px;border-bottom:1px solid #ccd;text-align:right}</style>'
        f'<h1>Co-DINO-Inst ViT-L vehicle teacher</h1><p>{parameters:.1f}M parameters including the training '
        f'identity classifier. Selected epoch: {epoch}, using VRAI calibration mAP.</p>'
        '<h2>Held-out cross-camera retrieval</h2>' + table(retrieval) +
        '<p>Same checkpoint, same final query/gallery; global versus mask pooling is an inference ablation. '
        'The head was trained using masks. VRAI identities are separate from training and calibration.</p>'
        '<h2>Conventional KITTI MOTS validation</h2>' + table(mots) +
        '<p>All validation frames, official TrackEval mask matching. TP/(TP+FN) is mask-based recall; '
        'it does not isolate box detection recall. Timing in timing.csv includes decode, network, '
        'tracking and RLE export; exclude startup and compare on the same hardware.</p>'
        '<p>Training data: COCO pretrained detector/masks, KITTI training sequences, VRAI training identities. '
        'Additional identity training means this is not a matched-budget ablation of the earlier model. '
        'These results do not by themselves establish state of the art.</p>')
