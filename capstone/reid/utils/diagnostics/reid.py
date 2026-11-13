"""Pooling ablations and a training-only cached-feature ReID head overfit check."""
import copy
import csv
import random
from collections import defaultdict
from html import escape

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.ops import roi_align
from torchvision.transforms.functional import pil_to_tensor

from utils.core.metrics import cross_view_metrics
from utils.teacher.assets import sha256
from utils.teacher.data import build
from utils.teacher.model import CoDINOJoint
from utils.teacher.settings import ROOT, resolve
from utils.teacher.training import validate_checkpoint
from utils.distillation.settings import read


def pool(features, masks=None):
    if masks is None:
        return features.float().mean((2, 3))
    weights = F.interpolate(masks.float(), features.shape[-2:], mode='bilinear', align_corners=False)
    return (features.float()*weights).sum((2, 3))/weights.sum((2, 3)).clamp_min(1e-5)


def scores(q, qids, g, gids):
    correct, count, ap = cross_view_metrics(q, qids, ['query']*len(q), g, gids, ['gallery']*len(g))
    return {'Rank1': 100*correct/count, 'mAP': 100*ap, 'queries': count, 'gallery': len(g)}


@torch.no_grad()
def crops(model, records):
    collected = defaultdict(list)
    for start in range(0, len(records), 2):
        images = []
        for row in records[start:start+2]:
            with Image.open(row['path']) as raw:
                images.append(pil_to_tensor(raw.convert('RGB')).float().cuda()/255)
        batch, meta, _, _, _ = model._prepare(images, crop=True)
        encoded = model.detector.query_head(model.detector.extract_feat(batch, meta), meta)[-1]
        boxes = [batch.new_tensor([[0, 0, m['img_shape'][1], m['img_shape'][0]]]) for m in meta]
        _, (output, masks) = model._roi_identity(encoded, boxes)
        for mode, weights in [('global', None), ('predicted', masks.sigmoid())]:
            collected[mode].append(pool(output['mask_feats'], weights).cpu())
        if start % 64 == 0:
            print(f'Crop features {start+len(images)}/{len(records)}', flush=True)
    return {mode: torch.cat(values) for mode, values in collected.items()}


def write_table(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run():
    cfg = yaml.safe_load((ROOT/'configs/reid_diagnostics.yaml').read_text())
    settings, root = read()
    rng = random.Random(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(4)
    kitti, vrai, calibration, _ = build(settings, root)
    path = resolve(root, cfg['checkpoint'])
    saved = torch.load(path, map_location='cpu')
    validate_checkpoint(saved, settings, kitti, vrai)
    model = CoDINOJoint(settings, root, len(kitti.identity_to_label)+len(vrai.identities), pretrained=False)
    model.load_state_dict(saved['model']); epoch = saved['epoch']; del saved
    model.cuda().eval()
    output_dir = resolve(root, cfg['output']); output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    chosen = set(rng.sample(sorted({r['id'] for r in calibration[0]}), cfg['calibration_identities']))
    parts = [[r for r in split if r['id'] in chosen] for split in calibration]
    vectors = [crops(model, split) for split in parts]
    for mode in ('global', 'predicted'):
        with torch.no_grad():
            q, g = [F.normalize(model.reid.projection(v[mode].cuda()).float(), dim=1).cpu() for v in vectors]
        results.append({'dataset': 'VRAI calibration subset', 'pooling': mode,
                        **scores(q, [r['id'] for r in parts[0]], g, [r['id'] for r in parts[1]])})
    write_table(output_dir/'pooling.csv', results)
    # Identical GT boxes and ROI features in all three modes: isolates pooling,
    # not detection quality. KITTI frames are temporal views, not cross-camera.
    from utils.full_frame.kitti import KittiMOTSCars
    ds = KittiMOTSCars(resolve(root, settings['data']['kitti_root']), settings['data']['val_sequences'])
    keys = sorted(ds.annotations)[::cfg['kitti_frame_stride']]
    keys = sorted(rng.sample(keys, min(cfg['kitti_frames'], len(keys))))
    records = []
    mask_ious = []
    with torch.no_grad():
        for index, key in enumerate(keys):
            image, target = ds._frame(*key)
            if not len(target['boxes']):
                continue
            batch, meta, boxes, _, truth = model._prepare([image.cuda()], [target])
            encoded = model.detector.query_head(model.detector.extract_feat(batch, meta), meta)[-1]
            _, (out, masks) = model._roi_identity(encoded, boxes)
            truth_tensor = torch.as_tensor(truth[0].masks, device='cuda').float()[:, None]
            # One mask per ROI: the ROI batch index refers to its own instance.
            rois = torch.cat([torch.arange(len(boxes[0]), device='cuda')[:, None], boxes[0]], 1)
            oracle = roi_align(truth_tensor, rois, output_size=masks.shape[-2:], aligned=True)
            predicted_binary = masks.sigmoid() >= .5; true_binary = oracle >= .5
            intersection = (predicted_binary & true_binary).sum((1, 2, 3)).float()
            union = (predicted_binary | true_binary).sum((1, 2, 3)).clamp_min(1)
            mask_ious.extend((intersection/union).cpu().tolist())
            features = out['mask_feats']
            values = {mode: F.normalize(model.reid.projection(pool(features, weights)), dim=1).cpu()
                      for mode, weights in [('global', None), ('predicted', masks.sigmoid()), ('ground_truth', oracle)]}
            for j, track in enumerate(target['track_ids'].tolist()):
                records.append((f'{key[0]}:{track}', key[1], {m: v[j] for m, v in values.items()}))
            print(f'KITTI diagnostic frame {index+1}/{len(keys)}', flush=True)
    by_id = defaultdict(list)
    for row in records: by_id[row[0]].append(row)
    query, gallery = [], []
    for identity, rows in sorted(by_id.items()):
        rows.sort(key=lambda r: r[1])
        if len(rows) >= 2:
            query.append(rows[0]); gallery.extend(rows[1:])
    if len(query) < 2:
        raise ValueError('Need at least two repeated KITTI identities; increase kitti_frames')
    for mode in ('global', 'predicted', 'ground_truth'):
        results.append({'dataset': 'KITTI GT-box temporal subset', 'pooling': mode,
                        **scores(torch.stack([r[2][mode] for r in query]), [r[0] for r in query],
                                 torch.stack([r[2][mode] for r in gallery]), [r[0] for r in gallery])})
    write_table(output_dir/'pooling.csv', results)
    # Fixed TRAINING images only, no augmentation and no backbone updates.
    # This tests whether the existing ROI representation + trainable projection
    # can memorize identities; it is intentionally not a generalization score.
    train_ids = rng.sample(sorted(vrai.identities), cfg['train_identities'])
    training = []
    for identity in train_ids:
        for camera in sorted(vrai.by_identity[identity])[:2]:
            paths = sorted(vrai.by_identity[identity][camera])
            for p in rng.sample(paths, min(cfg['images_per_camera'], len(paths))):
                training.append({'path': str(p), 'id': identity, 'camera': camera})
    train_features = crops(model, training)
    projection = copy.deepcopy(model.reid.projection).cpu()
    del model; torch.cuda.empty_cache()
    mapping = {identity: i for i, identity in enumerate(train_ids)}
    labels = torch.tensor([mapping[r['id']] for r in training])
    qa = [i for i, r in enumerate(training) if r['camera'] == sorted({x['camera'] for x in training})[0]]
    ga = [i for i in range(len(training)) if i not in qa]
    curves = []
    for mode, features in train_features.items():
        head = copy.deepcopy(projection)
        classifier = torch.nn.Linear(settings['model']['embedding_dim'], len(train_ids), bias=False)
        optimizer = torch.optim.Adam(list(head.parameters())+list(classifier.parameters()), lr=cfg['overfit_learning_rate'])
        for step in range(cfg['overfit_steps']+1):
            embeddings = F.normalize(head(features), dim=1)
            logits = 30*F.linear(embeddings, F.normalize(classifier.weight, dim=1))
            loss = F.cross_entropy(logits, labels)
            if step % 50 == 0 or step == cfg['overfit_steps']:
                values = scores(embeddings.detach()[qa], [training[i]['id'] for i in qa],
                                embeddings.detach()[ga], [training[i]['id'] for i in ga])
                curves.append({'pooling': mode, 'step': step, 'loss': float(loss.detach()),
                               'training_accuracy': 100*float((logits.argmax(1)==labels).float().mean()), **values})
                print(f'Training-only head overfit {mode}: {curves[-1]}', flush=True)
            if step < cfg['overfit_steps']:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
    write_table(output_dir/'overfit.csv', curves)
    def table(rows):
        return '<table><tr>'+''.join(f'<th>{escape(k)}</th>' for k in rows[0])+'</tr>'+''.join(
            '<tr>'+''.join(f'<td>{v:.3f}</td>' if isinstance(v, float) else f'<td>{escape(str(v))}</td>' for v in row.values())+'</tr>' for row in rows)+'</table>'
    provenance = {'checkpoint': str(path), 'sha256': sha256(path), 'epoch': epoch, 'config': cfg,
                  'vrai_calibration_ids': sorted(chosen), 'training_only_ids': train_ids,
                  'kitti_frames': [list(key) for key in keys], 'kitti_roi_mask_iou': sum(mask_ious)/len(mask_ious)}
    (output_dir/'provenance.yaml').write_text(yaml.safe_dump(provenance))
    (output_dir/'summary.html').write_text('<!doctype html><meta charset="utf-8"><title>ReID diagnostics</title>'
        '<style>body{font:16px system-ui;margin:32px}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style>'
        '<h1>Joint ReID diagnostics</h1><p>VRAI uses a fixed calibration subset, not final test. KITTI uses GT boxes and '
        'temporal views, not cross-camera views. Do not compare their scores directly. Ground-truth masks are an oracle available only on KITTI.</p>'
        +table(results)+f'<p>KITTI mean instance IoU inside GT ROIs: {sum(mask_ious)/len(mask_ious):.3f}. Not full-image mask AP.</p>'
        '<h2>Training-only cached-feature head overfit</h2><p>Only the projection and a fresh small classifier are optimized on fixed training images. '
        'The backbone and ROI features are frozen. These scores measure memorization, not generalization or full-network trainability. '
        'No production checkpoint is modified.</p>'+table(curves))
    print(f'Report: {output_dir / "summary.html"}', flush=True)

