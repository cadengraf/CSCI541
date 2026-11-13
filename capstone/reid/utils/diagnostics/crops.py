"""Inference-only crop geometry/resolution ablation on fixed calibration IDs."""
import random
import time
from html import escape
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from utils.core.metrics import cross_view_metrics
from utils.teacher.assets import sha256
from utils.teacher.data import build
from utils.teacher.evaluation import write_csv
from utils.teacher.model import CoDINOJoint
from utils.teacher.settings import ROOT, resolve
from utils.teacher.training import validate_checkpoint
from utils.distillation.settings import read


def reshape(image, geometry, size):
    if geometry == 'aspect':
        return image
    if geometry == 'square':
        return F.interpolate(image[None], size=(size, size), mode='bilinear', align_corners=False)[0]
    raise ValueError(f'Unknown geometry {geometry}')


@torch.no_grad()
def extract(model, records, variant):
    embeddings = {'mask': [], 'global': []}
    for start in range(0, len(records), 2):
        images = []
        for row in records[start:start+2]:
            with Image.open(row['path']) as raw:
                image = pil_to_tensor(raw.convert('RGB')).float().cuda()/255
            images.append(reshape(image, variant['geometry'], variant['size']))
        batch, metadata, _, _, _ = model._prepare(images, crop=True)
        encoded = model.detector.query_head(model.detector.extract_feat(batch, metadata), metadata)[-1]
        boxes = [batch.new_tensor([[0, 0, m['img_shape'][1], m['img_shape'][0]]]) for m in metadata]
        mask_embeddings, (output, masks) = model._roi_identity(encoded, boxes)
        features = output['mask_feats']
        if model.reid_spatial_adapter is not None:
            features = model.reid_spatial_adapter(features)
        embeddings['mask'].append(mask_embeddings.float().cpu())
        embeddings['global'].append(model.reid(features, None).float().cpu())
        if start % 64 == 0:
            print(f'{variant["name"]}: crop {start+len(images)}/{len(records)}', flush=True)
    return {key: torch.cat(values) for key, values in embeddings.items()}


def run(full=False):
    config_path = ROOT/('configs/crop_validation.yaml' if full else 'configs/crop_diagnostics.yaml')
    cfg = yaml.safe_load(config_path.read_text())
    settings, root = read()
    if not torch.cuda.is_available():
        raise RuntimeError('Crop diagnostics require CUDA')
    names = [v['name'] for v in cfg['variants']]
    if len(names) != len(set(names)) or not names:
        raise ValueError('Variants must have unique names')
    if any(v['geometry'] not in ('aspect', 'square') or v['size'] < 32 for v in cfg['variants']):
        raise ValueError('Variants require aspect/square geometry and size >=32')
    kitti, vrai, calibration, _ = build(settings, root)
    count = cfg['calibration_identities']
    available = sorted({r['id'] for r in calibration[0]})
    count = len(available) if count is None else count
    if not 2 <= count <= len(available):
        raise ValueError('calibration_identities must be between 2 and available IDs')
    chosen = set(random.Random(cfg['seed']).sample(available, count))
    query, gallery = [[row for row in split if row['id'] in chosen] for split in calibration]
    path = resolve(root, cfg['checkpoint'])
    saved = torch.load(path, map_location='cpu')
    validate_checkpoint(saved, settings, kitti, vrai)
    model = CoDINOJoint(settings, root, len(kitti.identity_to_label)+len(vrai.identities), pretrained=False)
    model.load_state_dict(saved['model']); epoch = saved['epoch']; del saved
    model.cuda().eval()
    directory = resolve(root, cfg['output']); directory.mkdir(parents=True, exist_ok=True)
    rows = []
    original_size = model.settings['model']['crop_size']
    # Preserve preprocessing metadata separately; modifying crop size is only
    # local to this disposable inference model, never the YAML or checkpoint.
    try:
        for variant in cfg['variants']:
            model.settings['model']['crop_size'] = variant['size']
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); started = time.perf_counter()
            q = extract(model, query, variant); g = extract(model, gallery, variant)
            torch.cuda.synchronize(); elapsed = time.perf_counter()-started
            for pooling in ('mask', 'global'):
                correct, n, ap = cross_view_metrics(q[pooling], [r['id'] for r in query], [r['camera'] for r in query],
                    g[pooling], [r['id'] for r in gallery], [r['camera'] for r in gallery])
                result = dict(variant=variant['name'], geometry=variant['geometry'], size=variant['size'],
                              pooling=pooling, Rank1=100*correct/n, mAP=100*ap, queries=n, gallery=len(gallery),
                              extraction_ms_per_crop=1000*elapsed/(len(query)+len(gallery)),
                              peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
                rows.append(result)
                print(f'{variant["name"]} {pooling}: Rank-1 {result["Rank1"]:.2f}%, mAP {result["mAP"]:.2f}%', flush=True)
            write_csv(directory/'summary.csv', rows)
    finally:
        model.settings['model']['crop_size'] = original_size
    manifest_hashes = {key: sha256(resolve(root, settings['data'][key])) for key in
                       ('vrai_train', 'vrai_calibration_query', 'vrai_calibration_gallery')}
    provenance = dict(config=cfg, checkpoint_sha256=sha256(path), checkpoint_epoch=epoch,
                      calibration_ids=sorted(chosen), manifests=manifest_hashes,
                      hardware=torch.cuda.get_device_name(), torch_version=str(torch.__version__))
    (directory/'provenance.yaml').write_text(yaml.safe_dump(provenance))
    keys = list(rows[0])
    table = '<table><tr>'+''.join(f'<th>{escape(k)}</th>' for k in keys)+'</tr>'+''.join(
        '<tr>'+''.join(f'<td>{v:.3f}</td>' if isinstance(v, float) else f'<td>{escape(str(v))}</td>' for v in row.values())+'</tr>' for row in rows)+'</table>'
    (directory/'summary.html').write_text('<!doctype html><meta charset="utf-8"><title>Crop preprocessing diagnostics</title>'
        '<style>body{font:16px system-ui;margin:32px}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #ccc}</style>'
        '<h1>Crop geometry and resolution</h1><p>Fixed checkpoint and seeded calibration identities. Same-size aspect versus '
        'square comparisons change crop geometry; 384 versus 256 also changes resolution. Square crops are stretched without '
        'padding before the existing model normalization. Aspect crops retain the current longest-side cap. Both paths retain '
        'the existing batch padding to multiples of 32 and use identical original crops and full-crop ROI boxes.</p>'+table+
        f'<p>This run uses {count} calibration identities. Subset, full calibration and final-test scores are not interchangeable. '
        'Mask and global pooling reuse one feature extraction per variant. Extraction timing is shared by those rows and includes '
        'image loading, transfers, both pooling modes, padding and feature extraction. No warm-up exclusion; this is not a latency leaderboard.</p>'
        '<p>This changes inference preprocessing only. It does not establish how training with square crops would perform, '
        'nor explain the entire specialist versus joint gap. Only calibration is evaluated; no final-test embeddings, feature dumps '
        'or new model weights are saved.</p>')
    print(f'Report: {directory / "summary.html"}', flush=True)
