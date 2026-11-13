"""Train-only memorization through the real shared backbone/mask/ReID path."""
import random
from html import escape

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from utils.teacher.assets import sha256
from utils.teacher.data import build
from utils.teacher.model import CoDINOJoint
from utils.teacher.settings import ROOT, resolve
from utils.teacher.training import validate_checkpoint
from utils.distillation.settings import read
from .reid import scores, write_table


def overfit_network():
    """Fit a fixed train-only cross-view batch through the real joint ROI path."""
    from utils.teacher.training import optimizer_for, total_loss, backward_step
    cfg = yaml.safe_load((ROOT/'configs/reid_diagnostics.yaml').read_text())
    settings, root = read()
    rng = random.Random(cfg['seed']); torch.manual_seed(cfg['seed']); torch.set_num_threads(4)
    kitti, vrai, _, _ = build(settings, root)
    offset = len(kitti.identity_to_label)
    path = resolve(root, cfg['checkpoint'])
    saved = torch.load(path, map_location='cpu')
    validate_checkpoint(saved, settings, kitti, vrai)
    model = CoDINOJoint(settings, root, offset+len(vrai.identities), pretrained=False).cuda()
    model.load_state_dict(saved['model']); del saved
    chosen = rng.sample(sorted(vrai.identities), cfg['network_identities'])
    images, labels, records = [], [], []
    for identity in chosen:
        for camera in sorted(vrai.by_identity[identity])[:2]:
            path_to_image = rng.choice(sorted(vrai.by_identity[identity][camera]))
            with Image.open(path_to_image) as raw:
                images.append(pil_to_tensor(raw.convert('RGB')).float().cuda()/255)
            labels.append(offset+vrai.identity_to_label[identity])
            records.append({'path': str(path_to_image), 'id': identity, 'camera': camera})
    labels = torch.tensor(labels, device='cuda')
    print(model.phase(settings['training']['frozen_backbone_epochs']), flush=True)
    settings['training']['learning_rate'] = cfg['network_head_learning_rate']
    settings['training']['backbone_learning_rate'] = cfg['network_backbone_learning_rate']
    optimizer = optimizer_for(model, settings)
    scaler = torch.cuda.amp.GradScaler(enabled=settings['training']['mixed_precision'], init_scale=1024)
    rows = []
    qa, ga = list(range(0, len(images), 2)), list(range(1, len(images), 2))
    for step in range(cfg['network_steps']+1):
        if step % 10 == 0 or step == cfg['network_steps']:
            model.eval()
            with torch.no_grad():
                emb = model.crop_embeddings(images)
                losses = model.reid.losses(emb, labels, settings['training']['triplet_margin'])
                value = total_loss(losses, 1).item()
                predictions = F.linear(emb, F.normalize(model.reid.classifier, dim=1)).argmax(1)
                row = {'step': step, 'loss': value, 'training_accuracy': 100*float((predictions==labels).float().mean()),
                       **scores(emb.cpu()[qa], [records[i]['id'] for i in qa],
                                emb.cpu()[ga], [records[i]['id'] for i in ga])}
            rows.append(row); print(f'Network train-only overfit: {row}', flush=True)
            directory = resolve(root, cfg['output']); directory.mkdir(parents=True, exist_ok=True)
            write_table(directory/'network_overfit.csv', rows)
        if step == cfg['network_steps']: break
        model.train(); optimizer.zero_grad(set_to_none=True)
        embeddings = model.crop_embeddings(images)
        loss = total_loss(model.reid.losses(embeddings, labels, settings['training']['triplet_margin']), 1)
        updated = backward_step(model, optimizer, scaler, loss)
        if updated:
            if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                       for p in model.detector.backbone.parameters() if p.requires_grad):
                raise RuntimeError('No finite nonzero backbone gradient')
            if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.detector.mask_head.parameters()):
                raise RuntimeError('No mask-head gradient')
        del embeddings, loss
    provenance = {'checkpoint': str(path), 'sha256': sha256(path), 'config': cfg, 'records': records,
                  'scope': 'Training-only memorization, predicted masks, last 4 backbone blocks and ROI/ReID path. No detection/mask loss. Not held-out performance.'}
    (directory/'network_overfit.yaml').write_text(yaml.safe_dump(provenance))
    (directory/'network_overfit.html').write_text('<!doctype html><meta charset="utf-8"><h1>Network overfit diagnostic</h1>'
        '<style>body{font:16px system-ui;margin:32px}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style><p>Fixed training images only. Last four backbone blocks and mask/ReID path updated with classification and triplet losses. '
        'No production checkpoint modified. These scores measure memorization, not generalization. '
        f'Training classification: {rows[0]["training_accuracy"]:.1f}% to {rows[-1]["training_accuracy"]:.1f}%. '
        f'Tiny-gallery Rank-1: {rows[0]["Rank1"]:.1f}% to {rows[-1]["Rank1"]:.1f}%. '
        'See <a href="summary.html">pooling and head diagnostics</a>.</p><table><tr>'
        + ''.join(f'<th>{escape(k)}</th>' for k in rows[0]) + '</tr>'
        + ''.join('<tr>'+''.join(f'<td>{v:.3f}</td>' if isinstance(v, float) else f'<td>{v}</td>' for v in row.values())+'</tr>' for row in rows)+'</table>')
    print('Network diagnostic finished; no model checkpoint written.', flush=True)
