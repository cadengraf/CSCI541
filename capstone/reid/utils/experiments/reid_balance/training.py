"""Resume-safe matched identity-only and full multi-task continuation."""
import csv
import gc
import math
import random
import uuid
import torch
from utils.teacher.assets import sha256
from utils.teacher.data import build
from utils.teacher.model import CoDINOJoint
from utils.teacher.retrieval import evaluate
from utils.teacher.training import optimizer_for, total_loss, backward_step, atomic_save, validate_checkpoint
from utils.full_frame.training import _pair_collate
from .settings import read
from .batches import stage_rng, seed_for, pair_order, fingerprint


def append(path, record):
    fields = ('run_id', 'event', 'epoch', 'step', 'batch_key', 'kitti_identity_loss',
              'kitti_auxiliary_loss', 'vrai_loss', 'calibration_rank1', 'calibration_map', 'skipped_updates')
    path.parent.mkdir(parents=True, exist_ok=True)
    first = not path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if first: writer.writeheader()
        writer.writerow(record)


def train(arm, check=False, settings_reader=read):
    settings, root = settings_reader(arm)
    cfg, spec = settings['matched_reid'], settings['training']
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires CUDA')
    torch.manual_seed(cfg['seed'])
    kitti, vrai, calibration, _ = build(settings, root)
    order = pair_order(settings, 0, len(kitti))
    offset = len(kitti.identity_to_label)
    origin = root/cfg['start_checkpoint']
    if sha256(origin) != cfg['start_sha256']:
        raise ValueError('Starting checkpoint hash changed')
    manifests = {key: sha256(root/value) for key, value in settings['data'].items() if key.startswith('vrai_')}
    path = root/spec['checkpoint']
    saved = torch.load(path if path.exists() and not check else origin, map_location='cpu')
    if path.exists() and not check:
        validate_checkpoint(saved, settings, kitti, vrai)
        if saved['manifests'] != manifests:
            raise ValueError('VRAI manifests changed')
        if saved['finished']:
            print(f'{arm}: training already complete', flush=True)
            return
    else:
        if saved.get('manifests') != manifests:
            raise ValueError('Origin VRAI manifests differ from current data')
        for key in ('source', 'pretrained', 'data', 'model'):
            original, current = saved['settings'][key], settings[key]
            if key == 'model' and 'model_variants' in cfg:
                original, current = dict(original), dict(current)
                ignored = cfg.get('warmstart_model_fields', ['reid_spatial_adapter'])
                if not set(ignored) <= {'reid_spatial_adapter', 'crop_geometry', 'crop_size'}:
                    raise ValueError('Only adapter and crop-policy fields may change at warm start')
                for field in ignored:
                    original.pop(field, None)
                    current.pop(field, None)
            if original != current:
                raise ValueError(f'Origin {key} differs')
        if saved['identity_to_label'] != kitti.identity_to_label or saved['vrai_identity_to_label'] != vrai.identity_to_label:
            raise ValueError('Origin identity maps differ')
    model = CoDINOJoint(settings, root, offset+len(vrai.identities), pretrained=False).cuda()
    if path.exists() and not check:
        model.load_state_dict(saved['model'])
    else:
        missing, unexpected = model.load_state_dict(saved['model'], strict=False)
        allowed = {k for k in model.state_dict() if k.startswith('reid_spatial_adapter.')}
        if set(missing) != allowed or unexpected:
            raise ValueError(f'Unexpected warm-start mismatch: missing={missing}, extra={unexpected}')
    optimizer = optimizer_for(model, settings)
    scaler = torch.cuda.amp.GradScaler(enabled=spec['mixed_precision'], init_scale=1024)
    epoch_start = step_start = skipped = 0
    best, run_id = -1., uuid.uuid4().hex
    if path.exists() and not check:
        optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
        epoch_start, step_start, best = saved['epoch'], saved['step'], saved['best_score']
        skipped, run_id = saved['skipped_updates'], saved['run_id']
    del saved

    def payload(epoch, step, finished=False, full=True):
        result = dict(model=model.state_dict(), settings=settings, epoch=epoch, step=step,
                      finished=finished, best_score=best, run_id=run_id, skipped_updates=skipped,
                      manifests=manifests, identity_to_label=kitti.identity_to_label,
                      vrai_identity_to_label=vrai.identity_to_label)
        if full: result.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict())
        return result

    def calibrate(epoch):
        nonlocal best
        values = evaluate(model, calibration, torch.device('cuda'), mixed_precision=spec['mixed_precision'])
        print(f'{arm} epoch {epoch}: calibration Rank-1 {values["rank1"]:.2%}, mAP {values["map"]:.2%}', flush=True)
        append(root/spec['log'], dict(run_id=run_id, event='calibration', epoch=epoch, step=0,
                                     calibration_rank1=values['rank1'], calibration_map=values['map'], skipped_updates=skipped))
        if values['map'] > best:
            best = values['map']; atomic_save(payload(epoch, 0, full=False), root/spec['best_checkpoint'])

    if not check and best < 0:
        calibrate(0); atomic_save(payload(0, 0), path)
    epochs = [0] if check else range(epoch_start, cfg['epochs'])
    for epoch in epochs:
        print(f'{arm} epoch {epoch+1}: {model.phase(epoch)}', flush=True)
        model.train()
        order = pair_order(settings, epoch, len(kitti))
        if check:
            order = [max(range(len(kitti)), key=lambda i: sum(len(kitti.annotations[k]) for k in kitti.pairs[i]))]
            torch.cuda.reset_peak_memory_stats()
        for step, pair_index in enumerate(order, 1):
            if not check and epoch == epoch_start and step <= step_start: continue
            factor = .1+.9*(1+math.cos(math.pi*(epoch+step/cfg['steps_per_epoch'])/cfg['epochs']))/2
            for group, rate in zip(optimizer.param_groups, (spec['learning_rate'], spec['backbone_learning_rate'])):
                group['lr'] = factor*rate
            images, targets = _pair_collate([kitti[pair_index]])
            with stage_rng(settings, epoch, step, 1):
                optimizer.zero_grad(set_to_none=True)
                losses = model.kitti_losses([image.cuda() for image in images], targets)
                identities = {k: v for k, v in losses.items() if k in ('loss_identity', 'loss_triplet')}
                if len(identities) != 2:
                    raise RuntimeError('KITTI update must contain identity and triplet loss')
                auxiliary = {k: v for k, v in losses.items() if k not in identities}
                identity_loss = total_loss(identities, spec['reid_weight'])
                auxiliary_loss = total_loss(auxiliary, spec['reid_weight'])
                selected = identity_loss + auxiliary_loss if cfg.get('use_auxiliary_losses', arm == 'joint') else identity_loss
                updated = backward_step(model, optimizer, scaler, selected)
                skipped += int(not updated)
                if check and not updated: raise RuntimeError('KITTI smoke update overflowed')
                identity_value, auxiliary_value = identity_loss.item(), auxiliary_loss.item()
            del images, targets, losses, identities, auxiliary, identity_loss, auxiliary_loss, selected
            with stage_rng(settings, epoch, step, 2):
                crops, labels = vrai.sample(random.Random(seed_for(settings, epoch, step, 2)), spec['identities_per_batch'])
                optimizer.zero_grad(set_to_none=True)
                embeddings = model.crop_embeddings([image.cuda() for image in crops])
                loss = total_loss(model.reid.losses(embeddings, labels.cuda()+offset, spec['triplet_margin']), spec['reid_weight'])
                updated = backward_step(model, optimizer, scaler, loss)
                skipped += int(not updated)
                if check:
                    if not updated: raise RuntimeError('VRAI smoke update overflowed')
                    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                               for p in model.detector.backbone.parameters() if p.requires_grad)
                    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.detector.mask_head.parameters())
                    if model.reid_spatial_adapter is not None:
                        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                                   for p in model.reid_spatial_adapter.parameters())
                vrai_value = loss.item()
            del crops, labels, embeddings, loss
            key = fingerprint(kitti, pair_index, settings, epoch, step)
            if check:
                print(f'{arm}: both updates passed; batch {key}; identity {identity_value:.3f}, auxiliary {auxiliary_value:.3f}, '
                      f'VRAI {vrai_value:.3f}; peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB', flush=True)
            elif step % 25 == 0 or step == len(order):
                print(f'{arm} {epoch+1}/{cfg["epochs"]} step {step}/{len(order)}: identity {identity_value:.3f}, '
                      f'auxiliary {auxiliary_value:.3f}, VRAI {vrai_value:.3f}', flush=True)
                append(root/spec['log'], dict(run_id=run_id, event='train', epoch=epoch+1, step=step, batch_key=key,
                    kitti_identity_loss=identity_value, kitti_auxiliary_loss=auxiliary_value, vrai_loss=vrai_value, skipped_updates=skipped))
            if not check and step % spec['save_every_steps'] == 0:
                atomic_save(payload(epoch, step), path)
        if not check:
            calibrate(epoch+1)
            atomic_save(payload(epoch+1, 0, finished=epoch+1 == cfg['epochs']), path)
            step_start = 0
    del model, optimizer; gc.collect(); torch.cuda.empty_cache()
    if check: print(f'{arm}: check complete; no checkpoints written', flush=True)
