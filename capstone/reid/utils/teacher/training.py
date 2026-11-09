"""Joint pretrained teacher fine-tuning, calibration selection, and step resume."""

import csv
import math
from pathlib import Path
import random
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.full_frame.training import _pair_collate
from .data import build
from .model import CoDINOJoint
from .retrieval import evaluate
from .settings import read, resolve


def total_loss(losses, reid_weight):
    values = []
    for name, value in losses.items():
        if 'loss' not in name:
            continue
        value = sum(item.mean() for item in value) if isinstance(value, (tuple, list)) else value.mean()
        if name in ('loss_identity', 'loss_triplet'):
            value = value * reid_weight
        values.append(value)
    return sum(values)


def optimizer_for(model, settings):
    spec = settings['training']
    backbone, heads = [], []
    for name, parameter in model.named_parameters():
        (backbone if name.startswith('detector.backbone.') else heads).append(parameter)
    return torch.optim.AdamW([{'params': heads, 'lr': spec['learning_rate']},
                              {'params': backbone, 'lr': spec['backbone_learning_rate']}],
                             weight_decay=spec['weight_decay'])


def backward_step(model, optimizer, scaler, loss):
    if not torch.isfinite(loss):
        raise RuntimeError(f'Non-finite teacher loss: {loss.item()}')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    if not torch.isfinite(norm):
        if not scaler.is_enabled():
            raise RuntimeError('Non-finite teacher gradients; checkpoint was not overwritten')
        # GradScaler skips this update and reduces its scale on overflow.
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        print('AMP overflow: skipped optimizer update and reduced loss scale', flush=True)
        return False
    scaler.step(optimizer)
    scaler.update()
    return True


def append_log(path, record):
    fields = ('run_id', 'event', 'epoch', 'step', 'kitti_loss', 'vrai_loss', 'calibration_map', 'calibration_rank1')
    path.parent.mkdir(parents=True, exist_ok=True)
    first = not path.is_file()
    with path.open('a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if first:
            writer.writeheader()
        writer.writerow(record)


def train():
    settings, root = read()
    spec = settings['training']
    if not torch.cuda.is_available():
        raise RuntimeError('Run teacher training on the CUDA GPU host')
    device = torch.device('cuda')
    random.seed(spec['seed']); np.random.seed(spec['seed']); torch.manual_seed(spec['seed'])
    kitti, vrai, calibration, _ = build(settings, root)
    offset = len(kitti.identity_to_label)
    checkpoint = resolve(root, spec['checkpoint'])
    resume = torch.load(checkpoint, map_location='cpu') if checkpoint.is_file() else None
    if resume is not None:
        validate_checkpoint(resume, settings, kitti, vrai)
    model = CoDINOJoint(settings, root, offset + len(vrai.identities), pretrained=resume is None).to(device)
    optimizer = optimizer_for(model, settings)
    scaler = torch.cuda.amp.GradScaler(enabled=spec['mixed_precision'], init_scale=1024)
    epoch_start = step_start = 0
    best_score = -1.0
    run_id = uuid.uuid4().hex
    if resume is not None:
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        scaler.load_state_dict(resume['scaler'])
        epoch_start, step_start = resume['epoch'], resume['step']
        best_score, run_id = resume['best_score'], resume['run_id']
        for key in ('model', 'optimizer', 'scaler'):
            del resume[key]
        print(f'Resuming teacher epoch {epoch_start + 1}, step {step_start + 1}', flush=True)
    print(f'Co-DINO-Inst ViT-L teacher: {len(kitti)} KITTI pairs / {offset} track IDs; '
          f'{len(vrai.identities)} VRAI training IDs; {len(calibration[0])} calibration queries', flush=True)
    rng = random.Random()

    def save(epoch, step, finished=False):
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                   'settings': settings, 'identity_to_label': kitti.identity_to_label,
                   'vrai_identity_to_label': vrai.identity_to_label, 'epoch': epoch, 'step': step,
                   'best_score': best_score, 'run_id': run_id, 'finished': finished,
                   'sampler_rng': rng.getstate(), 'torch_rng': torch.get_rng_state(),
                   'cuda_rng': torch.cuda.get_rng_state_all(), 'numpy_rng': np.random.get_state(),
                   'python_rng': random.getstate()}
        atomic_save(payload, checkpoint)
        print(f'Saved teacher: {epoch} complete epochs, current step {step}', flush=True)

    for epoch in range(epoch_start, spec['epochs']):
        loader = DataLoader(kitti, batch_size=spec['batch_pairs'], shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(spec['seed'] + epoch), collate_fn=_pair_collate)
        rng.seed(spec['seed'] + 10000 + epoch)
        if resume is not None and epoch == epoch_start and step_start:
            rng.setstate(resume['sampler_rng'])
            torch.set_rng_state(resume['torch_rng'].byte())
            torch.cuda.set_rng_state_all([item.byte() for item in resume['cuda_rng']])
            np.random.set_state(resume['numpy_rng']); random.setstate(resume['python_rng'])
        print(f'Epoch {epoch+1}/{spec["epochs"]}: {model.phase(epoch)}', flush=True)
        model.train()
        for step, (images, targets) in enumerate(loader, 1):
            if epoch == epoch_start and step <= step_start:
                continue
            factor = 0.1 + 0.9 * (1 + math.cos(math.pi * (epoch + step / len(loader)) / spec['epochs'])) / 2
            optimizer.param_groups[0]['lr'] = spec['learning_rate'] * factor
            optimizer.param_groups[1]['lr'] = spec['backbone_learning_rate'] * factor
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=spec['mixed_precision']):
                losses = model.kitti_losses([image.to(device) for image in images], targets)
                kitti_loss = total_loss(losses, spec['reid_weight'])
            backward_step(model, optimizer, scaler, kitti_loss)
            kitti_value = kitti_loss.item()
            del losses, kitti_loss, images, targets
            vrai_value = None
            if step % spec['kitti_steps_per_vrai_step'] == 0:
                optimizer.zero_grad(set_to_none=True)
                images, local_labels = vrai.sample(rng, spec['identities_per_batch'])
                labels = local_labels.to(device) + offset
                with torch.cuda.amp.autocast(enabled=spec['mixed_precision']):
                    embeddings = model.crop_embeddings([image.to(device) for image in images])
                    losses = model.reid.losses(embeddings, labels, spec['triplet_margin'])
                    loss = total_loss(losses, spec['reid_weight'])
                backward_step(model, optimizer, scaler, loss)
                vrai_value = loss.item()
                del images, labels, embeddings, losses, loss
            if step % 50 == 0 or step == len(loader):
                print(f'Epoch {epoch+1}/{spec["epochs"]} step {step}/{len(loader)} '
                      f'KITTI loss={kitti_value:.3f}, VRAI loss={vrai_value}', flush=True)
                append_log(resolve(root, spec['log']), {'run_id': run_id, 'event': 'train', 'epoch': epoch+1,
                           'step': step, 'kitti_loss': kitti_value, 'vrai_loss': vrai_value})
            if step % spec['save_every_steps'] == 0:
                save(epoch, step)
        print('Evaluating fixed VRAI calibration IDs for checkpoint selection', flush=True)
        metrics = evaluate(model, calibration, device, mixed_precision=spec['mixed_precision'])
        append_log(resolve(root, spec['log']), {'run_id': run_id, 'event': 'calibration', 'epoch': epoch+1,
                   'step': 0, 'calibration_map': metrics['map'], 'calibration_rank1': metrics['rank1']})
        print(f'Calibration: Rank-1 {metrics["rank1"]:.2%}, mAP {metrics["map"]:.2%}', flush=True)
        if metrics['map'] > best_score:
            best_score = metrics['map']
            atomic_save({'model': model.state_dict(), 'settings': settings, 'run_id': run_id,
                         'identity_to_label': kitti.identity_to_label, 'vrai_identity_to_label': vrai.identity_to_label,
                         'epoch': epoch+1, 'step': 0, 'best_score': best_score,
                         'selection': 'VRAI calibration mAP; final query/gallery excluded'},
                        resolve(root, spec['best_checkpoint']))
        save(epoch + 1, 0, finished=epoch + 1 == spec['epochs'])
        step_start = 0
        resume = None
    if spec['evaluate_after_train']:
        from .evaluation import run
        run()


def atomic_save(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    torch.save(payload, temporary)
    temporary.replace(destination)


def validate_checkpoint(saved, settings, kitti=None, vrai=None):
    previous = saved['settings']
    for key in ('source', 'pretrained', 'data', 'model'):
        if previous[key] != settings[key]:
            raise ValueError(f'Teacher checkpoint {key} differs from configs/teacher.yaml')
    for key in ('epochs', 'frozen_backbone_epochs', 'identities_per_batch', 'batch_pairs',
                'kitti_steps_per_vrai_step', 'learning_rate', 'backbone_learning_rate',
                'weight_decay', 'triplet_margin', 'reid_weight', 'seed', 'calibration_max_ids', 'mixed_precision'):
        if previous['training'].get(key) != settings['training'].get(key):
            raise ValueError(f'Teacher checkpoint training.{key} differs')
    if kitti is not None and saved['identity_to_label'] != kitti.identity_to_label:
        raise ValueError('KITTI training identity mapping changed')
    if vrai is not None and saved['vrai_identity_to_label'] != vrai.identity_to_label:
        raise ValueError('VRAI training identity mapping changed')
