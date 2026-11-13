"""Continue the joint checkpoint with frozen specialist identity supervision."""

import csv
import math
import random
import uuid
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.teacher.assets import sha256
from utils.teacher.data import build
from utils.teacher.model import CoDINOJoint
from utils.teacher.retrieval import evaluate
from utils.teacher.training import optimizer_for, total_loss, backward_step, atomic_save, validate_checkpoint
from utils.full_frame.training import _pair_collate
from .settings import read
from .targets import load as load_targets, expected
from .data import audit, sample
from .losses import IdentityDistillation, check_geometry


def log(path, record):
    fields = ('run_id', 'event', 'epoch', 'step', 'kitti_loss', 'vrai_loss', 'alignment', 'relation',
              'calibration_rank1', 'calibration_map')
    path.parent.mkdir(parents=True, exist_ok=True)
    first = not path.is_file()
    with path.open('a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if first:
            writer.writeheader()
        writer.writerow(record)


def train(check=False, control=False):
    if check:
        check_geometry()
    settings, root = read(control=control)
    spec, distill = settings['training'], settings['distillation']
    if not torch.cuda.is_available():
        raise RuntimeError('Joint distillation requires the CUDA host')
    random.seed(spec['seed']); np.random.seed(spec['seed']); torch.manual_seed(spec['seed'])
    kitti, vrai, calibration, _ = build(settings, root)
    index, targets = load_targets(settings, root, check=check)
    audit(vrai, index, check=check)
    manifests = {key: sha256(root / value) for key, value in settings['data'].items() if key.startswith('vrai_')}
    offset = len(kitti.identity_to_label)
    path = root / spec['checkpoint']
    recovery = torch.load(path, map_location='cpu') if path.is_file() and not check else None
    if recovery is not None:
        validate_checkpoint(recovery, settings, kitti, vrai)
        if recovery['manifests'] != manifests or recovery['target_source'] != expected(settings, root):
            raise ValueError('Distillation data/teacher targets changed since checkpoint')
    model = CoDINOJoint(settings, root, offset+len(vrai.identity_to_label), pretrained=False).cuda()
    kd = IdentityDistillation(settings['model']['embedding_dim'], distill['teacher_dimensions']).cuda()
    parameters = nn.ModuleDict({'joint': model, 'distillation': kd})
    optimizer = optimizer_for(model, settings)
    optimizer.param_groups[0]['params'].extend(kd.parameters())
    scaler = torch.cuda.amp.GradScaler(enabled=spec['mixed_precision'], init_scale=1024)
    rng = random.Random(spec['seed'])
    run_id, epoch_start, step_start, best = uuid.uuid4().hex, 0, 0, -1.
    if recovery is None:
        source = root / distill['joint_start']
        if sha256(source) != distill['joint_sha256']:
            raise ValueError('Starting joint checkpoint hash differs from config')
        original = torch.load(source, map_location='cpu')
        for key in ('source', 'pretrained', 'data', 'model'):
            if original['settings'][key] != settings[key]:
                raise ValueError(f'Starting joint checkpoint {key} differs')
        if original['identity_to_label'] != kitti.identity_to_label or original['vrai_identity_to_label'] != vrai.identity_to_label:
            raise ValueError('Starting joint checkpoint has another identity mapping')
        model.load_state_dict(original['model'])
        del original
    else:
        model.load_state_dict(recovery['model']); kd.load_state_dict(recovery['distillation_projection'])
        optimizer.load_state_dict(recovery['optimizer']); scaler.load_state_dict(recovery['scaler'])
        run_id, epoch_start, step_start, best = recovery['run_id'], recovery['epoch'], recovery['step'], recovery['best_score']
        rng.setstate(recovery['sampler_rng'])
        torch.set_rng_state(recovery['torch_rng'].cpu().byte())
        torch.cuda.set_rng_state_all([value.cpu().byte() for value in recovery['cuda_rng']])
        np.random.set_state(recovery['numpy_rng']); random.setstate(recovery['python_rng'])
        del recovery
        print(f'Resuming joint distillation epoch {epoch_start+1}, after pair step {step_start}', flush=True)

    def payload(epoch, step, finished=False, full=True):
        saved = {'model': model.state_dict(), 'distillation_projection': kd.state_dict(), 'settings': settings,
                 'identity_to_label': kitti.identity_to_label, 'vrai_identity_to_label': vrai.identity_to_label,
                 'run_id': run_id, 'epoch': epoch, 'step': step, 'finished': finished, 'best_score': best,
                 'manifests': manifests, 'target_source': expected(settings, root)}
        if full:
            saved.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), sampler_rng=rng.getstate(),
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                         numpy_rng=np.random.get_state(), python_rng=random.getstate())
        return saved

    def calibration_score(epoch):
        nonlocal best
        metrics = evaluate(model, calibration, torch.device('cuda'), mixed_precision=spec['mixed_precision'])
        print(f'Joint calibration: Rank-1 {metrics["rank1"]:.2%}, mAP {metrics["map"]:.2%}', flush=True)
        log(root / spec['log'], {'run_id': run_id, 'event': 'calibration', 'epoch': epoch, 'step': 0,
                                'calibration_rank1': metrics['rank1'], 'calibration_map': metrics['map']})
        if metrics['map'] > best:
            best = metrics['map']
            atomic_save(payload(epoch, 0, full=False), root / spec['best_checkpoint'])

    if not check and best < 0:
        calibration_score(0)
        atomic_save(payload(0, 0), path)
    epochs = [0, spec['frozen_backbone_epochs']] if check else range(epoch_start, spec['epochs'])
    print(f'Joint identity transfer: {len(kitti)} KITTI pairs; {len(index)} cached training targets; '
          f'KD weights {distill["projection_weight"]}/{distill["relation_weight"]}', flush=True)
    for epoch in epochs:
        print(f'Epoch {epoch+1}: {model.phase(epoch)}', flush=True)
        model.train(); kd.train()
        loader = DataLoader(kitti, batch_size=spec['batch_pairs'], shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(spec['seed']+epoch), collate_fn=_pair_collate)
        if check:
            # Use a crowded real pair to exercise segmentation memory as well.
            crowded = max(range(len(kitti)), key=lambda i: sum(len(kitti.annotations[key]) for key in kitti.pairs[i]))
            batches = [(1, _pair_collate([kitti[crowded]]))]
        else:
            batches = enumerate(loader, 1)
        for step, (images, labels) in batches:
            if not check and epoch == epoch_start and step <= step_start:
                continue
            factor = .1 + .9 * (1 + math.cos(math.pi * (epoch + step/len(loader)) / spec['epochs'])) / 2
            for group, rate in zip(optimizer.param_groups, (spec['learning_rate'], spec['backbone_learning_rate'])):
                group['lr'] = factor * rate
            optimizer.zero_grad(set_to_none=True)
            if check:
                torch.cuda.reset_peak_memory_stats()
            with torch.cuda.amp.autocast(enabled=spec['mixed_precision']):
                losses = model.kitti_losses([image.cuda() for image in images], labels)
                loss = total_loss(losses, spec['reid_weight'])
            updated = backward_step(parameters, optimizer, scaler, loss)
            kitti_value = loss.item()
            if check:
                if not updated:
                    raise RuntimeError('Gradient overflow during KITTI check')
                print(f'KITTI mask/box/identity update passed: {loss.item():.3f}; '
                      f'{torch.cuda.max_memory_allocated()/2**30:.2f} GiB', flush=True)
            del loss, losses, images, labels
            alignment_value = relation_value = vrai_value = None
            if step % spec['kitti_steps_per_vrai_step'] == 0 or check:
                optimizer.zero_grad(set_to_none=True)
                images, labels, teacher = sample(vrai, index, targets, rng, spec['identities_per_batch'])
                labels = labels.cuda() + offset
                teacher = teacher.cuda()
                with torch.cuda.amp.autocast(enabled=spec['mixed_precision']):
                    embedding = model.crop_embeddings([image.cuda() for image in images])
                    losses = model.reid.losses(embedding, labels, spec['triplet_margin'])
                    alignment, relation = kd(embedding, teacher, labels)
                    loss = total_loss(losses, spec['reid_weight']) + distill['projection_weight']*alignment + distill['relation_weight']*relation
                updated = backward_step(parameters, optimizer, scaler, loss)
                vrai_value, alignment_value, relation_value = loss.item(), alignment.item(), relation.item()
                if check:
                    if not updated or not torch.isfinite(loss):
                        raise RuntimeError('Invalid specialist-guided joint update')
                    assert model.reid.classifier.grad is not None
                    assert any(p.grad is not None for p in model.detector.mask_head.parameters())
                    if epoch:
                        assert any(p.grad is not None for p in model.detector.backbone.parameters())
                    print(f'Distillation update passed: {vrai_value:.3f}; alignment {alignment_value:.4f}, '
                          f'relation {relation_value:.4f}; peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB', flush=True)
                del images, labels, teacher, embedding, losses, loss, alignment, relation
            if not check and (step % 25 == 0 or step == len(loader)):
                print(f'Epoch {epoch+1}/{spec["epochs"]} pair {step}/{len(loader)}: '
                      f'KITTI {kitti_value:.3f}, VRAI {vrai_value}, alignment {alignment_value}, relation {relation_value}', flush=True)
                log(root / spec['log'], {'run_id': run_id, 'event': 'train', 'epoch': epoch+1, 'step': step,
                    'kitti_loss': kitti_value, 'vrai_loss': vrai_value, 'alignment': alignment_value, 'relation': relation_value})
            if not check and step % spec['save_every_steps'] == 0:
                atomic_save(payload(epoch, step), path)
                print(f'Saved distillation checkpoint after pair {step}', flush=True)
        if not check:
            calibration_score(epoch+1)
            atomic_save(payload(epoch+1, 0, finished=epoch+1 == spec['epochs']), path)
            step_start = 0
    if check:
        print('Joint distillation checks passed; no trained checkpoints written', flush=True)
    elif spec['evaluate_after_train']:
        del model, parameters, optimizer
        import gc
        gc.collect(); torch.cuda.empty_cache()
        from utils.teacher.evaluation import run
        run(experiment=settings)
