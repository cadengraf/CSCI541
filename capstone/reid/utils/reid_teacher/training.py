"""Fine-tune the vehicle pretrained specialist with calibration-only selection."""

import csv
import math
import random
import uuid
import numpy as np
import torch
import torch.nn.functional as F
from utils.teacher.assets import sha256
from utils.tracking.common import ROOT, save
from .data import read, partitions, Batches, transform
from .model import build, phase, objective
from .evaluation import score


def signatures(settings):
    return {name: sha256(ROOT / path) for name, path in settings['data'].items()}


def validate(saved, settings, mapping=None):
    old = saved['settings']
    for key in ('source', 'pretrained', 'data', 'model', 'training'):
        if old[key] != settings[key]:
            raise ValueError(f'ReID specialist checkpoint {key} differs from config')
    if saved['manifests'] != signatures(settings):
        raise ValueError('ReID data manifests changed since this checkpoint')
    if mapping is not None and mapping != saved['identity_mapping']:
        raise ValueError('ReID training identities changed')


def optimizer_for(model, settings):
    spec = settings['training']
    shared, classifiers = [], []
    for name, parameter in model.named_parameters():
        (classifiers if name.startswith('classifier') else shared).append(parameter)
    return torch.optim.AdamW([{'params': shared, 'lr': spec['learning_rate']},
                              {'params': classifiers, 'lr': spec['classifier_learning_rate']}],
                             weight_decay=spec['weight_decay'])


def append(settings, row):
    path = ROOT / settings['training']['log']
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ('run_id', 'event', 'epoch', 'step', 'loss', 'classification', 'triplet', 'calibration_rank1', 'calibration_map')
    first = not path.is_file()
    with path.open('a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if first:
            writer.writeheader()
        writer.writerow(row)


def train(check=False):
    settings = read()
    spec = settings['training']
    if not torch.cuda.is_available():
        raise RuntimeError('Run ReID specialist fine-tuning on the CUDA GPU host')
    random.seed(spec['seed']); np.random.seed(spec['seed']); torch.manual_seed(spec['seed'])
    data = partitions(settings)
    sampler = Batches(data['train'], settings)
    manifest_hashes = signatures(settings)
    destination = ROOT / spec['checkpoint']
    recovery = None if check or not destination.is_file() else torch.load(destination, map_location='cpu', weights_only=False)
    if recovery is not None:
        validate(recovery, settings, sampler.mapping)
    model = build(settings, len(sampler.identities), pretrained=recovery is None).cuda()
    optimizer = optimizer_for(model, settings)
    scaler = torch.amp.GradScaler('cuda', enabled=spec['mixed_precision'], init_scale=1024)
    run_id, best, start_epoch, start_step = uuid.uuid4().hex, -1., 0, 0
    rng = random.Random(spec['seed'])
    if recovery is not None:
        model.load_state_dict(recovery['model'])
        optimizer.load_state_dict(recovery['optimizer']); scaler.load_state_dict(recovery['scaler'])
        run_id, best = recovery['run_id'], recovery['best_map']
        start_epoch, start_step = recovery['epoch'], recovery['step']
        rng.setstate(recovery['sampler_rng'])
        torch.set_rng_state(recovery['torch_rng'].cpu().byte())
        torch.cuda.set_rng_state_all([value.cpu().byte() for value in recovery['cuda_rng']])
        random.setstate(recovery['python_rng']); np.random.set_state(recovery['numpy_rng'])
        del recovery
        print(f'Resuming epoch {start_epoch+1}, after step {start_step}', flush=True)
    print(f'Vehicle specialist: {len(data["train"])} images / {len(sampler.identities)} IDs; '
          f'batch {spec["identities_per_batch"] * spec["images_per_identity"]}, two cameras per identity', flush=True)

    def payload(epoch, step, finished=False, full=True):
        state = {'settings': settings, 'manifests': manifest_hashes, 'identity_mapping': sampler.mapping,
                 'model': model.state_dict(), 'run_id': run_id, 'best_map': best,
                 'epoch': epoch, 'step': step, 'finished': finished}
        if full:
            state.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                         sampler_rng=rng.getstate(), torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(), numpy_rng=np.random.get_state())
        return state

    if not check and best < 0:
        metrics = score(model, data['calibration_query'], data['calibration_gallery'],
                        transform(settings['model']['image_size']), settings)
        best = metrics['map']
        save(payload(0, 0, full=False), ROOT / spec['best_checkpoint'])
        append(settings, {'run_id': run_id, 'event': 'calibration', 'epoch': 0, 'step': 0,
                          'calibration_rank1': metrics['rank1'], 'calibration_map': metrics['map']})
        save(payload(0, 0), destination)
        print(f'Preserved pretrained baseline: calibration mAP {best:.2%}', flush=True)
    epochs = [0, spec['classifier_warmup_epochs']] if check else range(start_epoch, spec['epochs'])
    for epoch in epochs:
        print(f'Epoch {epoch+1}: {phase(model, epoch, settings)}', flush=True)
        model.train()
        beginning = start_step if epoch == start_epoch and not check else 0
        steps = range(beginning+1, (1 if check else spec['steps_per_epoch'])+1)
        for step in steps:
            factor = .1 + .9 * (1 + math.cos(math.pi * (epoch + step / spec['steps_per_epoch']) / spec['epochs'])) / 2
            for group, rate in zip(optimizer.param_groups, (spec['learning_rate'], spec['classifier_learning_rate'])):
                group['lr'] = factor * rate
            images, labels = sampler.sample(rng)
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad(set_to_none=True)
            if check:
                torch.cuda.reset_peak_memory_stats()
            with torch.autocast('cuda', dtype=torch.float16, enabled=spec['mixed_precision']):
                loss, classification, metric = objective(model(images), labels, settings)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite specialist loss; checkpoint not overwritten')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not torch.isfinite(norm) and not scaler.is_enabled():
                raise RuntimeError('Non-finite specialist gradients')
            scaler.step(optimizer); scaler.update()
            if check:
                if not torch.isfinite(norm):
                    raise RuntimeError('Gradient overflow during specialist check')
                if epoch:
                    assert model.base.patch_embed.proj.weight.grad is not None
                print(f'Update passed: loss {loss.item():.3f}, CE {classification.item():.3f}, '
                      f'triplet {metric.item():.3f}; peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB', flush=True)
            elif step % 25 == 0 or step == spec['steps_per_epoch']:
                print(f'Epoch {epoch+1}/{spec["epochs"]}, step {step}/{spec["steps_per_epoch"]}: '
                      f'loss {loss.item():.3f}, CE {classification.item():.3f}, triplet {metric.item():.3f}', flush=True)
                append(settings, {'run_id': run_id, 'event': 'train', 'epoch': epoch+1, 'step': step,
                                  'loss': loss.item(), 'classification': classification.item(), 'triplet': metric.item()})
            del loss, classification, metric, images, labels
            if not check and step % spec['save_every_steps'] == 0:
                save(payload(epoch, step), destination)
                print(f'Checkpoint saved after step {step}', flush=True)
        if check:
            continue
        metrics = score(model, data['calibration_query'], data['calibration_gallery'],
                        transform(settings['model']['image_size']), settings)
        print(f'Calibration: Rank-1 {metrics["rank1"]:.2%}, mAP {metrics["map"]:.2%}', flush=True)
        append(settings, {'run_id': run_id, 'event': 'calibration', 'epoch': epoch+1, 'step': 0,
                          'calibration_rank1': metrics['rank1'], 'calibration_map': metrics['map']})
        if metrics['map'] > best:
            best = metrics['map']
            save(payload(epoch+1, 0, full=False), ROOT / spec['best_checkpoint'])
        save(payload(epoch+1, 0, finished=epoch+1 == spec['epochs']), destination)
        start_step = 0
    if check:
        model.eval()
        with torch.no_grad():
            inputs, _ = sampler.sample(rng)
            vectors = F.normalize(model(inputs[:2].cuda()).float(), dim=1)
            assert vectors.shape == (2, 3840) and torch.isfinite(vectors).all()
        print('Specialist check passed; no training checkpoints written', flush=True)
    else:
        del model, optimizer
        torch.cuda.empty_cache()
        from .evaluation import run
        run(final=True)
