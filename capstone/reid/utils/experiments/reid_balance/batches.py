"""Stateless, identical sampling and augmentation schedules for both arms."""
from contextlib import contextmanager
import hashlib
import random
import torch


def seed_for(settings, epoch, step, stage):
    cfg = settings['matched_reid']
    return cfg['seed'] + 10*(epoch*cfg['steps_per_epoch']+step)+stage


@contextmanager
def stage_rng(settings, epoch, step, stage):
    # Each update receives identical torch randomness across arms and resumes.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed_for(settings, epoch, step, stage))
        yield


def pair_order(settings, epoch, count):
    cfg = settings['matched_reid']
    if cfg['steps_per_epoch'] > count:
        raise ValueError('steps_per_epoch exceeds the available distinct KITTI pairs')
    return random.Random(cfg['seed']+epoch).sample(range(count), cfg['steps_per_epoch'])


def fingerprint(kitti, pair_index, settings, epoch, step):
    # VRAI sampling uses seed_for(stage=2), so this identifies its deterministic
    # sampling schedule without recording bulky per-image tensor caches.
    value = repr((kitti.pairs[pair_index], seed_for(settings, epoch, step, 2)))
    return hashlib.sha256(value.encode()).hexdigest()[:16]
