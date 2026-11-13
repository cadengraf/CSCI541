"""Separate experiment settings and matched no-distillation control paths."""

from pathlib import Path
import copy
from utils.teacher.settings import read as read_teacher

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/distill_joint.yaml'


def read(control=False):
    settings, root = read_teacher(CONFIG)
    if control:
        settings = copy.deepcopy(settings)
        settings['distillation']['projection_weight'] = 0.
        settings['distillation']['relation_weight'] = 0.
        for key in ('checkpoint', 'best_checkpoint', 'log'):
            settings['training'][key] = settings['training'][key].replace('distilled_joint', 'distillation_control')
        settings['tracking']['name'] = 'distillation_control'
        for key in ('predictions_root', 'metrics_root'):
            settings['tracking'][key] = settings['tracking'][key].replace('distilled_joint', 'distillation_control')
    return settings, root
