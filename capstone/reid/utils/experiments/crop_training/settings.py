"""Crop-only model overrides; full-frame settings remain identical."""
from utils.teacher.settings import ROOT
from utils.experiments.reid_balance.settings import read as base_read

CONFIG = ROOT/'configs/crop_training.yaml'
ARMS = ('aspect_384', 'square_256')


def read(arm):
    settings, root = base_read(arm, config_path=CONFIG, arms=ARMS)
    spec = settings['model']
    expected = {'aspect_384': ('aspect', 384), 'square_256': ('square', 256)}
    if (spec.get('crop_geometry'), spec['crop_size']) != expected[arm]:
        raise ValueError('Crop training arms must retain their named policies')
    return settings, root
