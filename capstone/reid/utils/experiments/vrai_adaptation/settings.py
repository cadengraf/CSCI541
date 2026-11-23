"""Separate outputs and strict unchanged square-crop architecture."""
from utils.teacher.settings import ROOT
from utils.experiments.reid_balance.settings import read as base_read

CONFIG = ROOT/'configs/vrai_adaptation.yaml'
ARM = 'vrai_only'


def read(arm=ARM):
    settings, root = base_read(arm, config_path=CONFIG, arms=(ARM,))
    cfg = settings['matched_reid']
    if cfg['training_mode'] != 'vrai_only' or cfg['use_auxiliary_losses']:
        raise ValueError('VRAI adaptation must exclude KITTI training and auxiliary losses')
    return settings, root
