"""Shared training schedule; only dedicated ReID spatial processing differs."""
from utils.teacher.settings import ROOT
from utils.experiments.reid_balance.settings import read as base_read

ARMS = ('raw_roi', 'reid_adapter')
CONFIG = ROOT/'configs/roi_reid.yaml'


def read(arm):
    return base_read(arm, config_path=CONFIG, arms=ARMS)
