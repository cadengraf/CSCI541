"""Separate GPU processes per arm; shared trainer, evaluator and reports."""
from .settings import ARMS, read


def run(check=False, evaluate_only=False):
    from utils.teacher.launcher import launch
    if not evaluate_only:
        for arm in ARMS:
            launch(f'roi-reid-{arm}'+('-check' if check else ''))
    if check:
        print('Raw ROI and dedicated ReID adapter checks passed; no training checkpoints written.', flush=True)
        return
    for arm in ARMS:
        if read(arm)[0]['matched_reid']['evaluate_kitti_after_train']:
            launch(f'roi-reid-{arm}-eval')
    from utils.experiments.reid_balance.report import render
    render(settings_reader=read, arms=ARMS, feature_comparison=True)
