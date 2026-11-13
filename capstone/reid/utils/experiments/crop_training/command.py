"""Train two crop policies, evaluate KITTI/calibration, then cross policies."""
from .settings import ARMS, read


def run(check=False, evaluate_only=False):
    from utils.teacher.launcher import launch
    if not evaluate_only:
        for arm in ARMS:
            launch(f'crop-train-{arm}'+('-check' if check else ''))
    if check:
        print('Both crop-training arms passed; no trained checkpoints written.', flush=True)
        return
    for arm in ARMS:
        launch(f'crop-train-{arm}-eval')
    launch('crop-train-cross-eval')
    from utils.experiments.reid_balance.report import render
    render(settings_reader=read, arms=ARMS, crop_comparison=True)
