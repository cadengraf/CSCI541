"""Run arms in separate processes so GPU memory is released between models."""


def run(check=False, evaluate_only=False):
    from utils.teacher.launcher import launch
    from .settings import ARMS, read
    if not evaluate_only:
        for arm in ARMS:
            launch(f'reid-balance-{arm}' + ('-check' if check else ''))
    if check:
        print('Both matched arms passed. No training checkpoints written.', flush=True)
        return
    for arm in ARMS:
        cfg = read(arm)[0]['matched_reid']
        if cfg['evaluate_kitti_after_train']:
            launch(f'reid-balance-{arm}-eval')
    from .report import render
    render()
