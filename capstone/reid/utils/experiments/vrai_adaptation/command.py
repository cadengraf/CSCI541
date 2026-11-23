"""User command: adaptation followed by calibration and KITTI evaluation."""


def run(check=False, evaluate_only=False):
    from utils.teacher.launcher import launch
    if check:
        return launch('vrai-adapt-check')
    if not evaluate_only:
        launch('vrai-adapt')
    launch('vrai-adapt-eval')
    from .report import render
    render()
