"""Root command orchestration across specialist and joint model environments."""


def run(mode):
    from utils.teacher.launcher import launch
    if mode.endswith('-eval'):
        return launch(mode)
    from .targets import prepare
    prepare(check=mode.endswith('-check'))
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    launch(mode)
