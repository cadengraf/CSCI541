"""Keep model/data/sampling fixed; pin training implementation for resume."""
from pathlib import Path
from utils.teacher.assets import sha256
from utils.teacher.settings import ROOT
from utils.experiments.reid_balance.settings import read as base_read

CONFIG = ROOT/'configs/joint_metric.yaml'
ARMS = ('control', 'candidate')


def read(arm):
    settings, root = base_read(arm, config_path=CONFIG, arms=ARMS)
    cfg = settings['matched_reid']
    if cfg['device'] != 'auto' or not cfg['use_auxiliary_losses']:
        raise ValueError('Keep automatic devices and joint KITTI supervision')
    if cfg['identities_per_batch'] < 2 or cfg['metric_temperature'] <= 0:
        raise ValueError('AdaSP needs at least two identities and positive temperature')
    files = [*Path(__file__).parent.glob('*.py')]
    files += [ROOT/'utils/experiments/reid_balance'/name for name in ['training.py','batches.py','settings.py']]
    files += [ROOT/'utils/teacher'/name for name in ['model.py','data.py','training.py','retrieval.py']]
    files += [ROOT/'utils/full_frame/multiview.py', ROOT/'utils/experiments/appearance/losses.py']
    cfg['implementation'] = {str(p.relative_to(root)):sha256(p) for p in sorted(files)}
    cfg['vrai_metric'] = 'adasp' if arm=='candidate' else 'triplet'
    return settings, root
