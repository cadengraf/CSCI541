"""Build two independent experiment configurations from one pinned origin."""
import copy
import yaml
from utils.teacher.settings import ROOT, read as read_teacher

CONFIG = ROOT / 'configs/reid_balance.yaml'
ARMS = ('reid_only', 'joint')


def read(arm, config_path=CONFIG, arms=ARMS):
    if arm not in arms:
        raise ValueError(arm)
    cfg = yaml.safe_load(config_path.read_text())
    settings, root = read_teacher(ROOT/cfg['base_config'])
    settings = copy.deepcopy(settings)
    # The starting weights were distilled, but neither continuation uses KD.
    settings.pop('distillation', None)
    settings['matched_reid'] = dict(cfg, arm=arm, format=1)
    settings['model'].update(cfg.get('model_variants', {}).get(arm, {}))
    spec = settings['training']
    for key in ('epochs', 'identities_per_batch', 'learning_rate', 'backbone_learning_rate',
                'frozen_backbone_epochs', 'save_every_steps', 'seed', 'calibration_max_ids'):
        spec[key] = cfg[key]
    spec.update(batch_pairs=1, kitti_steps_per_vrai_step=1, evaluate_after_train=False,
                checkpoint=f'{cfg["checkpoint_dir"]}/{arm}.last.pt',
                best_checkpoint=f'{cfg["checkpoint_dir"]}/{arm}.best.pt',
                log=f'{cfg["output"]}/{arm}/training.csv')
    tracking = settings['tracking']
    tracking.update(name=f'{cfg.get("experiment_name", "balance")}_{arm}',
                    predictions_root=f'data/kitti_mots/{cfg.get("predictions_name", "reid_balance")}/{arm}',
                    metrics_root=f'{cfg["output"]}/{arm}/kitti_mots')
    if cfg['steps_per_epoch'] < 1 or cfg['epochs'] < 1 or cfg['save_every_steps'] < 1:
        raise ValueError('epochs, steps_per_epoch and save_every_steps must be positive')
    return settings, root
