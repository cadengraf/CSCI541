"""Strict experiment provenance and the existing identity-disjoint partitions."""
from pathlib import Path
from copy import deepcopy
import yaml
from utils.teacher.assets import sha256
from utils.tracking.common import ROOT
from utils.reid_teacher.data import partitions


def read(path='configs/appearance_training.yaml'):
    experiment=yaml.safe_load((ROOT/path).read_text())
    settings=yaml.safe_load((ROOT/experiment['base_config']).read_text())
    settings=deepcopy(settings)
    settings['training'].update(experiment['training'])
    if experiment['device']!='auto':raise ValueError('Device selection must be auto')
    if experiment['selection']['final_evaluation']!='disabled':raise ValueError('This pilot must not use final evaluation for selection')
    if experiment['training']['identities_per_batch']<2 or experiment['training']['images_per_identity']%2:
        raise ValueError('Invalid balanced cross-camera batch')
    if not 1<=experiment['mining']['hard_identities_per_batch']<=experiment['training']['identities_per_batch']:
        raise ValueError('Invalid hard-identity quota')
    if sha256(ROOT/experiment['start_checkpoint'])!=experiment['start_sha256']:
        raise ValueError('Original specialist checkpoint changed')
    data=partitions(settings)
    files=list(Path(__file__).parent.glob('*.py'))
    files += [ROOT/'utils/reid_teacher'/name for name in ['model.py','data.py','training.py','evaluation.py']]
    files += [ROOT/'utils/vehicle_reid/operating_points.py',ROOT/experiment['adasp_source']]
    signature=dict(experiment=experiment,settings=settings,
        manifests={name:sha256(ROOT/value) for name,value in settings['data'].items()},
        implementation={str(path.relative_to(ROOT)):sha256(path) for path in sorted(files)})
    return experiment,settings,data,signature
