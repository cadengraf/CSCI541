"""Explicit inference exports of controlled appearance-training checkpoints."""
from pathlib import Path
import csv
import torch
import torch.nn.functional as F
import yaml
from utils.tracking.common import ROOT,load,save
from utils.teacher.assets import sha256


def export(source,selection_path):
    source=ROOT/source;selection_path=ROOT/selection_path
    digest=sha256(source)
    if selection_path.exists():
        manifest=yaml.safe_load(selection_path.read_text())
        if manifest['training_checkpoint_sha256']!=digest:raise ValueError('Appearance training checkpoint changed')
        if sha256(ROOT/manifest['candidate']['checkpoint'])!=manifest['candidate']['checkpoint_sha256']:
            raise ValueError('Appearance inference export changed')
        return manifest
    state=load(source)
    if not state['finished'] or state['epoch']!=3 or state['step']!=0:
        raise ValueError('Benchmark requires the completed epoch-3 degradation encoder')
    expected=dict(augmentation=True,hard_sampling=False,metric='triplet')
    if state['signature']['variant']!=expected or state['signature']['name']!='degradation':
        raise ValueError('Wrong appearance-training variant')
    destination=selection_path.parent/'epoch3_inference.pt'
    payload=dict(format='appearance_inference_v1',model=state['model'],settings=state['settings'],
        identity_mapping=state['identity_mapping'],training_signature=state['signature'],epoch=state['epoch'],
        training_checkpoint_sha256=digest)
    save(payload,destination)
    candidate=dict(kind='transreid_appearance',model='TransReID VRAI degradation epoch 3',
        checkpoint=str(destination.relative_to(ROOT)),checkpoint_sha256=sha256(destination),dimension=3840)
    manifest=dict(selection='Explicit final-trained degradation epoch, frozen before video evaluation; not clean-selected epoch 0',
        finalized_selection=True,training_checkpoint=str(source.relative_to(ROOT)),training_checkpoint_sha256=digest,
        candidate=candidate,preprocessing=dict(image_size=256,normalization='transreid',crop='same RGB box crop as original'))
    selection_path.parent.mkdir(parents=True,exist_ok=True)
    selection_path.write_text(yaml.safe_dump(manifest))
    return manifest


def load_encoder(device,selection_path):
    from utils.reid_teacher.model import build
    from utils.reid_teacher.data import transform
    manifest=yaml.safe_load(Path(selection_path).read_text());candidate=manifest['candidate']
    if candidate['kind']!='transreid_appearance' or not manifest['finalized_selection']:
        raise ValueError('Not a frozen appearance encoder manifest')
    path=ROOT/candidate['checkpoint']
    if sha256(path)!=candidate['checkpoint_sha256']:raise ValueError('Appearance model checksum mismatch')
    state=load(path)
    if state['format']!='appearance_inference_v1' or state['epoch']!=3:
        raise ValueError('Unexpected appearance inference export')
    if state['training_checkpoint_sha256']!=manifest['training_checkpoint_sha256']:
        raise ValueError('Appearance source provenance mismatch')
    settings=state['settings']
    expected=state['training_signature']['inputs']['manifests']
    for name,value in settings['data'].items():
        if sha256(ROOT/value)!=expected[name]:raise ValueError('Appearance data manifests changed')
    mapping=state['identity_mapping']
    if sorted(mapping.values())!=list(range(len(mapping))):raise ValueError('Invalid classifier identity mapping')
    model=build(settings,len(mapping),pretrained=False)
    model.load_state_dict(state['model']);model.to(device).eval().requires_grad_(False)
    class Frozen(torch.nn.Module):
        def __init__(self,network):super().__init__();self.network=network
        @torch.no_grad()
        def forward(self,images):return F.normalize(self.network(images).float(),dim=1)
    return Frozen(model).to(device).eval().requires_grad_(False),transform(settings['model']['image_size']),manifest
