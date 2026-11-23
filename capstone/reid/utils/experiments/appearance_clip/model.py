"""Pinned CLIP-ReID model; learned text guides training, image encoder serves inference."""
import importlib
from pathlib import Path
import subprocess
import sys
import torch
import torch.nn.functional as F
from utils.tracking.common import ROOT
from utils.teacher.assets import sha256
from .assets import prepare


def belongs_to_source(module,source):
    """Handle ordinary modules and namespace packages with no __file__."""
    locations=[]
    file=getattr(module,'__file__',None)
    if file:locations.append(file)
    locations.extend(getattr(module,'__path__',[]))
    root=Path(source).resolve()
    return bool(locations) and all(Path(value).resolve().is_relative_to(root) for value in locations)


def build(experiment,identities):
    spec=experiment['clip']
    source=ROOT/spec['source']
    actual=subprocess.check_output(['git','-C',str(ROOT/spec['repository_root']),'rev-parse','HEAD'],text=True).strip()
    if actual!=spec['revision']:raise ValueError('CLIP-ReID source revision changed')
    deps=ROOT/experiment['state_root']/'python_deps'
    if deps.exists():sys.path.insert(0,str(deps))
    for name in ['config','model']:
        if name in sys.modules:
            if not belongs_to_source(sys.modules[name],source):
                raise RuntimeError('CLIP-ReID must run in a fresh worker to avoid upstream import collisions')
    sys.path.insert(0,str(source))
    weights=prepare(experiment)
    from config import cfg as default
    module=importlib.import_module('model.make_model_clipreid')
    # Redirect the upstream loader to a verified workspace file. No home-cache
    # writes and no modifications to upstream code are needed.
    module.clip._download=lambda url: str(weights)
    cfg=default.clone()
    cfg.MODEL.NAME='ViT-B-16';cfg.MODEL.STRIDE_SIZE=[16,16]
    cfg.MODEL.SIE_CAMERA=False;cfg.MODEL.SIE_VIEW=False
    cfg.INPUT.SIZE_TRAIN=[256,256];cfg.INPUT.SIZE_TEST=[256,256]
    cfg.DATASETS.NAMES='veri';cfg.TEST.NECK_FEAT='before'
    model=module.make_model(cfg,identities,0,0).float().cuda()
    model.register_buffer('identity_text',torch.zeros(identities,512,device='cuda'),persistent=True)
    original=model.forward
    def forward(*args,**kwargs):
        result=original(*args,**kwargs)
        if model.training and not kwargs.get('get_text',False) and not kwargs.get('get_image',False):
            logits,features,projected=result
            return logits,features,projected@model.identity_text.T
        return result
    model.forward=forward
    return model


def phase(model):
    for name,parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(('image_encoder','classifier','bottleneck')))
        if name.startswith('bottleneck') and name.endswith('bias'):parameter.requires_grad_(False)


def objective(outputs,labels,settings):
    logits,features,text_scores=outputs
    spec=settings['training']
    ce=sum(F.cross_entropy(value.float(),labels,label_smoothing=spec['label_smoothing']) for value in logits)
    ce=ce+F.cross_entropy(text_scores.float(),labels,label_smoothing=spec['label_smoothing'])
    same=labels[:,None]==labels[None,:]
    same.fill_diagonal_(False)
    different=labels[:,None]!=labels[None,:]
    valid=same.any(1)&different.any(1)
    metric=0
    for feature in features:
        distance=torch.cdist(feature.float(),feature.float())
        positive=distance.masked_fill(~same,-1).max(1).values
        negative=distance.masked_fill(~different,float('inf')).min(1).values
        metric=metric+F.relu(positive[valid]-negative[valid]+spec['triplet_margin']).mean()
    return ce+metric,ce,metric
