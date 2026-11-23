"""Warm-start the existing encoder; expand identity heads without label collisions."""
import torch
from utils.tracking.common import ROOT,load
from utils.reid_teacher.model import build


def initialize(settings,config,mapping,device):
    model=build(settings,len(mapping),pretrained=False).to(device)
    source=load(ROOT/config['warmstart']);weights=source['model']
    shared={name:value for name,value in weights.items() if not name.startswith('classifier')}
    missing,extra=model.load_state_dict(shared,strict=False)
    if extra or any(not key.startswith('classifier') for key in missing):raise ValueError('Warm-start backbone mismatch')
    old=source['identity_mapping'];pairs=[(index,mapping['vrai:'+identity]) for identity,index in old.items() if 'vrai:'+identity in mapping]
    with torch.no_grad():
        if pairs:
            before,after=zip(*pairs)
            for name,value in model.state_dict().items():
                if name.startswith('classifier'):
                    value[list(after)]=weights[name][list(before)].to(device)
    print(f'Warm-start: encoder restored; retained {len(pairs)} VRAI classifier rows; total {len(mapping)} identities',flush=True)
    return model
