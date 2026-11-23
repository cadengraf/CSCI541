"""Prepared records, source-balanced identities, and honest camera metadata."""
from collections import defaultdict
from copy import deepcopy
import csv
from pathlib import Path
from PIL import Image
import torch
import yaml
from utils.tracking.common import ROOT
from utils.teacher.assets import sha256
from utils.reid_teacher.data import transform
from utils.experiments.appearance.data import Degradation


def read(config):
    root=ROOT/config['state_root'];audit=yaml.safe_load((root/'prepared.yaml').read_text())
    data={}
    for name,digest in audit['prepared_manifests'].items():
        path=root/name
        if sha256(path)!=digest:raise ValueError('Prepared aerial manifest changed')
        with path.open() as file:data[name.removesuffix('.csv')]=list(csv.DictReader(file))
    settings=deepcopy(yaml.safe_load((ROOT/config['base_config']).read_text()))
    settings['training'].update(config['training'])
    settings['training']['identities_per_batch']=2*config['training']['identities_per_domain']
    if config['device']!='auto' or config['selection']['test_evaluation']!='disabled':
        raise ValueError('Use automatic devices and keep final tests out of training selection')
    if sha256(ROOT/config['warmstart'])!=config['warmstart_sha256']:raise ValueError('Warm-start specialist changed')
    train_ids={row['id'] for row in data['train']};dev_ids=set()
    for name,rows in data.items():
        if name!='train':dev_ids.update(row['id'] for row in rows)
    if train_ids&dev_ids:raise ValueError('Prepared training/development identities overlap')
    files=[*Path(__file__).parent.glob('*.py')]
    files+=[ROOT/'utils/reid_teacher'/name for name in ['model.py','data.py','training.py','evaluation.py']]
    files+=[ROOT/'utils/experiments/appearance'/name for name in ['data.py','losses.py','evaluation.py']]
    files+=[ROOT/'utils/vehicle_reid/operating_points.py']
    signature=dict(config=config,prepared=audit['prepared_manifests'],audit=audit['signature'],
        implementation={str(path.relative_to(ROOT)):sha256(path) for path in sorted(files)})
    return settings,data,signature


class Sampler:
    def __init__(self,rows,settings,config,variant,neighbors=None):
        self.settings=settings;self.config=config;self.variant=variant;self.neighbors=neighbors
        self.grouped=defaultdict(list);self.domains=defaultdict(list)
        for row in rows:self.grouped[row['id']].append(row)
        self.identities=sorted(self.grouped);self.mapping={identity:i for i,identity in enumerate(self.identities)}
        for identity,group in self.grouped.items():
            domains={row['domain'] for row in group}
            if len(domains)!=1 or identity.split(':')[0] not in domains:raise ValueError('Invalid namespaced training identity')
            domain=group[0]['domain'];self.domains[domain].append(identity)
            if len({row['path'] for row in group})<2:raise ValueError('Training identities need distinct images')
            if domain=='vrai' and len({row['camera'] for row in group})<2:raise ValueError('VRAI identity lost camera separation')
            if domain=='vru' and any(row['camera'] for row in group):raise ValueError('Do not invent camera labels for VRU')
        if set(self.domains)!={'vrai','vru'}:raise ValueError('Both aerial training domains are required')
        if variant['hard_sampling']:
            if neighbors is None or set(neighbors)!=set(self.identities):raise ValueError('Incomplete training-only neighbor graph')
            for identity,others in neighbors.items():
                if any(other==identity or other not in self.mapping or other.split(':')[0]!=identity.split(':')[0] for other in others):
                    raise ValueError('Hard neighbors must be different training identities in the same domain')
        self.degrade=Degradation(config['augmentation'])
        self.conversion=transform(settings['model']['image_size'],train=True)

    def choose(self,rng):
        result=[];count=self.config['training']['identities_per_domain']
        for domain in ['vrai','vru']:
            pool=self.domains[domain]
            if len(pool)<count:raise ValueError('Not enough identities per source')
            hard=self.variant['hard_sampling'] and rng.random()<self.config['mining']['hard_batch_probability']
            if hard:
                anchor=rng.choice(pool);n=min(count-1,self.config['mining']['hard_identities_per_domain']-1,len(self.neighbors[anchor]))
                selected=[anchor]+rng.sample(self.neighbors[anchor],n)
                selected+=rng.sample([identity for identity in pool if identity not in selected],count-len(selected))
            else:selected=rng.sample(pool,count)
            result.extend(selected)
        return result

    def records(self,rng):
        result=[];count=self.settings['training']['images_per_identity']
        if count<2 or count%2:raise ValueError('Use an even number of observations per identity')
        for identity in self.choose(rng):
            rows=self.grouped[identity]
            if rows[0]['domain']=='vrai':
                cameras=defaultdict(list)
                for row in rows:cameras[row['camera']].append(row)
                selected=[]
                for camera in rng.sample(sorted(cameras),2):
                    pool=cameras[camera];n=count//2
                    selected+=rng.sample(pool,n) if len(pool)>=n else rng.choices(pool,k=n)
            else:
                selected=rng.sample(rows,count) if len(rows)>=count else rng.sample(rows,len(rows))+rng.choices(rows,k=count-len(rows))
            result.extend((row,self.mapping[identity]) for row in selected)
        return result

    def sample(self,rng,augmentation_rng):
        images=[];labels=[]
        for row,label in self.records(rng):
            with Image.open(row['path']) as raw:image=raw.convert('RGB')
            if self.variant['augmentation']:image=self.degrade(image,augmentation_rng)
            images.append(self.conversion(image));labels.append(label)
        return torch.stack(images),torch.tensor(labels)
