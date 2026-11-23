"""Controlled aerial video recovery: one resumable command, two frozen methods."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml
from utils.tracking.common import ROOT,load,save
from utils.teacher.assets import sha256
from utils.vehicle_reid.tracklets import build
from .data import freeze,visible_indices
from .metrics import Metrics


def extract(config,selection):
    digest=sha256(ROOT/config['output']/'selection.pt');model=None;conversion=None
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for sequence,case in selection['cases'].items():
        directory=ROOT/config['state_root']/'features'/sequence;directory.mkdir(parents=True,exist_ok=True);path=directory/'state.pt'
        state=load(path) if path.exists() else dict(signature=digest,parts=[],frames=0,seconds=0.,complete=False,device=str(device))
        if state['signature']!=digest:raise ValueError('Recovery extraction signature changed')
        if state['complete']:print(sequence+': completed feature bank reused',flush=True);continue
        if model is None:
            from utils.reid_teacher.evaluation import load_selected
            model,conversion,_=load_selected(device,ROOT/config['selection'])
            print(f'Original specialist extraction on {device}',flush=True)
        pending=[];started=time.perf_counter()
        for index in range(state['frames'],len(case['frames'])):
            row=case['frames'][index];outputs=[]
            if row['observations']:
                with Image.open(row['path']) as source:rgb=source.convert('RGB')
                for offset in range(0,len(row['observations']),config['batch_size']):
                    batch=torch.stack([conversion(rgb.crop(tuple(o['box']))) for o in row['observations'][offset:offset+config['batch_size']]]).to(device)
                    with torch.inference_mode():outputs.append(F.normalize(model(batch).float(),dim=1).cpu())
            vectors=torch.cat(outputs) if outputs else torch.empty((0,3840))
            if vectors.shape!=(len(row['observations']),3840) or not torch.isfinite(vectors).all():raise ValueError('Invalid recovery embeddings')
            pending.append(vectors)
            if len(pending)>=config['checkpoint_frames'] or index+1==len(case['frames']):
                part=directory/f'chunk_{len(state["parts"]):04d}.pt';save(pending,part)
                state['parts'].append(dict(file=part.name,frames=len(pending)));state['frames']=index+1
                state['seconds']+=time.perf_counter()-started;state['complete']=index+1==len(case['frames'])
                save(state,path);pending=[];started=time.perf_counter()
                print(f'{sequence}: {index+1}/{len(case["frames"])} embedding frames committed',flush=True)


def features(config,sequence):
    directory=ROOT/config['state_root']/'features'/sequence;state=load(directory/'state.pt')
    if not state['complete'] or state['signature']!=sha256(ROOT/config['output']/'selection.pt'):raise ValueError('Incomplete or changed recovery feature bank')
    return [frame for part in state['parts'] for frame in load(directory/part['file'])]


def methods(config):
    reference=deepcopy(yaml.safe_load((ROOT/config['reference_config']).read_text())['association'])
    candidate=deepcopy(yaml.safe_load((ROOT/config['candidate_config']).read_text())['association'])
    if reference['engine']!='reference' or candidate['engine']!='aggregated':raise ValueError('Use the fixed reference and temporal query-pool candidate')
    if reference['local']!=candidate['local']:raise ValueError('Local association must be identical')
    return dict(reference=reference,query_pool=candidate)


def replay(config,selection,sequence,scenario,bank):
    case=selection['cases'][sequence];digest=sha256(ROOT/config['output']/'selection.pt')
    path=ROOT/config['state_root']/'replay'/sequence/f'{scenario}.pt'
    state=load(path) if path.exists() else dict(signature=digest,next_frame=0,complete=False,
        algorithms={name:build(spec) for name,spec in methods(config).items()},
        metrics={name:Metrics(config['evaluation'],case['target'],scenario,case['blackout']) for name in methods(config)},
        rows=[],seconds={name:0. for name in methods(config)})
    if state['signature']!=digest:raise ValueError('Recovery replay signature changed')
    if state['complete']:print(f'{sequence}/{scenario}: completed replay reused',flush=True);return state
    if len(bank)!=len(case['frames']):raise ValueError('Feature/frame alignment mismatch')
    for index in range(state['next_frame'],len(case['frames'])):
        row=case['frames'][index];indices,hidden=visible_indices(case,row,scenario)
        boxes=np.asarray([row['observations'][i]['box'] for i in indices],dtype=np.float32).reshape(-1,4)
        vectors=bank[index][indices].numpy();quality=[row['reliable'][i] for i in indices]
        observations=[row['observations'][i] for i in indices];record=dict(indices=indices,hidden=hidden,methods={})
        for name,algorithm in state['algorithms'].items():
            start=time.perf_counter()
            # Neither physical labels nor future frames enter the association interface.
            decisions=algorithm.update(boxes,vectors,quality,row['frame'])
            state['seconds'][name]+=time.perf_counter()-start
            if len(decisions)!=len(indices):raise ValueError('Association output alignment mismatch')
            state['metrics'][name].update(row['frame'],row['truth'],observations,decisions,quality,hidden)
            record['methods'][name]=decisions
        state['rows'].append(record);state['next_frame']=index+1
        if (index+1)%config['checkpoint_frames']==0 or index+1==len(case['frames']):
            if index+1==len(case['frames']):
                state.update(complete=True,summary={name:metric.summary(row['frame']) for name,metric in state['metrics'].items()})
            save(state,path);print(f'{sequence}/{scenario}: replay {index+1}/{len(case["frames"])} committed',flush=True)
    return state


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/vehicle_recovery.yaml');args=parser.parse_args(argv)
    config=yaml.safe_load((ROOT/args.config).read_text())
    if config['device']!='auto':raise ValueError('Device must be auto')
    torch.set_num_threads(config['cpu_threads']);root=ROOT/config['state_root'];root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Video recovery queue already running') from None
        output=ROOT/config['output'];output.mkdir(parents=True,exist_ok=True);status=dict(status='preparing',completed=[],current=None)
        def progress():
            temporary=output/'queue.partial';temporary.write_text(yaml.safe_dump(status));temporary.replace(output/'queue.yaml')
        progress()
        try:
            selection=freeze(config);status.update(status='running',current='embedding extraction');progress();extract(config,selection)
            for sequence in selection['cases']:
                bank=features(config,sequence)
                for scenario in config['scenarios']:
                    status['current']=f'{sequence}/{scenario}';progress();replay(config,selection,sequence,scenario,bank)
                    status['completed'].append(status['current']);progress()
                del bank
            from .report import report,render
            status['current']='video rendering';progress();render(config,selection);report(config,selection)
            status.update(status='complete',current=None);progress()
        except BaseException as error:
            status.update(status='failed',error=str(error));progress();raise
