"""Full VRAI/VRU test retrieval and frozen-threshold rejection; no training."""
import argparse
from copy import deepcopy
import csv
import fcntl
import gc
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml
from utils.tracking.common import ROOT,load,save
from utils.teacher.assets import sha256
from utils.reid_teacher.model import build
from utils.reid_teacher.data import transform
from utils.experiments.appearance.evaluation import retrieval
from utils.experiments.aerial_training.data import read
from utils.benchmarking.vru import single_gallery_metrics
from utils.vehicle_reid.operating_points import statistics


def records(path,domain):
    with path.open() as file:rows=list(csv.DictReader(file))
    return [{**row,'id':domain+':'+row['id']} for row in rows]


def prepare(config,training):
    queue=yaml.safe_load((ROOT/training['output']/'queue.yaml').read_text())
    if queue['status']!='complete':raise ValueError('Finish all mixed aerial training before test evaluation')
    settings,data,signature=read(training)
    models={}
    for name in config['variants']:
        report=yaml.safe_load((ROOT/training['output']/name/'report.yaml').read_text())
        if not report['finished']:raise ValueError('Incomplete training variant')
        path=ROOT/training['checkpoint_dir']/name/'best.pt'
        models[name]=dict(checkpoint=str(path.relative_to(ROOT)),sha256=sha256(path),epoch=report['best_epoch'],
                          development_macro_map=report['best']['macro_map'])
    manifest=dict(models=models,selection='All checkpoints fixed by macro development mAP before test evaluation',
        training_signature=signature,test_manifests={key:sha256(ROOT/value) for key,value in training['data'].items() if '_test_' in key},
        limit_identities=config.get('limit_identities'))
    output=ROOT/config['output'];output.mkdir(parents=True,exist_ok=True)
    path=output/'selection.yaml'
    if path.exists() and yaml.safe_load(path.read_text())!=manifest:raise ValueError('Frozen test selection changed; use new output/state paths')
    path.write_text(yaml.safe_dump(manifest))
    return settings,data,manifest


def test_data(training,domain,train,limit=None):
    query=records(ROOT/training['data'][domain+'_test_query'],domain)
    gallery=records(ROOT/training['data'][domain+'_test_gallery'],domain)
    if limit:
        ids=set(sorted({row['id'] for row in query})[:limit])
        query=[row for row in query if row['id'] in ids];gallery=[row for row in gallery if row['id'] in ids]
    train_ids={row['id'] for row in train};test_ids={row['id'] for row in query}
    if train_ids&test_ids or test_ids!={row['id'] for row in gallery}:raise ValueError('Train/test identity overlap or invalid query/gallery IDs')
    paths=[{str(Path(row['path']).resolve()) for row in group} for group in [train,query,gallery]]
    if paths[0]&(paths[1]|paths[2]) or paths[1]&paths[2]:raise ValueError('Train/query/gallery images overlap')
    if domain=='vru' and len(test_ids)!=len(gallery):raise ValueError('VRU requires one gallery image per identity')
    if domain=='vrai' and {row['camera'] for row in query}&{row['camera'] for row in gallery}:raise ValueError('VRAI query/gallery cameras overlap')
    return dict(query=query,gallery=gallery)


@torch.inference_mode()
def open_scores(query,gallery,query_ids,gallery_ids,device):
    identities=sorted(set(gallery_ids));unknown=set(identities[::5]);known=[identity for identity in identities if identity not in unknown]
    if len(known)<2:raise ValueError('Not enough known identities')
    first={}
    for index,identity in enumerate(gallery_ids):first.setdefault(identity,index)
    reference=gallery[[first[identity] for identity in known]].to(device)
    best=[];positions=[]
    for start in range(0,len(query),128):
        values,index=(query[start:start+128].to(device)@reference.T).topk(2,dim=1)
        best.append(values.cpu());positions.extend(index[:,0].cpu().tolist())
    top=torch.cat(best).numpy()
    return dict(similarity=top[:,0],margin=top[:,0]-top[:,1],
        correct=np.array([known[index]==identity for index,identity in zip(positions,query_ids)]),
        known=np.array([identity not in unknown for identity in query_ids]),identities=np.asarray(query_ids))


def extract(config,selection,model,data,conversion,domain,name):
    directory=ROOT/config['state_root']/name/domain;directory.mkdir(parents=True,exist_ok=True)
    state_path=directory/'state.pt'
    signed=dict(selection=selection,domain=domain,variant=name,config=config,
        implementation=sha256(Path(__file__)),metric=sha256(ROOT/'utils/benchmarking/vru.py'))
    state=load(state_path) if state_path.exists() else dict(signature=signed,parts={group:[] for group in data},seconds=0.,complete=False)
    if state['signature']!=signed:raise ValueError('Test inputs/code changed; use new state paths')
    if state['complete']:print(f'{name} {domain}: completed test reused',flush=True);return state['result']
    device=next(model.parameters()).device
    def save_state():save(state,state_path)
    for group,rows in data.items():
        start=sum(item['rows'] for item in state['parts'][group]);pending=[];started=time.perf_counter()
        for offset in range(start,len(rows),config['batch_size']):
            images=[]
            for row in rows[offset:offset+config['batch_size']]:
                with Image.open(row['path']) as image:images.append(conversion(image.convert('RGB')))
            with torch.inference_mode():vectors=F.normalize(model(torch.stack(images).to(device)).float(),dim=1).cpu()
            if vectors.shape[1]!=3840 or not torch.isfinite(vectors).all():raise ValueError('Invalid test embeddings')
            pending.append(vectors)
            if (offset//config['batch_size']+1)%10==0:print(f'{name} {domain}/{group}: {min(offset+config["batch_size"],len(rows))}/{len(rows)}',flush=True)
            if len(pending)>=config['checkpoint_batches'] or offset+config['batch_size']>=len(rows):
                part=directory/f'{group}_{len(state["parts"][group]):05d}.pt';bank=torch.cat(pending)
                save(bank,part);state['parts'][group].append(dict(file=part.name,rows=len(bank)))
                state['seconds']+=time.perf_counter()-started;save_state();pending=[];started=time.perf_counter()
    vectors={group:torch.cat([load(directory/item['file']) for item in state['parts'][group]]) for group in data}
    query_ids=[row['id'] for row in data['query']];gallery_ids=[row['id'] for row in data['gallery']]
    if domain=='vru':result=single_gallery_metrics(vectors['query'],query_ids,vectors['gallery'],gallery_ids)
    else:result=retrieval(vectors['query'],vectors['gallery'],query_ids,gallery_ids)
    checkpoint=load(ROOT/selection['models'][name]['checkpoint']);point=checkpoint['best_metrics'][domain]['fit'];del checkpoint
    observed=statistics(open_scores(vectors['query'],vectors['gallery'],query_ids,gallery_ids,device),point['threshold'],point['margin'])
    result.update(variant=name,domain=domain,open_set=observed,extraction_seconds=state['seconds'],
        checkpoint_sha256=selection['models'][name]['sha256'],thresholds_from='development fit identities; unchanged on test')
    state.update(complete=True,result=result);save_state()
    print(f"{name} {domain}: Rank-1 {result['rank1']:.2%}, mAP {result['map']:.2%}",flush=True)
    return result


def worker(config,training,name):
    torch.set_num_threads(config['cpu_threads']);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    settings,train_data,selection=prepare(config,training);spec=selection['models'][name]
    checkpoint=load(ROOT/spec['checkpoint'])
    if checkpoint['signature']['inputs']!=selection['training_signature'] or checkpoint['epoch']!=spec['epoch']:
        raise ValueError('Selected checkpoint provenance differs')
    model=build(settings,len(checkpoint['identity_mapping']),pretrained=False).to(device)
    model.load_state_dict(checkpoint['model']);model.eval().requires_grad_(False);del checkpoint
    conversion=transform(settings['model']['image_size']);results=[]
    for domain in ['vrai','vru']:
        data=test_data(training,domain,train_data['train'],config.get('limit_identities'))
        results.append(extract(config,selection,model,data,conversion,domain,name))
    (ROOT/config['output']/f'{name}.yaml').write_text(yaml.safe_dump(results))


def summarize(config):
    output=ROOT/config['output'];rows=[]
    for name in config['variants']:
        path=output/f'{name}.yaml'
        if path.exists():rows+=yaml.safe_load(path.read_text())
    (output/'report.yaml').write_text(yaml.safe_dump(dict(results=rows,smoke_check=bool(config.get('limit_identities')),
        selection='Frozen development-selected checkpoints; test never chooses an epoch or threshold',
        limitations=['VRAI is the existing local identity-disjoint train-derived test, not official VRAI test',
            'VRU is the existing fixed local one-reference-per-ID test; not camera-filtered',
            'VRAI multi-positive mAP and VRU single-positive mAP are separate protocols',
            'Different domain-specific rejection thresholds were fitted on development only',
            'Retrieval and extraction cost do not establish video recovery or end-to-end FPS'])))
    lines=['# Mixed aerial encoders: held-out retrieval','',
        ('SMOKE CHECK ONLY. ' if config.get('limit_identities') else '')+'Checkpoints and rejection thresholds frozen before test evaluation. No training.','',
        '| Variant | Dataset | Queries | Gallery | Rank-1 | mAP | Correct known acceptance | Unknown FAR |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    flat=[]
    for row in rows:
        o=row['open_set'];lines.append(f"| {row['variant']} | {row['domain']} | {row['queries']} | {row['gallery']} | {100*row['rank1']:.2f} | {100*row['map']:.2f} | {100*o['known_recovery']:.2f} | {100*o['unknown_far']:.2f} |")
        flat.append({key:value for key,value in row.items() if key!='open_set'})
    lines+=['','VRAI and VRU scores use different gallery protocols and must not be ranked against one another. This is crop retrieval, not video return recovery.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    if flat:
        fields=list(dict.fromkeys(key for row in flat for key in row))
        with (output/'summary.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(flat)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/aerial_benchmark.yaml')
    parser.add_argument('--worker',help=argparse.SUPPRESS);parser.add_argument('--check',action='store_true')
    args=parser.parse_args(argv);config=yaml.safe_load((ROOT/args.config).read_text())
    if config['device']!='auto':raise ValueError('Device must remain auto')
    if args.check:config=deepcopy(config);config['output']+='/check16';config['state_root']+='/check16';config['limit_identities']=16
    training=yaml.safe_load((ROOT/config['training_config']).read_text())
    if args.worker:return worker(config,training,args.worker)
    root=ROOT/config['state_root'];root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Aerial test benchmark already running') from None
        prepare(config,training);status=dict(status='running',completed=[],current=None)
        output=ROOT/config['output']
        def progress():
            path=output/'queue.yaml';temporary=path.with_suffix('.partial');temporary.write_text(yaml.safe_dump(status));temporary.replace(path)
        try:
            for name in config['variants']:
                status['current']=name;progress()
                command=[sys.executable,'-m','utils.experiments.aerial_benchmark.command','--config',args.config,'--worker',name]
                if args.check:command.append('--check')
                subprocess.run(command,cwd=ROOT,check=True);status['completed'].append(name);progress();summarize(config)
            status.update(status='complete',current=None);progress()
        except BaseException as error:status.update(status='failed',error=str(error));progress();raise


if __name__=='__main__':main()
