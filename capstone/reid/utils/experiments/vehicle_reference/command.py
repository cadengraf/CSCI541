"""Complete the three-model reference gate; no training or deployment changes."""
import argparse
import fcntl
import subprocess
import sys
import torch
import yaml
from utils.tracking.common import ROOT,load,save
from utils.experiments.aerial_transfer.evaluation import evaluate
from utils.vehicle_reid.operating_points import scores,calibrate
from .data import signature
from .inference import build_model,open_state,extract,reuse_target,cost


def worker(config,name,phase):
    torch.set_num_threads(config['cpu_threads']);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    selection=load(ROOT/config['output']/'selection.pt');spec=selection['models'][name]
    state,path=open_state(config,name)
    print(f'{phase} {name}: {device}',flush=True)
    if phase=='calibrate':
        if state['calibration_complete']:print('Completed source calibration reused',flush=True);return
        model,conversion=build_model(config,spec,device)
        groups=['calibration_query','calibration_gallery']
        vectors=extract(model,conversion,selection['data'],groups,state,path,config,spec['dimension'])
        ids=[[row['id'] for row in selection['data'][group]] for group in groups]
        point=calibrate(scores(vectors[groups[0]],vectors[groups[1]],*ids),config['false_acceptance_budget'])
        timing=cost(model,conversion,selection['data']['calibration_query'][0],config)
        state.update(calibration_complete=True,operating_point=point,cost=timing);save(state,path)
        print('Frozen source fit: '+str(point),flush=True);return
    if state['complete']:
        (ROOT/config['output']/f'{name}.yaml').write_text(yaml.safe_dump(state['result']))
        print('Completed target score reused',flush=True);return
    points=yaml.safe_load((ROOT/config['output']/'thresholds.yaml').read_text())
    if points[name]!=state['operating_point']:raise ValueError('Frozen source point differs')
    if not reuse_target(config,name,state,path,selection):
        model,conversion=build_model(config,spec,device)
        extract(model,conversion,selection['data'],['target_query','target_gallery'],state,path,config,spec['dimension'])
    vectors={group:torch.cat(state['parts']['target_'+group]) for group in ['query','gallery']}
    data={group:selection['data']['target_'+group] for group in ['query','gallery']}
    result,decisions=evaluate(vectors,data,{'common_vrai_fit':points[name]},selection['target_protocol'])
    result.update(model=name,checkpoint_sha256=spec['sha256'],cost=state['cost'],calibration=points[name],
                  extraction_seconds=state['seconds'],target_reused_from=state.get('target_reused_from'),training=spec['training'])
    (ROOT/config['output']/f'{name}.yaml').write_text(yaml.safe_dump(result))
    state.update(complete=True,result=result,decisions=decisions);save(state,path)
    print(f'Target retrieval: {result["retrieval"]}',flush=True)


def freeze_points(config):
    points={}
    for name in config['models']:
        state,_=open_state(config,name)
        if not state['calibration_complete']:raise ValueError('Complete all source fits before target scoring')
        points[name]=state['operating_point']
    path=ROOT/config['output']/'thresholds.yaml'
    if path.exists() and yaml.safe_load(path.read_text())!=points:raise ValueError('Frozen thresholds changed')
    if not path.exists():path.write_text(yaml.safe_dump(points))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/vehicle_reference.yaml')
    parser.add_argument('--worker',help=argparse.SUPPRESS);parser.add_argument('--phase',choices=['calibrate','target'],help=argparse.SUPPRESS)
    args=parser.parse_args(argv);config=yaml.safe_load((ROOT/args.config).read_text())
    if config['device']!='auto':raise ValueError('Device must be auto')
    if args.worker:return worker(config,args.worker,args.phase)
    root=ROOT/config['state_root'];root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Reference comparison already running') from None
        output=ROOT/config['output'];output.mkdir(parents=True,exist_ok=True)
        status=dict(status='preparing',completed=[],current=None)
        def progress():
            temporary=output/'queue.partial';temporary.write_text(yaml.safe_dump(status));temporary.replace(output/'queue.yaml')
        progress()
        try:
            signature(config)
            for phase in ['calibrate','target']:
                if phase=='target':freeze_points(config)
                for name in config['models']:
                    status.update(status='running',current=f'{phase}:{name}');progress()
                    subprocess.run([sys.executable,str(ROOT/'benchmark.py'),'vehicle-reference','--config',args.config,
                                    '--worker',name,'--phase',phase],cwd=ROOT,check=True)
                    status['completed'].append(f'{phase}:{name}');progress()
            from .report import summarize,render
            summarize(config);render(config);status.update(status='complete',current=None);progress()
        except BaseException as error:
            status.update(status='failed',error=str(error));progress();raise
