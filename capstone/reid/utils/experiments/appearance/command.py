"""One resumable queue; each GPU stage runs in a fresh process."""
import argparse
from copy import deepcopy
import fcntl
import subprocess
import sys
import time
import yaml
from utils.tracking.common import ROOT,load
from .settings import read


def variant_for(name,experiment):
    if name=='combined':return experiment['combination']['settings']
    if name=='clip_reid':return dict(augmentation=False,hard_sampling=False,metric='triplet',backend='clip')
    return experiment['variants'][name]


def report(experiment):
    output=ROOT/experiment['output'];output.mkdir(parents=True,exist_ok=True)
    entries={}
    for name in [*experiment['variants'],'combined','clip_reid']:
        path=output/name/'report.yaml'
        if path.exists():entries[name]=yaml.safe_load(path.read_text())
    lines=['# Appearance training pilots','',
        'VRAI training identities only. Rejection thresholds fit on even-indexed calibration identities; checkpoint selection uses odd-indexed calibration identities. Final evaluation is disabled. Original specialist and live defaults are unchanged.','',
        '| Variant | Complete | Selected epoch | Calibration Rank-1 | Calibration mAP | Selection known acceptance | Selection unknown FAR | Selection wrong-known rate | Eligible |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for name,entry in entries.items():
        m=entry['best'];o=m['selection']
        lines.append(f"| {name} | {entry['finished']} | {entry['best_epoch']} | {100*m['closed_set']['rank1']:.2f} | {100*m['closed_set']['map']:.2f} | {100*o['known_recovery']:.2f} | {100*o['unknown_far']:.2f} | {100*o['wrong_known_rate']:.2f} | {m['eligible']} |")
    lines+=['','These are short continuation pilots, not converged leaderboard reproductions. Eligibility is an empirical calibration check, not a statistical guarantee. CLIP uses a different initialization, two-stage training, and a separately reported prompt-training budget.','',
            'Run or resume: `../.venv/bin/python train.py appearance`. Detached logs: `/tmp/appearance-training.log`.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    return entries


def combine_decision(experiment,entries):
    control=entries['continuation']['best'];rules=experiment['selection']
    reasons={}
    for name in experiment['combination']['requires']:
        candidate=entries[name]['best']
        reasons[name]=bool(candidate['eligible'] and
            candidate['selection']['known_recovery']>=control['selection']['known_recovery']+rules['minimum_known_gain'] and
            candidate['closed_set']['map']>=control['closed_set']['map']-rules['maximum_map_drop'])
    return dict(run=bool(experiment['combination']['enabled'] and all(reasons.values())),
                independent_improvements=reasons,criteria=rules,final_evaluation_used=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/appearance_training.yaml')
    parser.add_argument('--worker',help=argparse.SUPPRESS)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args(argv)
    experiment,settings,data,signature=read(args.config)
    output=ROOT/experiment['output'];output.mkdir(parents=True,exist_ok=True)
    if args.worker:
        import torch
        torch.set_num_threads(experiment['cpu_threads'])
        if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable in this worker; device remains auto')
        from .mining import run as mine
        if args.worker=='mining':return mine(experiment,settings,data,signature)
        from .training import run as train
        variant=variant_for(args.worker,experiment)
        neighbors=None
        if variant['hard_sampling']:
            mined=load(ROOT/experiment['state_root']/'mining.pt')
            if not mined['complete'] or mined['signature']!=signature:raise ValueError('Mining signature mismatch')
            neighbors=mined['neighbors']
        if variant.get('backend')=='clip':
            from utils.teacher.assets import sha256
            signature=deepcopy(signature)
            own=ROOT/'utils/experiments/appearance_clip'
            vendor=ROOT/experiment['clip']['source']
            signature['clip_implementation']={str(path.relative_to(ROOT)):sha256(path)
                for path in sorted([*own.glob('*.py'),*vendor.rglob('*.py')])}
            from utils.experiments.appearance_clip.prompt import prepare
            prepare(experiment,settings,data,signature)
            settings=deepcopy(settings);settings['training'].update(experiment['clip']['training'])
        if args.check and args.worker=='adasp':
            from .losses import verify_reference
            verify_reference(experiment)
        return train(args.worker,variant,experiment,settings,data,signature,neighbors,check=args.check)
    lock_path=ROOT/experiment['state_root']/'queue.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('An appearance-training queue is already running') from None
        queue=dict(signature=signature,status='running',completed=[],started=time.time(),current=None)
        def save_queue():
            path=output/'queue.yaml';temporary=path.with_suffix('.partial')
            temporary.write_text(yaml.safe_dump(queue));temporary.replace(path)
        def stage(name,check=False):
            queue['current']=name;save_queue()
            command=[sys.executable,'-m','utils.experiments.appearance.command','--config',args.config,'--worker',name]
            if check:command.append('--check')
            print(f'Starting {name}'+(' integration check' if check else ''),flush=True)
            subprocess.run(command,cwd=ROOT,check=True)
            queue['completed'].append(name);report(experiment);save_queue()
        try:
            if args.check:
                for name in ['continuation','degradation','adasp']:stage(name,check=True)
                queue['status']='checks_complete';save_queue();return
            stage('mining')
            for name in experiment['variants']:stage(name)
            entries=report(experiment);decision=combine_decision(experiment,entries)
            (output/'combination.yaml').write_text(yaml.safe_dump(decision))
            if decision['run']:stage('combined')
            else:print('Combined training skipped: independent improvements did not satisfy the pre-registered criteria',flush=True)
            if experiment.get('clip',{}).get('enabled'):stage('clip_reid')
            queue['status']='complete';queue['current']=None;save_queue();report(experiment)
        except BaseException as error:
            queue['status']='failed';queue['error']=str(error);save_queue();raise


if __name__=='__main__':main()
