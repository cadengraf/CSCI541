"""One-command sequential aerial data audit, checks, and training program."""
import argparse
import csv
import fcntl
from pathlib import Path
import subprocess
import sys
import time
import yaml
from utils.tracking.common import ROOT
from utils.teacher.assets import sha256
from .audit import prepare
from .data import read


def reference(config):
    spec=config['msinet_reference']
    if sha256(ROOT/spec['checkpoint'])!=spec['sha256']:raise ValueError('MSINet reference checkpoint changed')
    with (ROOT/spec['archived_scores']).open() as file:rows=list(csv.DictReader(file))
    row=next(row for row in rows if row['model']=="MSINet VRU clean (authors' checkpoint)")
    receipt=dict(checkpoint=spec['checkpoint'],sha256=spec['sha256'],archived_vru_scores=row,
                 caveat=spec['development_caveat'],used_for_training_or_development_selection=False)
    (ROOT/config['output']/'reference.yaml').write_text(yaml.safe_dump(receipt))


def report(config):
    output=ROOT/config['output'];entries={}
    for name in config['variants']:
        path=output/name/'report.yaml'
        if path.exists():entries[name]=yaml.safe_load(path.read_text())
    lines=['# Mixed aerial training','',
        'Balanced VRAI and VRU training. Existing test sets are not evaluated or used for selection. Source labels are namespaced and audited.','',
        '| Variant | Epoch complete | Best epoch | Macro development mAP | VRAI development mAP | VRU development mAP | Finished |',
        '|---|---:|---:|---:|---:|---:|---|']
    for name,row in entries.items():
        m=row['best'];lines.append(f"| {name} | {row['epoch']} | {row['best_epoch']} | {100*m['macro_map']:.2f} | {100*m['vrai']['retrieval']['map']:.2f} | {100*m['vru']['retrieval']['map']:.2f} | {row['finished']} |")
    lines+=['','These are development scores on different source-specific gallery protocols, macro-averaged to balance domains. They are not the existing full VRU/VRAI test scores. Rejection operating points are diagnostics; mAP selects checkpoints.','',
            'Run/resume: `../.venv/bin/python train.py aerial`. Logs: `/tmp/aerial-training.log`. The queue lock prevents duplicate runs.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/aerial_training.yaml')
    parser.add_argument('--worker',help=argparse.SUPPRESS);parser.add_argument('--check',action='store_true')
    args=parser.parse_args(argv);config=yaml.safe_load((ROOT/args.config).read_text())
    output=ROOT/config['output'];output.mkdir(parents=True,exist_ok=True)
    if args.worker:
        import torch
        torch.set_num_threads(config['cpu_threads'])
        if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable in this worker; device is auto')
        settings,data,signature=read(config)
        from .mining import run as mine
        if args.worker=='mining':return mine(config,settings,data,signature)
        from .training import run as train
        neighbors=None
        if config['variants'][args.worker]['hard_sampling']:neighbors=mine(config,settings,data,signature)
        train(args.worker,config,settings,data,signature,neighbors,check=args.check)
        report(config);return
    root=ROOT/config['state_root'];root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('An aerial training queue is already running') from None
        status=dict(status='auditing',current='audit',completed=[],started=time.time())
        def write():
            path=output/'queue.yaml';temporary=path.with_suffix('.partial');temporary.write_text(yaml.safe_dump(status));temporary.replace(path)
        write()
        try:
            audit=prepare(config);reference(config)
            settings,data,signature=read(config)
            status.update(status='running',completed=['audit'],data=audit['usable_training'],signature=signature);write();report(config)
            names=['mixed_baseline','mixed_degradation','mixed_adasp'] if args.check else list(config['variants'])
            for name in names:
                status['current']=name;write();print('Starting '+name+(' GPU integration check' if args.check else ''),flush=True)
                command=[sys.executable,'-m','utils.experiments.aerial_training.command','--config',args.config,'--worker',name]
                if args.check:command.append('--check')
                subprocess.run(command,cwd=ROOT,check=True)
                status['completed'].append(name);write()
            status.update(status='checks_complete' if args.check else 'complete',current=None);write();report(config)
        except BaseException as error:
            status.update(status='failed',error=str(error));write();raise


if __name__=='__main__':main()
