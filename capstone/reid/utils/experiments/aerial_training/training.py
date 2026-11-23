"""Substantial, balanced aerial fine-tuning with atomic optimizer/RNG checkpoints."""
import csv
import math
import random
import time
import numpy as np
import torch
import yaml
from utils.tracking.common import ROOT,load,save
from utils.reid_teacher.model import phase
from utils.reid_teacher.training import optimizer_for
from utils.experiments.appearance.losses import objective
from .data import Sampler
from .model import initialize
from .evaluation import evaluate,key


def run(name,config,settings,data,signature,neighbors=None,check=False):
    spec=config['training'];variant=config['variants'][name]
    if not torch.cuda.is_available():raise RuntimeError('Run aerial fine-tuning in the CUDA worker; device is auto')
    torch.set_num_threads(config['cpu_threads']);seed=config['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    sampler=Sampler(data['train'],settings,config,variant,neighbors)
    path=ROOT/config['checkpoint_dir']/name/'last.pt';best_path=path.parent/'best.pt'
    output=ROOT/config['output']/name;output.mkdir(parents=True,exist_ok=True)
    signed=dict(inputs=signature,variant=name)
    previous=load(path) if path.exists() and not check else None
    if previous is not None:
        if previous['signature']!=signed:raise ValueError('Training code/config/data changed; use new output/state paths')
        if previous['finished']:print(name+': completed training reused',flush=True);return
    model=initialize(settings,config,sampler.mapping,'cuda')
    optimizer=optimizer_for(model,settings);scaler=torch.amp.GradScaler('cuda',init_scale=1024,enabled=spec['mixed_precision'])
    rng=random.Random(seed);aug_rng=random.Random(seed+1701)
    start_epoch=start_step=0;best=None;best_epoch=0;elapsed=0.;skips=0
    if previous is not None:
        model.load_state_dict(previous['model']);optimizer.load_state_dict(previous['optimizer']);scaler.load_state_dict(previous['scaler'])
        start_epoch,start_step=previous['epoch'],previous['step'];best,best_epoch=previous['best_metrics'],previous['best_epoch']
        elapsed,skips=previous['elapsed_seconds'],previous['overflow_skips']
        rng.setstate(previous['sampler_rng']);aug_rng.setstate(previous['augmentation_rng'])
        torch.set_rng_state(previous['torch_rng'].cpu().byte());torch.cuda.set_rng_state_all([v.cpu().byte() for v in previous['cuda_rng']])
        random.setstate(previous['python_rng']);np.random.set_state(previous['numpy_rng'])
        print(f'{name}: resuming epoch {start_epoch+1}, after step {start_step}',flush=True)
    del previous
    started=time.perf_counter()
    def payload(epoch,step,finished=False,full=True):
        state=dict(signature=signed,settings=settings,identity_mapping=sampler.mapping,model=model.state_dict(),
            epoch=epoch,step=step,finished=finished,best_metrics=best,best_epoch=best_epoch,
            elapsed_seconds=elapsed+time.perf_counter()-started,overflow_skips=skips)
        if full:state.update(optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),sampler_rng=rng.getstate(),
            augmentation_rng=aug_rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(),numpy_rng=np.random.get_state())
        return state
    def record(epoch,metrics):
        row=dict(epoch=epoch,macro_map=metrics['macro_map'],macro_rank1=metrics['macro_rank1'],macro_known=metrics['macro_known_acceptance'])
        for domain in ['vrai','vru']:
            row.update({domain+'_map':metrics[domain]['retrieval']['map'],domain+'_rank1':metrics[domain]['retrieval']['rank1'],
                domain+'_known':metrics[domain]['open_set']['known_recovery'],domain+'_unknown_far':metrics[domain]['open_set']['unknown_far'],
                domain+'_wrong_known':metrics[domain]['open_set']['wrong_known_rate']})
        log=output/'development.csv';first=not log.exists()
        with log.open('a',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(row))
            if first:writer.writeheader()
            writer.writerow(row)
        print(f"{name} development epoch {epoch}: macro mAP {metrics['macro_map']:.2%}, "
              f"VRAI {metrics['vrai']['retrieval']['map']:.2%}, VRU {metrics['vru']['retrieval']['map']:.2%}",flush=True)
    if best is None and not check:
        best=evaluate(model,data,settings,config);record(0,best)
        save(payload(0,0,full=False),best_path);save(payload(0,0),path)
    epochs=[0,1] if check else range(start_epoch,spec['epochs'])
    objective_settings={**settings,'training':{**settings['training'],'metric_temperature':.04}}
    for epoch in epochs:
        print(name+': '+phase(model,epoch,settings),flush=True);model.train()
        begin=start_step if epoch==start_epoch and not check else 0
        for step in range(begin+1,(2 if check else spec['steps_per_epoch'])+1):
            factor=.1+.9*(1+math.cos(math.pi*(epoch+step/spec['steps_per_epoch'])/spec['epochs']))/2
            for group,rate in zip(optimizer.param_groups,[spec['learning_rate'],spec['classifier_learning_rate']]):group['lr']=factor*rate
            images,labels=sampler.sample(rng,aug_rng);images=images.cuda();labels=labels.cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=spec['mixed_precision']):loss,ce,metric=objective(model(images),labels,objective_settings,variant)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss; previous checkpoint retained')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm) and (check or not scaler.is_enabled()):raise RuntimeError('Non-finite gradients')
            old=scaler.get_scale();scaler.step(optimizer);scaler.update();skips+=int(scaler.get_scale()<old)
            if step%25==0 or check:
                print(f'{name} epoch {epoch+1}/{spec["epochs"]} step {step}/{spec["steps_per_epoch"]}: loss {loss.item():.4f}, '
                    f'CE {ce.item():.4f}, metric {metric.item():.4f}, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB',flush=True)
            del images,labels,loss,ce,metric
            if not check and step%spec['checkpoint_steps']==0:save(payload(epoch,step),path)
        if check:continue
        metrics=evaluate(model,data,settings,config);record(epoch+1,metrics)
        if key(metrics)>key(best):best,best_epoch=metrics,epoch+1;save(payload(epoch+1,0,full=False),best_path)
        finished=epoch+1==spec['epochs'];save(payload(epoch+1,0,finished=finished),path)
        report=dict(variant=name,epoch=epoch+1,finished=finished,best_epoch=best_epoch,best=best,
            elapsed_seconds=elapsed+time.perf_counter()-started,overflow_skips=skips,
            test_evaluation_used=False,default_model_changed=False)
        (output/'report.yaml').write_text(yaml.safe_dump(report))
        start_step=0
    if check:print(name+': warmup and full-backbone updates passed; no training checkpoints written',flush=True)
