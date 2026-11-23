"""Equal-budget specialist continuations with complete optimizer/RNG resume."""
import csv
import math
import random
import time
import numpy as np
import torch
import yaml
from utils.tracking.common import ROOT,load,save
from utils.reid_teacher.model import build,phase
from utils.reid_teacher.training import optimizer_for
from .data import Sampler
from .evaluation import evaluate,key
from .losses import objective


def run(name,variant,experiment,settings,data,signature,neighbors=None,check=False):
    torch.set_num_threads(experiment['cpu_threads'])
    if not torch.cuda.is_available():raise RuntimeError('Appearance training requires the CUDA worker; device is auto')
    seed=experiment['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    sampler=Sampler(data['train'],settings,variant,experiment,neighbors)
    folder=ROOT/experiment['checkpoint_dir']/name
    destination=folder/'last.pt';best_path=folder/'best.pt'
    output=ROOT/experiment['output']/name;output.mkdir(parents=True,exist_ok=True)
    signed=dict(inputs=signature,name=name,variant=variant)
    previous=load(destination) if destination.exists() and not check else None
    if previous is not None:
        if previous['signature']!=signed:raise ValueError(f'{name}: training inputs/code changed; use a new output/checkpoint directory')
        if previous['finished']:
            print(f'{name}: reusing completed training',flush=True)
            return previous['best_metrics']
    clip=variant.get('backend')=='clip'
    if clip:
        from utils.experiments.appearance_clip.model import build as clip_build
        model=clip_build(experiment,len(sampler.identities)).cuda()
        initial=experiment['clip']['stage1_checkpoint']
    else:
        model=build(settings,len(sampler.identities),pretrained=False).cuda()
        initial=experiment['start_checkpoint']
    source=load(ROOT/initial) if previous is None else previous
    mapping={identity:i for i,identity in enumerate(sampler.identities)}
    if source['identity_mapping']!=mapping:raise ValueError('Classifier identity mapping changed')
    model.load_state_dict(source['model'])
    optimizer=optimizer_for(model,settings)
    spec=settings['training']
    scaler=torch.amp.GradScaler('cuda',enabled=spec['mixed_precision'],init_scale=1024)
    rng=random.Random(seed);aug_rng=random.Random(seed+1701)
    start_epoch=start_step=0;best_metrics=None;best_epoch=0;elapsed=0.;overflows=0
    if previous is not None:
        optimizer.load_state_dict(previous['optimizer']);scaler.load_state_dict(previous['scaler'])
        start_epoch,start_step=previous['epoch'],previous['step']
        best_metrics,best_epoch=previous['best_metrics'],previous['best_epoch']
        elapsed,overflows=previous['elapsed_seconds'],previous['overflow_skips']
        rng.setstate(previous['sampler_rng']);aug_rng.setstate(previous['augmentation_rng'])
        torch.set_rng_state(previous['torch_rng'].cpu().byte())
        torch.cuda.set_rng_state_all([value.cpu().byte() for value in previous['cuda_rng']])
        random.setstate(previous['python_rng']);np.random.set_state(previous['numpy_rng'])
    del source,previous
    started=time.perf_counter()

    def payload(epoch,step,finished=False,full=True):
        state=dict(signature=signed,settings=settings,identity_mapping=mapping,model=model.state_dict(),
            epoch=epoch,step=step,finished=finished,best_metrics=best_metrics,best_epoch=best_epoch,
            elapsed_seconds=elapsed+time.perf_counter()-started,overflow_skips=overflows)
        if full:
            state.update(optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),sampler_rng=rng.getstate(),
                augmentation_rng=aug_rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(),numpy_rng=np.random.get_state())
        return state

    def record(epoch,metrics):
        path=output/'calibration.csv'
        row=dict(epoch=epoch,rank1=metrics['closed_set']['rank1'],map=metrics['closed_set']['map'],
            eligible=metrics['eligible'],**metrics['selection'])
        first=not path.exists()
        with path.open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            if first:writer.writeheader()
            writer.writerow(row)
        print(f"{name} calibration epoch {epoch}: R1 {metrics['closed_set']['rank1']:.2%}, mAP {metrics['closed_set']['map']:.2%}, "
              f"selection known {metrics['selection']['known_recovery']:.2%}, unknown FAR {metrics['selection']['unknown_far']:.2%}, "
              f"wrong known {metrics['selection']['wrong_known_rate']:.2%}; eligible={metrics['eligible']}",flush=True)

    if best_metrics is None and not check:
        best_metrics=evaluate(model,data,settings,experiment)
        record(0,best_metrics)
        save(payload(0,0,full=False),best_path)
        save(payload(0,0),destination)
    epochs=[0] if check else range(start_epoch,spec['epochs'])
    for epoch in epochs:
        if clip:
            from utils.experiments.appearance_clip.model import phase as clip_phase
            clip_phase(model)
        else:phase(model,epoch,settings)
        model.train()
        beginning=start_step if epoch==start_epoch and not check else 0
        for step in range(beginning+1,(2 if check else spec['steps_per_epoch'])+1):
            factor=.1+.9*(1+math.cos(math.pi*(epoch+step/spec['steps_per_epoch'])/spec['epochs']))/2
            for group,rate in zip(optimizer.param_groups,[spec['learning_rate'],spec['classifier_learning_rate']]):group['lr']=rate*factor
            images,labels=sampler.sample(rng,aug_rng)
            images=images.cuda();labels=labels.cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=spec['mixed_precision']):
                loss,classification,metric=objective(model(images),labels,settings,variant)
            if not torch.isfinite(loss):raise RuntimeError(f'{name}: non-finite loss; previous checkpoint preserved')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm) and not scaler.is_enabled():raise RuntimeError('Non-finite gradients')
            old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            overflows+=int(scaler.get_scale()<old_scale)
            if check and not torch.isfinite(norm):raise RuntimeError('Gradient overflow in integration check')
            if step%25==0 or check or step==spec['steps_per_epoch']:
                print(f'{name} epoch {epoch+1}/{spec["epochs"]} step {step}/{spec["steps_per_epoch"]}: '
                    f'loss {loss.item():.4f}, CE {classification.item():.4f}, metric {metric.item():.4f}, '
                    f'peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB',flush=True)
            del images,labels,loss,classification,metric
            if not check and step%spec['save_every_steps']==0:save(payload(epoch,step),destination)
        if check:continue
        metrics=evaluate(model,data,settings,experiment);record(epoch+1,metrics)
        if key(metrics)>key(best_metrics):
            best_metrics=metrics;best_epoch=epoch+1
            save(payload(epoch+1,0,full=False),best_path)
        save(payload(epoch+1,0,finished=epoch+1==spec['epochs']),destination)
        report=dict(name=name,variant=variant,best_epoch=best_epoch,best=best_metrics,
            epochs_complete=epoch+1,finished=epoch+1==spec['epochs'],pilot=True,
            elapsed_seconds=elapsed+time.perf_counter()-started,overflow_skips=overflows,
            final_evaluation_used=False,original_specialist_modified=False)
        (output/'report.yaml').write_text(yaml.safe_dump(report))
        start_step=0
    if check:
        print(f'{name}: two real training updates passed; no training checkpoint written',flush=True)
    return best_metrics
