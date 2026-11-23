"""Native pretrained-model adapters, resumable extraction, comparable GPU cost."""
import platform
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from utils.tracking.common import ROOT,load,save
from utils.teacher.assets import sha256
from utils.reid_teacher.data import transform
from .data import image


def build_model(config,spec,device):
    settings=yaml.safe_load((ROOT/config['base_config']).read_text())
    if spec['kind']=='msinet':
        from utils.benchmarking.msinet_baseline import load_model
        model=load_model(ROOT/settings['references']['msinet_source'],ROOT/spec['checkpoint'])
        conversion=transform(256,normalization='imagenet')
    else:
        from utils.reid_teacher.model import build
        checkpoint=load(ROOT/spec['checkpoint'])
        model=build(settings,len(checkpoint['identity_mapping']),pretrained=False)
        model.load_state_dict(checkpoint['model']);conversion=transform(settings['model']['image_size'])
    return model.to(device).eval().requires_grad_(False),conversion


def open_state(config,name):
    path=ROOT/config['state_root']/name/'state.pt';digest=sha256(ROOT/config['output']/'selection.pt')
    state=load(path) if path.exists() else dict(signature=digest,parts={},seconds={},calibration_complete=False,complete=False)
    if state['signature']!=digest:raise ValueError('Reference extraction inputs changed')
    return state,path


@torch.inference_mode()
def extract(model,conversion,data,groups,state,path,config,dimension):
    device=next(model.parameters()).device
    for group in groups:
        rows=data[group];parts=state['parts'].setdefault(group,[]);offset=sum(len(part) for part in parts)
        pending=[];start=time.perf_counter()
        for index in range(offset,len(rows),config['batch_size']):
            batch=torch.stack([conversion(image(row)) for row in rows[index:index+config['batch_size']]]).to(device)
            vectors=F.normalize(model(batch).float(),dim=1).cpu()
            if vectors.shape!=(len(batch),dimension) or not torch.isfinite(vectors).all():raise ValueError('Invalid reference embeddings')
            pending.append(vectors)
            if len(pending)>=config['checkpoint_batches'] or index+len(batch)==len(rows):
                parts.append(torch.cat(pending));state['seconds'][group]=state['seconds'].get(group,0.)+time.perf_counter()-start
                save(state,path);pending=[];start=time.perf_counter()
                print(f'{group}: {min(index+len(batch),len(rows))}/{len(rows)}',flush=True)
    return {group:torch.cat(state['parts'][group]) for group in groups}


def reuse_target(config,name,state,path,selection):
    if name=='msinet':return False
    source=ROOT/config['transfer_state']/name/'state.pt';prior=load(source)
    if not prior['complete'] or prior['signature']!=sha256(ROOT/config['transfer_selection']):raise ValueError('Prior target extraction is incomplete or changed')
    if prior['result']['checkpoint_sha256']!=selection['models'][name]['sha256']:raise ValueError('Prior target weights differ')
    for group in ['query','gallery']:
        parts=prior['parts'][group]
        if sum(len(part) for part in parts)!=len(selection['data']['target_'+group]):raise ValueError('Target feature count mismatch')
        state['parts']['target_'+group]=parts
    state['target_reused_from']=str(source.relative_to(ROOT));save(state,path);return True


@torch.inference_mode()
def cost(model,conversion,row,config):
    device=next(model.parameters()).device;input_image=conversion(image(row)).to(device)
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    result=dict(device=str(device),hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
        torch=str(torch.__version__),precision='float32',scope='Encoder plus L2 normalization on preloaded input; excludes all crop I/O, detection, masks, association and rendering',
        environment='Shared workstation; not an isolated deployment measurement',parameters_loaded=sum(p.numel() for p in model.parameters()),batches=[])
    invoked={}
    def record(module,args,output):
        for parameter in module.parameters(recurse=False):invoked[id(parameter)]=parameter.numel()
    handles=[module.register_forward_hook(record) for module in model.modules()]
    model(input_image[None]);sync()
    for handle in handles:handle.remove()
    result['parameters_in_invoked_modules']=sum(invoked.values())
    for batch_size in config['cost']['batches']:
        batch=input_image[None].repeat(batch_size,1,1,1)
        for _ in range(config['cost']['warmup_iterations']):F.normalize(model(batch).float(),dim=1)
        sync();baseline=torch.cuda.memory_allocated(device) if device.type=='cuda' else None
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        elapsed=[]
        for _ in range(config['cost']['measured_iterations']):
            sync();start=time.perf_counter();F.normalize(model(batch).float(),dim=1);sync();elapsed.append((time.perf_counter()-start)*1000)
        median=float(np.median(elapsed))
        result['batches'].append(dict(batch_size=batch_size,median_ms=median,p95_ms=float(np.quantile(elapsed,.95)),
            images_per_second=1000*batch_size/median,peak_allocated_mb=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
            incremental_peak_mb=(torch.cuda.max_memory_allocated(device)-baseline)/2**20 if device.type=='cuda' else None))
    return result
