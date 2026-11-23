"""Resumable CLIP-ReID prompt learning from frozen training-image features."""
import random
import time
import torch
from PIL import Image
from utils.tracking.common import ROOT,load,save
from utils.reid_teacher.data import transform
from .model import build


def contrastive(first,second,first_labels,second_labels):
    mask=first_labels[:,None]==second_labels[None,:]
    if not mask.any(1).all():raise ValueError('Prompt batch lacks a positive')
    logits=first.float()@second.float().T
    log_probability=logits-logits.logsumexp(1,keepdim=True)
    return -(log_probability*mask).sum(1).div(mask.sum(1)).mean()


def text_bank(model,batch=64):
    with torch.no_grad():
        for start in range(0,model.num_classes,batch):
            labels=torch.arange(start,min(start+batch,model.num_classes),device='cuda')
            with torch.autocast('cuda',dtype=torch.float16):values=model(label=labels,get_text=True)
            model.identity_text[start:start+len(labels)]=values.float()


def prepare(experiment,settings,data,signature):
    spec=experiment['clip'];destination=ROOT/spec['stage1_checkpoint']
    previous=load(destination) if destination.exists() else None
    if previous is not None:
        if previous['signature']!=signature:raise ValueError('CLIP prompt inputs/code changed')
        if previous['finished']:
            print('Reusing completed CLIP prompt learning',flush=True);return
    ids=sorted({row['id'] for row in data['train']});mapping={value:index for index,value in enumerate(ids)}
    seed=experiment['seed'];random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=build(experiment,len(ids))
    path=ROOT/experiment['state_root']/'clip_training_features.pt'
    features=load(path) if path.exists() else None
    if features is not None and features['signature']!=signature:raise ValueError('CLIP image-bank signature changed')
    records=data['train'];chunks=[] if features is None else features['chunks']
    begin=sum(len(chunk) for chunk in chunks);conversion=transform(settings['model']['image_size'])
    batch=settings['evaluation']['batch_size'];start_time=time.perf_counter()
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        for start in range(begin,len(records),batch):
            images=[]
            for row in records[start:start+batch]:
                with Image.open(row['path']) as image:images.append(conversion(image.convert('RGB')))
            with torch.autocast('cuda',dtype=torch.float16):values=model(x=torch.stack(images).cuda(),get_image=True)
            chunks.append(values.cpu().half())
            if (start//batch+1)%40==0 or start+batch>=len(records):
                save(dict(signature=signature,chunks=chunks,complete=start+batch>=len(records)),path)
                print(f'CLIP frozen train-image extraction: {min(start+batch,len(records))}/{len(records)}',flush=True)
    image_features=torch.cat(chunks).float().cuda()
    labels=torch.tensor([mapping[row['id']] for row in records],device='cuda')
    del chunks,features
    for name,parameter in model.named_parameters():parameter.requires_grad_(name=='prompt_learner.cls_ctx')
    optimizer=torch.optim.Adam([model.prompt_learner.cls_ctx],lr=spec['prompt_learning_rate'],weight_decay=settings['training']['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    rng=random.Random(seed);begin=0
    if previous is not None:
        model.load_state_dict(previous['model']);optimizer.load_state_dict(previous['optimizer']);scaler.load_state_dict(previous['scaler'])
        begin=previous['step'];rng.setstate(previous['sampler_rng'])
        torch.set_rng_state(previous['torch_rng'].cpu().byte())
        torch.cuda.set_rng_state_all([value.cpu().byte() for value in previous['cuda_rng']])
    def payload(step,finished=False):
        return dict(signature=signature,identity_mapping=mapping,model=model.state_dict(),optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),sampler_rng=rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            step=step,finished=finished,extraction_and_prompt_seconds=time.perf_counter()-start_time,
            training_images=len(records),prompt_steps=spec['prompt_steps'])
    model.train()
    for step in range(begin+1,spec['prompt_steps']+1):
        indices=torch.tensor(rng.sample(range(len(records)),spec['prompt_batch']),device='cuda')
        target=labels[indices];images=image_features[indices]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):text=model(label=target,get_text=True)
        loss=contrastive(images,text,target,target)+contrastive(text,images,target,target)
        if not torch.isfinite(loss):raise RuntimeError('Non-finite CLIP prompt loss')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_([model.prompt_learner.cls_ctx],5.)
        scaler.step(optimizer);scaler.update()
        if step%25==0:print(f'CLIP prompt step {step}/{spec["prompt_steps"]}: loss {loss.item():.4f}',flush=True)
        if step%50==0:save(payload(step),destination)
    model.eval();text_bank(model)
    save(payload(spec['prompt_steps'],True),destination)
    print('CLIP prompt learning complete; fixed identity-text bank prepared for image fine-tuning',flush=True)
