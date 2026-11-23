"""Real-image forward/backward checks for both CLIP-ReID stages."""
import random
import torch
from utils.experiments.appearance.settings import read
from utils.experiments.appearance.data import Sampler
from .model import build,phase,objective
from .prompt import contrastive,text_bank


def main():
    experiment,settings,data,_=read()
    torch.set_num_threads(experiment['cpu_threads']);torch.manual_seed(experiment['seed'])
    if not torch.cuda.is_available():raise RuntimeError('CUDA worker required')
    sampler=Sampler(data['train'],settings,experiment['variants']['continuation'],experiment)
    model=build(experiment,len(sampler.identities))
    images,labels=sampler.sample(random.Random(42),random.Random(1701));images=images.cuda();labels=labels.cuda()
    model.eval().requires_grad_(False)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):visual=model(x=images,get_image=True).float()
    model.prompt_learner.cls_ctx.requires_grad_(True)
    optimizer=torch.optim.Adam([model.prompt_learner.cls_ctx],lr=experiment['clip']['prompt_learning_rate'])
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):text=model(label=labels,get_text=True)
        loss=contrastive(visual,text,labels,labels)+contrastive(text,visual,labels,labels)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_([model.prompt_learner.cls_ctx],5.)
        if not torch.isfinite(norm):raise RuntimeError('CLIP prompt gradient overflow')
        scaler.step(optimizer);scaler.update()
        print(f'CLIP prompt check {step+1}: loss {loss.item():.4f}',flush=True)
    text_bank(model)
    phase(model);model.train()
    parameters=[value for value in model.parameters() if value.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=experiment['clip']['training']['learning_rate'])
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss,ce,metric=objective(model(images),labels,settings)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(parameters,5.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise RuntimeError('CLIP stage-two gradient/loss failure')
        scaler.step(optimizer);scaler.update()
        print(f'CLIP image check {step+1}: loss {loss.item():.4f}, CE {ce.item():.4f}, triplet {metric.item():.4f}',flush=True)
    model.eval()
    with torch.inference_mode():
        output=model(images[:2])
        if output.shape!=(2,1280) or not torch.isfinite(output).all():raise RuntimeError('Invalid CLIP-ReID inference embedding')
    print('CLIP-ReID two-stage CUDA check passed; no training checkpoints written',flush=True)


if __name__=='__main__':main()
