"""AdaSP with device-safe tensors and explicit balanced identity grouping."""
import torch
import torch.nn.functional as F
from utils.reid_teacher.model import objective as triplet_objective


def adasp(features,labels,temperature=.04):
    # Equivalent to the vendored reference, evaluated in float64 outside AMP.
    # Group by labels explicitly: upstream assumes contiguous balanced groups.
    with torch.autocast(features.device.type,enabled=False):
        unique,counts=torch.unique(labels,return_counts=True)
        if len(unique)<2 or not bool((counts==counts[0]).all()) or int(counts[0])<2:
            raise ValueError('AdaSP requires at least two equally sized identity groups')
        x=F.normalize(features.double()[torch.argsort(labels,stable=True)],dim=1)
        p=len(unique);k=int(counts[0]);scale=1/temperature
        group=torch.eye(p,device=x.device,dtype=x.dtype).repeat_interleave(k,dim=0)
        mask=1-2*(group@group.T)
        block=(torch.exp((x@x.T)*scale*mask)@group)
        eye=torch.eye(p,device=x.device,dtype=x.dtype)
        hh=group.T@block
        hh=hh*(1-eye)+torch.diag(1/torch.diag(hh))
        inverse_pos=1/torch.where(group.bool(),block,torch.ones_like(block))
        he=group.T@(block*(1-group)+inverse_pos*group)
        l=torch.diag(hh).log();s=torch.diag(he).log()
        a=l.detach()/scale;b=s.detach()/scale
        denom=a+b
        # Preserve the sign of the reference's denominator (usually negative).
        safe=torch.where(denom.abs()<1e-12,torch.full_like(denom,1e-12),denom)
        weight=torch.where(a<0,torch.zeros_like(a),2*a*b/safe)
        mixed=he*(1-eye)+torch.diag(torch.exp(l*weight+s*(1-weight)))
        result=-(torch.diag(mixed).log()-mixed.sum(1).log()).mean()
        return result.float()


def objective(outputs,labels,settings,variant):
    if variant.get('backend')=='clip':
        from utils.experiments.appearance_clip.model import objective as clip_objective
        return clip_objective(outputs,labels,settings)
    if variant['metric']=='triplet':return triplet_objective(outputs,labels,settings)
    if variant['metric']!='adasp':raise ValueError('Unsupported metric loss')
    logits,features=outputs
    spec=settings['training']
    ce=sum(F.cross_entropy(value.float(),labels,label_smoothing=spec['label_smoothing']) for value in logits)/len(logits)
    metric=sum(adasp(value,labels,spec['metric_temperature']) for value in features)/len(features)
    return ce+metric,ce,metric


def verify_reference(experiment):
    import importlib.util
    from utils.tracking.common import ROOT
    spec=importlib.util.spec_from_file_location('appearance_official_adasp',ROOT/experiment['adasp_source'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    labels=torch.arange(4,device='cuda').repeat_interleave(4)
    for noise in [.03,.8]:
        centers=torch.randn(4,32,device='cuda')
        values=(centers.repeat_interleave(4,dim=0)+noise*torch.randn(16,32,device='cuda')).double().requires_grad_(True)
        expected=module.AdaSPLoss(temp=.04)(values,labels)
        actual=adasp(values,labels,.04)
        torch.testing.assert_close(actual.double(),expected,rtol=1e-5,atol=1e-6)
        ga=torch.autograd.grad(actual,values,retain_graph=True)[0]
        ge=torch.autograd.grad(expected,values)[0]
        torch.testing.assert_close(ga,ge,rtol=1e-4,atol=1e-6)
    print('AdaSP values and gradients match the vendored reference on CUDA',flush=True)
