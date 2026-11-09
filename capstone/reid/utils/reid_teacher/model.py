"""Official pretrained TransReID JPM, adapted to unseen vehicle cameras."""

import collections.abc
from functools import wraps
import sys
from types import ModuleType
import torch
from torch.utils.checkpoint import checkpoint
from utils.teacher.assets import sha256
from utils.tracking.common import ROOT


def build(settings, identities, pretrained=True):
    import subprocess
    source = ROOT / settings['source']['directory']
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != settings['source']['revision']:
        raise ValueError('Unexpected TransReID source revision')
    # Upstream imports an internal Torch compatibility module removed in Torch 2.
    if 'torch._six' not in sys.modules:
        compatibility = ModuleType('torch._six')
        compatibility.container_abcs = collections.abc
        sys.modules['torch._six'] = compatibility
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from config import cfg as default
    from model.make_model import make_model
    cfg = default.clone()
    cfg.merge_from_file(str(source / 'configs/VeRi/vit_transreid_stride.yml'))
    cfg.MODEL.PRETRAIN_CHOICE = 'none'
    cfg.MODEL.SIE_CAMERA = False
    cfg.MODEL.SIE_VIEW = False
    size = settings['model']['image_size']
    cfg.INPUT.SIZE_TRAIN = [size, size]
    cfg.INPUT.SIZE_TEST = [size, size]
    cfg.MODEL.STRIDE_SIZE = [settings['model']['stride']] * 2
    model = make_model(cfg, num_class=identities, camera_num=0, view_num=0)
    if pretrained:
        path = ROOT / settings['pretrained']['path']
        if sha256(path) != settings['pretrained']['sha256']:
            raise ValueError('Vehicle pretrained checkpoint hash mismatch')
        weights = torch.load(path, map_location='cpu', weights_only=True)
        weights = {key.removeprefix('module.'): value for key, value in weights.items()}
        # Recreate identity classifiers for VRAI. Camera/view embeddings describe
        # VeRi's annotation space, which has no mapping to the VRAI camera labels.
        ignored = [key for key in weights if key.startswith('classifier') or key == 'base.sie_embed']
        shared = {key: value for key, value in weights.items() if key not in ignored}
        missing, extra = model.load_state_dict(shared, strict=False)
        if extra or any(not key.startswith('classifier') for key in missing):
            raise ValueError(f'TransReID checkpoint mismatch: {missing}, {extra}')
    if settings['model']['activation_checkpointing']:
        def wrap(module):
            original = module.forward
            @wraps(original)
            def forward(*args, **kwargs):
                if module.training and torch.is_grad_enabled():
                    return checkpoint(original, *args, use_reentrant=False, **kwargs)
                return original(*args, **kwargs)
            module.forward = forward
        for block in model.base.blocks:
            wrap(block)
        wrap(model.b1[0]); wrap(model.b2[0])
    return model


def phase(model, epoch, settings):
    warmup = epoch < settings['training']['classifier_warmup_epochs']
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not warmup or name.startswith('classifier'))
        if 'bottleneck' in name and name.endswith('bias'):
            parameter.requires_grad_(False)
    return 'identity classifier warmup' if warmup else 'full vehicle ReID fine-tuning'


def objective(outputs, labels, settings):
    import torch.nn.functional as F
    def triplet_loss(embedding):
        distance = torch.cdist(F.normalize(embedding.float(), dim=1), F.normalize(embedding.float(), dim=1))
        same = labels[:, None] == labels[None, :]
        same.fill_diagonal_(False)
        different = labels[:, None] != labels[None, :]
        positive = distance.masked_fill(~same, -1).max(1).values
        negative = distance.masked_fill(~different, float('inf')).min(1).values
        valid = same.any(1) & different.any(1)
        return F.relu(positive[valid] - negative[valid] + settings['training']['triplet_margin']).mean()
    logits, features = outputs
    spec = settings['training']
    classification = sum(F.cross_entropy(value.float(), labels, label_smoothing=spec['label_smoothing'])
                         for value in logits) / len(logits)
    metric = sum(triplet_loss(value)
                 for value in features) / len(features)
    return classification + metric, classification, metric
