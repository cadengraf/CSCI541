"""Co-DINO-Inst ViT-L plus a shared, mask-pooled vehicle identity branch.

Run this module only in the isolated teacher environment. The released detector
and RefineMask heads stay intact. New identity parameters are trained jointly.
"""

import math
import logging
import re
import sys
import warnings
from functools import wraps

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import nms

from .settings import resolve

COCO_CAR = 2  # contiguous 80-class COCO labels; not TorchVision's label 3


def full_precision_attention(detector):
    """The released MMCV CUDA deformable attention kernel accepts FP32 only."""
    from mmcv.ops import MultiScaleDeformableAttention

    def wrap(forward):
        @wraps(forward)
        def execute(*args, **kwargs):
            def convert(value):
                return value.float() if isinstance(value, torch.Tensor) and value.is_floating_point() else value
            with torch.cuda.amp.autocast(enabled=False):
                return forward(*(convert(value) for value in args),
                               **{key: convert(value) for key, value in kwargs.items()})
        return execute

    for module in detector.modules():
        if isinstance(module, MultiScaleDeformableAttention):
            module.forward = wrap(module.forward)


def bootstrap(settings, root):
    location = resolve(root, settings['source']['directory'])
    if not location.is_dir():
        raise FileNotFoundError('Run `python train.py teacher-setup` first')
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))
    warnings.filterwarnings('ignore', message='On January 1, 2023.*', module='mmcv')
    # The pinned upstream source predates removal of NumPy's dtype aliases.
    for name, value in {'int': int, 'float': float, 'bool': bool}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    import projects  # noqa: F401 -- registers official Co-DINO/RefineMask modules


def triplet_loss(features, labels, margin):
    distances = torch.cdist(features.float(), features.float())
    same = labels[:, None] == labels[None, :]
    same.fill_diagonal_(False)
    different = labels[:, None] != labels[None, :]
    valid = same.any(1) & different.any(1)
    if not valid.any():
        return features.sum() * 0
    positive = distances.masked_fill(~same, -1).max(1).values
    negative = distances.masked_fill(~different, float('inf')).min(1).values
    return F.relu(positive[valid] - negative[valid] + margin).mean()


class IdentityHead(nn.Module):
    def __init__(self, dimensions, identities, scale, margin):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 512),
                                        nn.GELU(), nn.Linear(512, dimensions))
        self.classifier = nn.Parameter(torch.randn(identities, dimensions) * 0.01)
        self.scale, self.margin = scale, margin

    def forward(self, features, masks=None):
        if masks is None:
            pooled = features.mean((2, 3))
        else:
            weights = F.interpolate(masks.float(), features.shape[-2:], mode='bilinear', align_corners=False)
            pooled = (features.float() * weights).sum((2, 3)) / weights.sum((2, 3)).clamp_min(1e-5)
        return F.normalize(self.projection(pooled.float()), dim=1)

    def losses(self, embeddings, labels, margin):
        cosine = F.linear(embeddings.float(), F.normalize(self.classifier.float(), dim=1))
        target_margin = F.one_hot(labels, self.classifier.shape[0]).to(cosine.dtype) * self.margin
        logits = self.scale * (cosine - target_margin)
        return {'loss_identity': F.cross_entropy(logits, labels, label_smoothing=0.1),
                'loss_triplet': triplet_loss(embeddings, labels, margin)}


class CoDINOJoint(nn.Module):
    def __init__(self, settings, root, identities, pretrained=True):
        super().__init__()
        bootstrap(settings, root)
        from mmcv import Config
        from mmcv.utils import get_logger
        from mmdet.models import build_detector
        get_logger('mmcv', log_level=logging.WARNING)
        self.settings = settings
        self.config = Config.fromfile(str(resolve(root, settings['source']['directory']) / settings['source']['config']))
        self.detector = build_detector(self.config.model)
        full_precision_attention(self.detector)
        backbone_forward = self.detector.backbone.forward

        def backbone_mixed_precision(*args, **kwargs):
            with torch.cuda.amp.autocast(enabled=settings['training']['mixed_precision']):
                outputs = backbone_forward(*args, **kwargs)
            return tuple(output.float() for output in outputs)

        self.detector.backbone.forward = backbone_mixed_precision
        if pretrained:
            path = resolve(root, settings['pretrained']['path'])
            if not path.is_file():
                raise FileNotFoundError('Run `python train.py teacher-setup` to download official weights')
            checkpoint = torch.load(path, map_location='cpu')
            state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
            state = {key.removeprefix('module.'): value for key, value in state.items()}
            # Upstream EMAHook swaps evaluation weights into the normal keys
            # before saving; ema_* buffers retain the unsmoothed training copy.
            state = {key: value for key, value in state.items() if not key.startswith('ema_')}
            missing, extra = self.detector.load_state_dict(state, strict=False)
            if missing or extra:
                raise ValueError(f'Teacher weights/config mismatch: missing={missing[:10]}, extra={extra[:10]}')
            del checkpoint, state
        model = settings['model']
        self.reid = IdentityHead(model['embedding_dim'], identities,
                                 model['classifier_scale'], model['classifier_margin'])
        self.phase(0)

    def phase(self, epoch):
        joint = epoch >= self.settings['training']['frozen_backbone_epochs']
        blocks = self.settings['model']['unfreeze_last_blocks'] if joint else 0
        for name, parameter in self.detector.backbone.named_parameters():
            match = re.search(r'blocks\.(\d+)\.', name)
            parameter.requires_grad_(bool(match and int(match.group(1)) >= 24 - blocks))
        return f'last {blocks} ViT blocks trainable; shared detection/mask/ReID heads trainable'

    def _prepare(self, images, targets=None, crop=False, augment=False):
        from mmdet.core import BitmapMasks
        tensors, metadata, boxes, masks, labels = [], [], [], [], []
        for index, image in enumerate(images):
            height, width = image.shape[-2:]
            spec = self.settings['model']
            short = spec['crop_size'] if crop else spec['short_side']
            maximum = spec['crop_size'] if crop else spec['max_side']
            scale = min(short / min(height, width), maximum / max(height, width))
            new_h, new_w = round(height * scale), round(width * scale)
            resized = F.interpolate(image[None], size=(new_h, new_w), mode='bilinear', align_corners=False)[0]
            flip = augment and bool(torch.rand(()) < 0.5)
            if flip:
                resized = resized.flip(-1)
            if augment:
                resized = (resized * (0.8 + 0.4 * torch.rand((), device=image.device))).clamp(0, 1)
            mean = image.new_tensor([123.675, 116.28, 103.53])[:, None, None]
            std = image.new_tensor([58.395, 57.12, 57.375])[:, None, None]
            tensors.append((resized * 255 - mean) / std)
            factors = np.array([new_w / width, new_h / height] * 2, dtype=np.float32)
            metadata.append({'img_shape': (new_h, new_w, 3), 'ori_shape': (height, width, 3),
                             'scale_factor': factors, 'flip': flip})
            if targets is not None:
                box = targets[index]['boxes'].to(image.device) * image.new_tensor(factors)
                target_mask = F.interpolate(targets[index]['masks'][:, None].float().to(image.device),
                                            size=(new_h, new_w), mode='nearest')[:, 0]
                if flip:
                    box = torch.stack((new_w - box[:, 2], box[:, 1], new_w - box[:, 0], box[:, 3]), dim=1)
                    target_mask = target_mask.flip(-1)
                boxes.append(box)
                masks.append(BitmapMasks(target_mask.byte().cpu().numpy(), new_h, new_w))
                labels.append(torch.full((len(box),), COCO_CAR, dtype=torch.long, device=image.device))
        pad_h = math.ceil(max(image.shape[-2] for image in tensors) / 32) * 32
        pad_w = math.ceil(max(image.shape[-1] for image in tensors) / 32) * 32
        batch = torch.stack([F.pad(image, (0, pad_w - image.shape[-1], 0, pad_h - image.shape[-2])) for image in tensors])
        for meta in metadata:
            meta['pad_shape'] = (pad_h, pad_w, 3)
            meta['batch_input_shape'] = (pad_h, pad_w)
        return batch, metadata, boxes, labels, masks

    def _refined_masks(self, stages):
        from mmdet.models.losses.cross_entropy_loss import generate_block_target
        previous = stages[1]
        for following in stages[2:]:
            interior = (generate_block_target(previous[:, 0].sigmoid() >= 0.5, boundary_width=1) != 1)[:, None]
            interior = F.interpolate(interior.float(), following.shape[-2:], mode='bilinear', align_corners=True) >= 0.5
            upsampled = F.interpolate(previous, following.shape[-2:], mode='bilinear', align_corners=True)
            previous = torch.where(interior, upsampled, following)
        return previous

    def _roi_identity(self, features, boxes, global_pool=False):
        from mmdet.core import bbox2roi
        rois = bbox2roi(boxes)
        if not len(rois):
            return features[0].new_zeros((0, self.settings['model']['embedding_dim'])), None
        car_labels = torch.full((len(rois),), COCO_CAR, dtype=torch.long, device=rois.device)
        output = self.detector._mask_forward(features, rois, car_labels)
        masks = self._refined_masks(output['stage_instance_preds'])
        embeddings = self.reid(output['mask_feats'], None if global_pool else masks.sigmoid())
        return embeddings, (output, masks)

    @torch.cuda.amp.autocast(enabled=False)
    def kitti_losses(self, images, targets, augment=True):
        batch, metadata, boxes, labels, masks = self._prepare(images, targets, augment=augment)
        captured = []

        def capture(module, inputs, output):
            captured.append(output[-1])

        handle = self.detector.query_head.register_forward_hook(capture)
        try:
            losses = self.detector.forward_train(batch, metadata, gt_bboxes=boxes, gt_labels=labels, gt_masks=masks)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError('Expected one shared Co-DINO encoder pass')
        embeddings, _ = self._roi_identity(captured[0], boxes)
        identities = torch.cat([target['identity_labels'].to(batch.device) for target in targets])
        if len(identities):
            losses.update(self.reid.losses(embeddings, identities, self.settings['training']['triplet_margin']))
        return losses

    @torch.cuda.amp.autocast(enabled=False)
    def crop_embeddings(self, images, global_pool=False):
        # Chunk crops to avoid running the 1500-query decoder on eight images at
        # once. Only encoder/ROI graphs contribute to crop identity gradients.
        outputs = []
        for start in range(0, len(images), 2):
            subset = images[start:start + 2]
            batch, metadata, _, _, _ = self._prepare(subset, crop=True)
            features = self.detector.extract_feat(batch, metadata)
            encoded = self.detector.query_head(features, metadata)[-1]
            boxes = [batch.new_tensor([[0, 0, meta['img_shape'][1], meta['img_shape'][0]]]) for meta in metadata]
            embeddings, _ = self._roi_identity(encoded, boxes, global_pool=global_pool)
            outputs.append(embeddings)
        return torch.cat(outputs)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, images):
        if self.training:
            raise ValueError('Use kitti_losses or crop_embeddings for teacher training')
        predictions = []
        for image in images:
            batch, metadata, _, _, _ = self._prepare([image])
            features = self.detector.extract_feat(batch, metadata)
            results, encoded = self.detector.query_head.simple_test(features, metadata, rescale=False,
                                                                   return_encoder_output=True)
            detections, labels = results[0]
            chosen = torch.where((labels == COCO_CAR) & (detections[:, 4] >= 0.05))[0]
            chosen = chosen[nms(detections[chosen, :4], detections[chosen, 4], 0.7)[:100]]
            detections = detections[chosen]
            boxes = detections[:, :4]
            embeddings, mask_output = self._roi_identity(encoded, [boxes])
            height, width = image.shape[-2:]
            if mask_output is None:
                full_masks = image.new_zeros((0, 1, height, width))
            else:
                output, logits = mask_output
                labels = torch.full((len(boxes),), COCO_CAR, dtype=torch.long, device=image.device)
                arrays = self.detector.mask_head.get_seg_masks(logits, boxes, labels,
                          self.detector.rcnn_test_cfg, metadata[0]['ori_shape'], metadata[0]['scale_factor'], True)
                full_masks = torch.from_numpy(np.stack(arrays)).to(image.device)[:, None].float()
            original_boxes = boxes / boxes.new_tensor(metadata[0]['scale_factor'])
            predictions.append({'boxes': original_boxes, 'scores': detections[:, 4],
                                'masks': full_masks, 'embeddings': embeddings,
                                'labels': torch.ones((len(boxes),), dtype=torch.long, device=image.device)})
        return predictions
