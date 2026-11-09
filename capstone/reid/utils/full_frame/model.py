"""One shared detector, instance-mask head, and mask-pooled vehicle ReID head."""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ShuffleNet_V2_X1_5_Weights, shufflenet_v2_x1_5
from torchvision.models.detection import (MaskRCNN, MaskRCNN_ResNet50_FPN_V2_Weights,
                                          maskrcnn_resnet50_fpn_v2)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


class ShuffleFPN(nn.Module):
    """Expose three ShuffleNet stages to a compact feature pyramid."""

    out_channels = 128

    def __init__(self, pretrained=False):
        super().__init__()
        self.body = shufflenet_v2_x1_5(
            weights=ShuffleNet_V2_X1_5_Weights.DEFAULT if pretrained else None
        )
        self.fpn = FeaturePyramidNetwork(
            [176, 352, 704], self.out_channels, extra_blocks=LastLevelMaxPool()
        )

    def forward(self, image):
        x = self.body.maxpool(self.body.conv1(image))
        low = self.body.stage2(x)
        mid = self.body.stage3(low)
        high = self.body.stage4(mid)
        return self.fpn(OrderedDict((('0', low), ('1', mid), ('2', high))))


def _resnet_detector(pretrained, image_size):
    """Start from COCO-trained Mask R-CNN, retaining its car/background heads."""
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    detector = maskrcnn_resnet50_fpn_v2(
        weights=weights, weights_backbone=None, min_size=image_size,
        max_size=image_size, rpn_pre_nms_top_n_train=500,
        rpn_post_nms_top_n_train=200, rpn_pre_nms_top_n_test=300,
        rpn_post_nms_top_n_test=100, box_detections_per_img=50,
        box_batch_size_per_image=128,
    )
    old_box = detector.roi_heads.box_predictor
    old_mask = detector.roi_heads.mask_predictor
    new_box = FastRCNNPredictor(old_box.cls_score.in_features, 2)
    new_mask = MaskRCNNPredictor(old_mask.conv5_mask.in_channels,
                                 old_mask.conv5_mask.out_channels, 2)
    if pretrained:
        categories = weights.meta['categories']
        if categories[3] != 'car':
            raise RuntimeError('Expected COCO car class at index 3')
        with torch.no_grad():
            new_box.cls_score.weight.copy_(old_box.cls_score.weight[[0, 3]])
            new_box.cls_score.bias.copy_(old_box.cls_score.bias[[0, 3]])
            for new_index, old_index in enumerate((0, 3)):
                new_box.bbox_pred.weight[4*new_index:4*(new_index+1)].copy_(
                    old_box.bbox_pred.weight[4*old_index:4*(old_index+1)])
                new_box.bbox_pred.bias[4*new_index:4*(new_index+1)].copy_(
                    old_box.bbox_pred.bias[4*old_index:4*(old_index+1)])
            new_mask.conv5_mask.load_state_dict(old_mask.conv5_mask.state_dict())
            new_mask.mask_fcn_logits.weight.copy_(old_mask.mask_fcn_logits.weight[[0, 3]])
            new_mask.mask_fcn_logits.bias.copy_(old_mask.mask_fcn_logits.bias[[0, 3]])
    detector.roi_heads.box_predictor = new_box
    detector.roi_heads.mask_predictor = new_mask
    return detector


def batch_hard_triplet(embeddings, identities, margin=0.2):
    """Ignore singleton identities; all identity numbers are sequence-scoped."""
    if len(identities) < 2:
        return embeddings.sum() * 0
    distances = torch.cdist(embeddings.float(), embeddings.float())
    same = identities[:, None] == identities[None, :]
    same.fill_diagonal_(False)
    different = identities[:, None] != identities[None, :]
    valid = same.any(1) & different.any(1)
    if not valid.any():
        return embeddings.sum() * 0
    hardest_positive = distances.masked_fill(~same, -1).max(1).values
    hardest_negative = distances.masked_fill(~different, float('inf')).min(1).values
    return F.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()


class FullFrameVehicleNet(nn.Module):
    """Mask R-CNN heads and a mask-pooled 128D ReID head share one image pass.

    Training targets include boxes, masks, labels=1, track_ids and identity_labels.
    Inference returns boxes, scores, labels, full-image masks and embeddings.
    The tracker/target selector remains outside this per-frame network.
    """

    def __init__(self, num_identities=0, pretrained=False, image_size=640,
                 architecture='resnet50_fpn_v2', crop_image_size=None):
        super().__init__()
        if architecture == 'resnet50_fpn_v2':
            self.detector = _resnet_detector(pretrained, image_size)
        elif architecture == 'shufflenet_fpn':
            backbone = ShuffleFPN(pretrained)
            anchors = AnchorGenerator(
                sizes=((16,), (32,), (64,), (128,)),
                aspect_ratios=((0.5, 1.0, 2.0),) * 4,
            )
            roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2'], output_size=7, sampling_ratio=2)
            self.detector = MaskRCNN(
                backbone, num_classes=None, min_size=image_size, max_size=image_size,
                rpn_anchor_generator=anchors,
                rpn_pre_nms_top_n_train=500, rpn_post_nms_top_n_train=200,
                rpn_pre_nms_top_n_test=300, rpn_post_nms_top_n_test=100,
                box_roi_pool=roi_pool, box_head=TwoMLPHead(128 * 7 * 7, 256),
                box_predictor=FastRCNNPredictor(256, 2),
                box_detections_per_img=50, box_batch_size_per_image=128,
                mask_roi_pool=MultiScaleRoIAlign(featmap_names=['0', '1', '2'], output_size=14, sampling_ratio=2),
                mask_head=MaskRCNNHeads(128, [128, 128], 1),
                mask_predictor=MaskRCNNPredictor(128, 128, 2),
            )
        else:
            raise ValueError(f'Unknown full-frame architecture: {architecture}')
        self.embed_head = nn.Linear(self.detector.backbone.out_channels, 128)
        self.id_classifier = nn.Linear(128, num_identities) if num_identities else None
        self.crop_transform = (GeneralizedRCNNTransform(
            crop_image_size, crop_image_size,
            self.detector.transform.image_mean, self.detector.transform.image_std)
            if crop_image_size is not None else self.detector.transform)

    @staticmethod
    def _pool(features, masks):
        weights = F.interpolate(masks.float(), size=features.shape[-2:], mode='bilinear', align_corners=False)
        weights = weights.clamp(0, 1)
        return (features * weights).sum((2, 3)) / weights.sum((2, 3)).clamp_min(1e-5)

    def _embeddings(self, features, boxes, image_sizes, masks):
        if not any(len(b) for b in boxes):
            return features['0'].new_zeros((0, 128))
        roi_features = self.detector.roi_heads.box_roi_pool(features, boxes, image_sizes)
        return F.normalize(self.embed_head(self._pool(roi_features, masks)), dim=1)

    def _predicted_masks_for_boxes(self, features, boxes, image_sizes):
        """Use the same predicted foreground signal for ReID in train and eval."""
        mask_features = self.detector.roi_heads.mask_roi_pool(features, boxes, image_sizes)
        mask_logits = self.detector.roi_heads.mask_predictor(
            self.detector.roi_heads.mask_head(mask_features)
        )
        return mask_logits.softmax(dim=1)[:, 1:2]

    def crop_embeddings(self, images, global_pool=False, both=False):
        """Embed known vehicle crops with the same backbone, mask, and ReID head.

        Each crop is its own oracle box. No detector proposal is used here;
        full-frame KITTI steps still train the detector and instance masks.
        """
        transformed, _ = self.crop_transform(images, None)
        features = self.detector.backbone(transformed.tensors)
        boxes = [image.new_tensor([[0, 0, width, height]])
                 for image, (height, width) in zip(images, transformed.image_sizes)]
        masks = self._predicted_masks_for_boxes(features, boxes, transformed.image_sizes)
        if both:
            return (self._embeddings(features, boxes, transformed.image_sizes, masks),
                    self._embeddings(features, boxes, transformed.image_sizes,
                                     torch.ones_like(masks)))
        if global_pool:
            masks = torch.ones_like(masks)
        return self._embeddings(features, boxes, transformed.image_sizes, masks)

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError('Training requires KITTI targets')
        original_sizes = [tuple(image.shape[-2:]) for image in images]
        transformed, targets = self.detector.transform(images, targets)
        features = self.detector.backbone(transformed.tensors)
        proposals, rpn_losses = self.detector.rpn(transformed, features, targets)
        detections, roi_losses = self.detector.roi_heads(
            features, proposals, transformed.image_sizes, targets
        )
        if self.training:
            boxes = [target['boxes'] for target in targets]
            losses = {**rpn_losses, **roi_losses}
            if any(len(b) for b in boxes):
                predicted_masks = self._predicted_masks_for_boxes(features, boxes, transformed.image_sizes)
                embeddings = self._embeddings(features, boxes, transformed.image_sizes, predicted_masks)
                ids = torch.cat([target['identity_labels'] for target in targets])
                losses['loss_reid_triplet'] = batch_hard_triplet(embeddings, ids)
                if self.id_classifier is not None:
                    losses['loss_reid_id'] = F.cross_entropy(self.id_classifier(embeddings), ids)
            else:
                losses['loss_reid_triplet'] = features['0'].sum() * 0
            return losses

        boxes = [det['boxes'] for det in detections]
        masks = [det['masks'] for det in detections]
        embeddings = self._embeddings(features, boxes, transformed.image_sizes,
                                      torch.cat(masks) if any(len(b) for b in boxes) else None)
        offset = 0
        for detection in detections:
            count = len(detection['boxes'])
            detection['embeddings'] = embeddings[offset:offset + count]
            offset += count
        return self.detector.transform.postprocess(detections, transformed.image_sizes, original_sizes)
