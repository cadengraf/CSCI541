"""Normalize detector outputs; compute all identity features with the frozen specialist."""
import cv2
import numpy as np
from PIL import Image
import torch


def rfdetr_predictions(result, image_shape, car_id, threshold):
    keep=np.flatnonzero((np.asarray(result.class_id)==car_id) & (np.asarray(result.confidence)>=threshold)) if len(result.xyxy) else np.array([],dtype=int)
    if result.mask is None and len(keep):raise ValueError('RF-DETR returned boxes without instance masks')
    h,w=image_shape[:2]
    if not len(keep):
        return dict(boxes=torch.empty((0,4)),scores=torch.empty(0),masks=torch.empty((0,1,h,w),dtype=torch.bool))
    masks=np.asarray(result.mask)[keep] if len(keep) else np.empty((0,h,w),dtype=bool)
    if masks.shape!=(len(keep),h,w):raise ValueError('RF-DETR masks are not aligned to the original frame')
    return dict(boxes=torch.as_tensor(np.asarray(result.xyxy)[keep].copy(),dtype=torch.float32),
        scores=torch.as_tensor(np.asarray(result.confidence)[keep].copy(),dtype=torch.float32),
        masks=torch.as_tensor(masks.copy(),dtype=torch.bool)[:,None])


class Detector:
    def __init__(self, name, config, root, device):
        self.name,self.config,self.device=name,config,device
        if name=='codino':
            from utils.teacher.model import CoDINOJoint
            saved=torch.load(root/config['codino']['checkpoint'],map_location='cpu',weights_only=False)
            self.model=CoDINOJoint(saved['settings'],root,saved['model']['reid.classifier'].shape[0],pretrained=False)
            self.model.load_state_dict(saved['model']);self.model.to(device).eval().requires_grad_(False)
        else:
            import hashlib
            from importlib.metadata import version
            import rfdetr
            spec=config['rfdetr']
            if version('rfdetr')!=spec['package_version']:raise ValueError('RF-DETR package version changed')
            checkpoint=root/spec['checkpoint']
            if spec.get('checkpoint_format')=='finetuned':
                from utils.teacher.assets import sha256
                if sha256(checkpoint)!=spec['sha256']:raise ValueError('Selected fine-tuned RF-DETR checkpoint changed')
                self.model=rfdetr.RFDETR.from_checkpoint(str(checkpoint),device=str(device),resolution=spec['resolution'])
                if self.model.model_config.num_classes!=1 or spec['coco_car_id']!=0:
                    raise ValueError('Fine-tuned benchmark expects the recorded one-class car=0 mapping')
            else:
                if hashlib.md5(checkpoint.read_bytes()).hexdigest()!=spec['md5']:
                    raise ValueError('RF-DETR checkpoint differs from the pinned upstream asset')
                self.model=getattr(rfdetr,spec['model_class'])(pretrain_weights=str(checkpoint),
                    device=str(device),resolution=spec['resolution'])

    @torch.inference_mode()
    def __call__(self, image):
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        if self.name=='codino':
            x=torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1).float().to(self.device)/255
            pred=self.model([x])[0]
            return {key:pred[key].detach().cpu() for key in ['boxes','scores','masks']}
        result=self.model.predict(Image.fromarray(rgb),threshold=self.config['score_threshold'])
        return rfdetr_predictions(result,image.shape,self.config['rfdetr']['coco_car_id'],self.config['score_threshold'])


def identity_predictions(image, raw, encoder, config):
    """Same mask disjointing, boxes and specialist preprocessing for both detectors."""
    from utils.full_frame.tracker import disjoint_predictions
    boxes=raw['boxes'].clone().float()
    h,w=image.shape[:2]
    boxes[:,[0,2]]=boxes[:,[0,2]].clamp(0,w);boxes[:,[1,3]]=boxes[:,[1,3]].clamp(0,h)
    valid=torch.isfinite(boxes).all(1)&((boxes[:,2:]-boxes[:,:2]).min(1).values>=2)
    raw={**raw,'boxes':boxes}
    raw={key:value[valid] for key,value in raw.items()}
    # The existing mask adapter carries an embedding field; filter masks before
    # paying for crop encoding, then replace this placeholder with specialist features.
    raw['embeddings']=torch.zeros((len(raw['boxes']),1))
    settings=dict(score_threshold=config['score_threshold'],mask_threshold=config['mask_threshold'],
                  min_mask_pixels=config['minimum_mask_pixels'])
    pred=disjoint_predictions(raw,settings)
    pred['embeddings']=torch.from_numpy(encoder(image,pred['boxes'].numpy())).float()
    return pred
