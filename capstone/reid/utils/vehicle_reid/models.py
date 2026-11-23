"""Adapters return full-resolution masks and independent specialist embeddings."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from utils.tracking.common import ROOT
from utils.teacher.assets import sha256


class Detector:
    def __init__(self, spec, device):
        if spec['backend'] not in ('codino', 'cache'):
            raise ValueError('detector.backend must be codino or cache')
        self.spec, self.device = spec, device
        self.model = None
        if spec['backend'] == 'cache':
            import yaml
            self.manifest = yaml.safe_load((ROOT / spec['cache'] / 'source.yaml').read_text())
            self.signature = {'backend': 'cache', 'source': self.manifest}
            return
        if device.type != 'cuda':
            raise RuntimeError('Live Co-DINO requires CUDA; cache is available for offline replay')
        checkpoint = ROOT / spec['checkpoint']
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if spec['backend'] == 'codino':
            from utils.teacher.settings import read
            from utils.teacher.model import CoDINOJoint
            settings, root = read()
            self.model = CoDINOJoint(settings, root, saved['model']['reid.classifier'].shape[0], pretrained=False)
            self.model.load_state_dict(saved['model'])
        self.model.to(device).eval().requires_grad_(False)
        self.signature = {'backend': spec['backend'], 'checkpoint_sha256': sha256(checkpoint)}

    @torch.inference_mode()
    def __call__(self, image, frame, source):
        if self.model is None:
            from utils.tracking.common import load, decode
            path = ROOT / self.spec['cache'] / Path(source).name / f'{frame:06d}.pt'
            record = load(path)
            if tuple(record['size']) != image.shape[:2]:
                raise ValueError(f'Cache/image dimensions differ: {path}')
            prediction = decode(record)
            # The caller reports historical detector cost separately.
            historical_ms = record['model_ms']
        else:
            rgb = np.ascontiguousarray(image[:, :, ::-1])
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(self.device) / 255
            prediction = self.model([tensor])[0]
            historical_ms = None
        keep = prediction['scores'].detach().cpu() >= self.spec['score_threshold']
        boxes = prediction['boxes'].detach().cpu()[keep].numpy().astype(np.float32)
        scores = prediction['scores'].detach().cpu()[keep].numpy()
        masks = prediction['masks'].detach().cpu()[keep].numpy()
        if masks.ndim == 4:
            masks = masks[:, 0]
        return boxes, scores, masks >= .5, historical_ms


class Embeddings:
    def __init__(self, spec, device):
        import yaml
        manifest = yaml.safe_load((ROOT / spec['selection']).read_text())
        if manifest['candidate']['kind'] == 'transreid_appearance':
            from .appearance_encoder import load_encoder
            self.model, self.transform, selection = load_encoder(device, ROOT / spec['selection'])
        else:
            from utils.reid_teacher.evaluation import load_selected
            self.model, self.transform, selection = load_selected(device, ROOT / spec['selection'])
        self.device, self.batch_size = device, spec['batch_size']
        self.signature = selection['candidate']

    @torch.inference_mode()
    def __call__(self, image, boxes):
        rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        outputs = []
        for start in range(0, len(boxes), self.batch_size):
            crops = [self.transform(rgb.crop(tuple(map(int, box)))) for box in boxes[start:start+self.batch_size]]
            outputs.append(self.model(torch.stack(crops).to(self.device)).cpu().numpy())
        return np.concatenate(outputs) if outputs else np.zeros((0, 3840), dtype=np.float32)


def observations(image, boxes, scores, masks, spec):
    """Reject malformed crops; flag likely truncation/occlusion for memory gating."""
    height, width = image.shape[:2]
    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    boxes = np.round(boxes).astype(np.float32)
    keep = np.isfinite(boxes).all(1) & ((boxes[:, 2:]-boxes[:, :2]).min(1) >= 2)
    boxes, scores, masks = boxes[keep], scores[keep], masks[keep]
    reliable = []
    for index, (box, score, mask) in enumerate(zip(boxes, scores, masks)):
        x1, y1, x2, y2 = map(int, box)
        area = (x2-x1)*(y2-y1)
        coverage = mask[y1:y2, x1:x2].sum() / area
        overlap = 0.
        for other, other_box in enumerate(boxes):
            if other == index:
                continue
            intersection = np.maximum(0, np.minimum(box[2:], other_box[2:])-np.maximum(box[:2], other_box[:2])).prod()
            overlap = max(overlap, float(intersection/area))
        reliable.append(bool(score >= spec['min_score'] and min(x2-x1, y2-y1) >= spec['min_side'] and
            min(x1, y1, width-x2, height-y2) >= spec['border'] and coverage >= spec['min_mask_fraction'] and
            overlap <= spec['max_box_overlap']))
    return boxes, scores, masks, reliable
