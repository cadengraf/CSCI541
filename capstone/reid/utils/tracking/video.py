"""One selected predicted vehicle: current tracker versus CUTIE, as an MP4."""

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from utils.full_frame.tracker import VehicleTracker, disjoint_predictions
from utils.teacher.settings import read as teacher_settings
from .common import ROOT, read, load, decode
from .propagation import cutie_processor


def choose(prediction, image, settings):
    if not len(prediction['boxes']):
        raise ValueError('No vehicle detected in the selected start frame; change start_frame')
    point = settings['click_xy']
    if settings['selection'] == 'click' and point is None:
        if not __import__('os').environ.get('DISPLAY'):
            raise RuntimeError('No desktop display; set selected.click_xy: [x, y] for SSH')
        points = []
        preview = image.copy()
        for index, box in enumerate(prediction['boxes'].int().tolist()):
            cv2.rectangle(preview, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
            cv2.putText(preview, str(index), tuple(box[:2]), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
        cv2.namedWindow('Click a vehicle, then press Enter')
        cv2.setMouseCallback('Click a vehicle, then press Enter',
                             lambda event, x, y, flags, param: points.append((x, y)) if event == cv2.EVENT_LBUTTONDOWN else None)
        cv2.imshow('Click a vehicle, then press Enter', preview)
        cv2.waitKey(0); cv2.destroyAllWindows()
        if not points:
            raise ValueError('No vehicle selected')
        point = points[-1]
    if point is not None:
        x, y = map(int, point)
        height, width = image.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError('click_xy is outside the source image')
        for index, mask in enumerate(prediction['masks']):
            if mask[y, x]:
                return index
        raise ValueError('Selected point is outside all predicted vehicle masks')
    if settings['selection'] != 'largest':
        raise ValueError('selected.selection must be click or largest')
    return int(prediction['masks'].flatten(1).sum(1).argmax())


def panel(image, mask, title, status):
    result = image.copy()
    if mask is not None and mask.any():
        foreground = mask.astype(bool)
        result[foreground] = (.55 * result[foreground] + .45 * np.array([0, 255, 0])).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    cv2.rectangle(result, (0, 0), (result.shape[1], 58), (20, 20, 20), -1)
    cv2.putText(result, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
    cv2.putText(result, status, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, .5, (220, 220, 220), 1)
    return result


@torch.inference_mode()
def run(check=False):
    settings = read()
    teacher, _ = teacher_settings()
    spec = settings['selected']
    folder = ROOT / teacher['data']['kitti_root'] / 'training/image_02' / str(spec['sequence']).zfill(4)
    frames = sorted(folder.glob('*.png'))[spec['start_frame']:spec['start_frame'] + (6 if check else spec['frames'])]
    if not frames:
        raise ValueError('No frames in selected interval')
    records = [ROOT / settings['cache'] / folder.name / f'{path.stem}.pt' for path in frames]
    if not all(path.is_file() for path in records):
        raise FileNotFoundError('Run `python demo.py cutie` to cache the selected interval first')
    processor = cutie_processor(settings)
    tracker = VehicleTracker(teacher['tracking'])
    output = ROOT / spec['output']
    if check:
        output = output.with_name(output.stem + '_check.mp4')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + '.partial.mp4')
    writer, target = None, None
    try:
        for index, (path, record) in enumerate(zip(frames, records)):
            image = cv2.imread(str(path))
            prediction = disjoint_predictions(decode(load(record)), teacher['tracking'])
            identities = tracker.update(prediction, index)
            if index == 0:
                selected = choose(prediction, image, spec)
                target = identities[selected]
                seed = prediction['masks'][selected].long().cuda()
                print(f'Selected predicted vehicle {selected}; initial tracker ID {target}', flush=True)
            tensor = to_tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).cuda()
            with torch.autocast('cuda', dtype=torch.float16):
                probabilities = processor.step(tensor, seed, objects=[1]) if index == 0 else processor.step(tensor)
            cutie_mask = processor.output_prob_to_mask(probabilities).cpu().numpy() == 1
            current = prediction['masks'][identities.index(target)].numpy() if target in identities else None
            left = panel(image, current, 'Teacher + current spatial / ReID tracker',
                         f'frame {path.stem} | ID {target} ' + ('visible' if current is not None else 'not associated'))
            right = panel(image, cutie_mask, 'CUTIE | same predicted mask initialization',
                          'Temporal propagation only; no cross-camera ReID or automatic reacquisition')
            combined = np.concatenate((left, right), axis=1)
            scale = min(1, 1920 / combined.shape[1])
            width, height = int(combined.shape[1]*scale)//2*2, int(combined.shape[0]*scale)//2*2
            combined = cv2.resize(combined, (width, height))
            if writer is None:
                writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'mp4v'), spec['fps'], (width, height))
                if not writer.isOpened():
                    raise RuntimeError('OpenCV could not create an MP4 writer')
            writer.write(combined)
            if index % 50 == 0:
                print(f'Selected-object video: {index+1}/{len(frames)} frames', flush=True)
    finally:
        if writer is not None:
            writer.release()
    temporary.replace(output)
    print(f'Video: {output}', flush=True)
