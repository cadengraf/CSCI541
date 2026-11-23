"""Segmented MP4 output with atomic identity checkpoints and exact frame resume."""
import argparse
import csv
import hashlib
from pathlib import Path
import time
import subprocess
import platform
import sys
import cv2
import numpy as np
import torch
import yaml
from utils.tracking.common import ROOT, save
from .tracklets import build as build_association
from .evaluation import Diagnostics
from .models import Detector, Embeddings, observations


class Source:
    def __init__(self, path):
        self.path = Path(path)
        self.files = None
        self.capture = None
        if self.path.is_dir():
            self.files = sorted(path for path in self.path.iterdir() if path.suffix.lower() in ('.jpg', '.png', '.jpeg'))
            if not self.files:
                raise ValueError('Source directory contains no frames')
            if any(not path.stem.isdigit() for path in self.files):
                raise ValueError('Frame filenames must be numeric and zero padded')
            self.files.sort(key=lambda path: int(path.stem))
            if any(int(b.stem) != int(a.stem)+1 for a, b in zip(self.files, self.files[1:])):
                raise ValueError('Frame directory must be contiguous')
            metadata = [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in self.files]
            self.count = len(self.files)
        else:
            info = self.path.stat()
            metadata = [(str(self.path), info.st_size, info.st_mtime_ns)]
            self.capture = cv2.VideoCapture(str(self.path))
            if not self.capture.isOpened():
                raise ValueError(f'Cannot open video: {self.path}')
            self.count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.signature = hashlib.sha256(repr(metadata).encode()).hexdigest()

    def seek(self, frame):
        if self.capture is not None:
            # Sequential decode gives reliable resume even with inter-frame codecs.
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(frame):
                if not self.capture.read()[0]:
                    raise ValueError('Resume position exceeds video')

    def read(self, frame):
        if self.files is not None:
            return cv2.imread(str(self.files[frame])), int(self.files[frame].stem)
        ok, image = self.capture.read()
        return (image if ok else None), frame

    def close(self):
        if self.capture is not None:
            self.capture.release()


def select_target(selection, masks, boxes, reliable):
    if selection is None:
        return None
    if selection == 'largest':
        candidates = [index for index, good in enumerate(reliable) if good]
        if not candidates:
            raise ValueError('No reliable target on first frame; adjust start_frame or select a later frame')
        return max(candidates, key=lambda index: masks[index].sum())
    x, y = map(int, selection)
    hits = [index for index, mask in enumerate(masks) if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]]
    if not hits:
        raise ValueError('Selection does not intersect a predicted mask')
    chosen = min(hits, key=lambda index: masks[index].sum())
    if not reliable[chosen]:
        raise ValueError('Selected observation is unreliable; use another start_frame')
    return chosen


def render(image, masks, boxes, decisions, labels, target, frame, elapsed):
    image = image.copy()
    for mask, box, decision, label in zip(masks, boxes, decisions, labels):
        identity = decision['identity']
        color = (80, 220, 80) if identity == target and target is not None else (220, 170, 60)
        if label == 'wrong_identity':
            color = (60, 60, 240)
        if identity is None:
            color = (120, 160, 220)
        image[mask] = (.7*image[mask] + .3*np.array(color)).astype(np.uint8)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        display = identity if identity is not None else f'L{decision["tracklet_id"]}' if "tracklet_id" in decision else "?"
        text = f'{display} {decision["status"]} {decision["similarity"]:.2f}'
        cv2.putText(image, text, (x1, max(18, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    visible = target is not None and any(decision['identity'] == target for decision in decisions)
    banner = f'frame {frame} | target {target}: {"visible" if visible else "no confident match"} | compute {elapsed:.0f} ms'
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(image, banner, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    return image


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def finalize_video(output, parts):
    """Remux committed segments into the primary visual output without re-encoding."""
    playlist = output / 'segments.partial.txt'
    destination = output / 'annotated.partial.mp4'
    playlist.write_text(''.join(f"file 'part{part:06d}.mp4'\n" for part in range(parts)))
    try:
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0',
                        '-i', str(playlist), '-c', 'copy', '-movflags', '+faststart', str(destination)], check=True)
        destination.replace(output / 'annotated.mp4')
    finally:
        playlist.unlink(missing_ok=True)


def ensure_codino_runtime(backend, arguments):
    """Keep one public command while using the already installed MMCV environment."""
    if backend != 'codino':
        return False
    from utils.teacher.settings import read, resolve
    settings, root = read()
    environment = resolve(root, settings['environment'])
    if Path(sys.prefix).resolve() == environment.resolve():
        return False
    python = environment / 'bin/python'
    if not python.is_file():
        raise FileNotFoundError('Co-DINO runtime is missing; run python train.py teacher-setup')
    print('Starting Co-DINO in its existing runtime', flush=True)
    subprocess.run([str(python), str(ROOT / 'demo.py'), 'vehicle', *arguments], cwd=ROOT, check=True)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/vehicle_reid.yaml')
    parser.add_argument('--source')
    parser.add_argument('--output')
    parser.add_argument('--frames', type=int)
    parser.add_argument('--backend', choices=['codino', 'cache'])
    parser.add_argument('--device', choices=['auto'], default='auto')
    parser.add_argument('--truth', help='KITTI MOTS annotations used only for diagnostics')
    parser.add_argument('--target-truth-id', type=int, help='Evaluation label for the selected target')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', action='store_true')
    group.add_argument('--profile', help='Prior state.pt: reuse target appearance in a new video/view; discard spatial state')
    arguments = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(arguments)
    config = yaml.safe_load((ROOT / args.config).read_text())
    for key in ('source', 'output', 'frames', 'device'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.truth is not None:
        config['evaluation']['truth'] = args.truth
    if args.target_truth_id is not None:
        config['evaluation']['target_truth_id'] = args.target_truth_id
    if args.backend:
        if args.backend != config['detector']['backend'] and args.backend != 'cache':
            config['detector']['checkpoint'] = 'checkpoints/teacher/co_dino_joint.best.pt'
        config['detector']['backend'] = args.backend
    if args.profile and config['association'].get('engine') == 'reference':
        # Importing a target profile explicitly requests global appearance search.
        config['association']['engine'] = 'tracklets'
        if args.output is None:
            config['output'] += '_profile'
        print('Target profile search uses the experimental global resolver',flush=True)
    if config['frames'] <= 0 or config['checkpoint_every'] <= 0 or config['fps'] <= 0:
        raise ValueError('frames, checkpoint_every, and fps must be positive')
    if config['detector']['backend'] not in ('codino', 'cache'):
        raise ValueError('detector.backend must be codino or cache')
    if ensure_codino_runtime(config['detector']['backend'], arguments):
        return
    torch.set_num_threads(config['cpu_threads'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'state.pt'
    if destination.exists() and not args.resume:
        raise ValueError('Output already has state.pt; use --resume or a new output folder')
    started = time.perf_counter()
    detector, encoder = Detector(config['detector'], device), Embeddings(config['reid'], device)
    model_signature = dict(detector=detector.signature, reid=encoder.signature)
    source = Source(ROOT / config['source'])
    if config['detector']['backend'] == 'cache' and source.files is None:
        raise ValueError('Cache replay requires its original numbered image directory')
    tracker, diagnostics = build_association(config['association']), Diagnostics(config['evaluation'])
    target, begin, part = None, config['start_frame'], 0
    if args.resume:
        saved = torch.load(destination, map_location='cpu', weights_only=False)
        old, current = dict(saved['config']), dict(config)
        old.pop('frames'); current.pop('frames')
        if old != current or saved['models'] != model_signature or saved['source'] != source.signature:
            raise ValueError('Resume config, models, or source changed')
        tracker, diagnostics, target = saved['tracker'], saved['diagnostics'], saved['target']
        begin, part = saved['next_frame'], saved['part']
    elif args.profile:
        saved = torch.load(ROOT / args.profile, map_location='cpu', weights_only=False)
        if saved['models']['reid'] != model_signature['reid']:
            raise ValueError('Target profile uses different embeddings')
        target = saved['target']
        if target is None or not saved['tracker'].identities[target].confirmed:
            raise ValueError('Profile requires a selected, confirmed target')
        tracker.identities = {target: saved['tracker'].identities[target]}
        tracker.next_id = max(saved['tracker'].next_id, target+1)
        tracker.reset_view()
    if begin < 0 or begin >= source.count:
        if args.resume and begin == min(source.count, config['start_frame']+config['frames']):
            finalize_video(output, part)
            source.close()
            print(f'Requested interval already complete: {output / "annotated.mp4"}', flush=True)
            return
        raise ValueError('start/resume frame outside source')
    end = min(source.count, config['start_frame']+config['frames'])
    if begin >= end:
        finalize_video(output, part)
        source.close()
        print('Requested interval already complete; use a larger --frames value to extend it', flush=True)
        return
    source.seek(begin)
    initialization_seconds = time.perf_counter()-started
    print(f'{device}: {begin}:{end}; detector={config["detector"]["backend"]}; init {initialization_seconds:.1f}s', flush=True)
    fields = ['frame', 'objects', 'unknown', 'detector_ms', 'reid_ms', 'association_ms', 'evaluation_ms',
              'render_encode_ms', 'total_ms', 'historical_detector_ms']
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for segment_start in range(begin, end, config['checkpoint_every']):
            temporary = output / f'part{part:06d}.partial.mp4'
            video = output / f'part{part:06d}.mp4'
            writer, rows = None, []
            segment_end = min(end, segment_start+config['checkpoint_every'])
            try:
                for frame in range(segment_start, segment_end):
                    sync(device); t0 = time.perf_counter()
                    image, actual_frame = source.read(frame)
                    if image is None:
                        raise ValueError(f'Cannot decode source frame {frame}')
                    boxes, scores, masks, historical_ms = detector(image, actual_frame, source.path)
                    boxes, scores, masks, reliable = observations(image, boxes, scores, masks, config['quality'])
                    sync(device); t1 = time.perf_counter()
                    features = encoder(image, boxes)
                    sync(device); t2 = time.perf_counter()
                    decisions = tracker.update(boxes, features, reliable, frame)
                    if frame == begin and not args.resume and not args.profile:
                        chosen = select_target(config['select'], masks, boxes, reliable)
                        if chosen is not None:
                            target = decisions[chosen]['identity']
                            if target is None:
                                raise ValueError('Selected target could not be initialized')
                            # Explicit selection enrolls this reliable observation as
                            # a persistent reference, even if it disappears immediately.
                            tracker.identities[target].confirmed = True
                    t3 = time.perf_counter()
                    labels = diagnostics.update(actual_frame, masks, decisions, target)
                    t4 = time.perf_counter()
                    annotated = render(image, masks, boxes, decisions, labels, target, actual_frame, (t3-t0)*1000)
                    height, width = annotated.shape[:2]
                    annotated = annotated[:height//2*2, :width//2*2]
                    if writer is None:
                        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'mp4v'), config['fps'],
                                                 (annotated.shape[1], annotated.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError('MP4 writer could not open')
                    writer.write(annotated)
                    t5 = time.perf_counter()
                    rows.append(dict(zip(fields, [actual_frame, len(boxes), sum(d['identity'] is None for d in decisions),
                        (t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000, (t4-t3)*1000, (t5-t4)*1000, (t5-t0)*1000, historical_ms])))
            finally:
                if writer is not None:
                    writer.release()
            temporary.replace(video)
            with (output / f'part{part:06d}.csv').open('w', newline='') as file:
                report = csv.DictWriter(file, fieldnames=fields)
                report.writeheader(); report.writerows(rows)
            save(dict(format=1, config=config, models=model_signature, source=source.signature,
                      tracker=tracker, diagnostics=diagnostics, target=target, next_frame=segment_end, part=part+1), destination)
            # Includes video finalization, reports, and checkpoint I/O. Model loading
            # and resume seek are reported separately, outside steady-state throughput.
            wall_seconds = time.perf_counter()-started-initialization_seconds
            summary = dict(frames_this_invocation=segment_end-begin, target=target, next_frame=segment_end,
                camera=config['camera'], backend=config['detector']['backend'], device=str(device),
                hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor() or platform.machine(),
                cpu_threads=torch.get_num_threads(),
                initialization_seconds=initialization_seconds, wall_seconds=wall_seconds,
                wall_fps=(segment_end-begin)/wall_seconds, counts=dict(diagnostics.counts),
                ground_truth=config['evaluation']['truth'], models=model_signature,
                peak_cuda_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else None,
                timing_scope='REPLAY: detector is offline; historical detector timings are separate' if detector.model is None
                             else 'ONLINE: decode, detector, specialist, association, evaluation, rendering, encoding, checkpoint I/O')
            (output / 'summary.yaml').write_text(yaml.safe_dump(summary))
            print(f'Saved {video.name}: frames through {segment_end-1}, target {target}; {summary["wall_fps"]:.2f} FPS', flush=True)
            part += 1
    finally:
        source.close()
    if part:
        finalize_start = time.perf_counter()
        finalize_video(output, part)
        summary['finalize_seconds'] = time.perf_counter()-finalize_start
        summary['wall_seconds'] += summary['finalize_seconds']
        summary['wall_fps'] = summary['frames_this_invocation']/summary['wall_seconds']
        (output / 'summary.yaml').write_text(yaml.safe_dump(summary))
    print(f'Video: {output / "annotated.mp4"}; resumable identity profile: {destination}', flush=True)
