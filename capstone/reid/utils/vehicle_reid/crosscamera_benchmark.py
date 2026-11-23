"""Measured cross-camera comparisons on identical partitions and rejection protocol."""
import csv
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml
from utils.tracking.common import ROOT, load, save
from utils.teacher.assets import sha256
from utils.reid_teacher.data import read, partitions
from utils.reid_teacher.evaluation import candidates, load_candidate
from utils.core.metrics import cross_view_metrics
from .operating_points import scores, statistics, calibrate


def run(config):
    settings = read()
    data = partitions(settings)
    identities = len({row['id'] for row in data['train']})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifests = {name: sha256(ROOT / value) for name, value in settings['data'].items()}
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for title, kind, path in candidates(settings, identities, include_finetuned=True):
        signature = dict(kind=kind, checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=sha256(path), manifests=manifests)
        destination = ROOT / 'data/vehicle_benchmark/retrieval' / kind / 'state.pt'
        banks = {stage: [[], []] for stage in ('calibration', 'final')}
        seconds = 0.
        extraction_hardware = []
        if destination.exists():
            previous = load(destination)
            if previous['signature'] != signature:
                raise ValueError(f'Cross-camera inputs changed for {kind}; choose a new checkpoint location')
            banks, seconds = previous['banks'], previous['seconds']
            extraction_hardware = previous.get('extraction_hardware', ['unrecorded legacy checkpoint'])
        network = None
        for stage, names in [('calibration', ['calibration_query', 'calibration_gallery']), ('final', ['query', 'gallery'])]:
            for group, name in enumerate(names):
                records = data[name]
                offset = sum(len(chunk) for chunk in banks[stage][group])
                if offset == len(records):
                    continue
                if network is None:
                    network, transform, pair = load_candidate(kind, path, settings, identities)
                    network.to(device).eval().requires_grad_(False)
                    hardware = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
                    if hardware not in extraction_hardware:
                        extraction_hardware.append(hardware)
                batch = settings['evaluation']['batch_size']
                start_time = time.perf_counter()
                for start in range(offset, len(records), batch):
                    images = []
                    for row in records[start:start+batch]:
                        with Image.open(row['path']) as image:
                            images.append(transform(image.convert('RGB')))
                    with torch.inference_mode():
                        result = network(torch.stack(images).to(device))
                        vectors = F.normalize((result[1] if pair else result).float(), dim=1).cpu()
                    banks[stage][group].append(vectors)
                    if (start//batch+1) % 10 == 0 or start+batch >= len(records):
                        seconds += time.perf_counter()-start_time
                        save(dict(signature=signature, banks=banks, seconds=seconds, extraction_hardware=extraction_hardware), destination)
                        print(f'{kind} {stage}/{name}: {min(start+batch, len(records))}/{len(records)} crops committed ({device})', flush=True)
                        start_time = time.perf_counter()
        analyzed = {}
        for stage, names in [('calibration', ['calibration_query', 'calibration_gallery']), ('final', ['query', 'gallery'])]:
            query, gallery = [torch.cat(chunks) for chunks in banks[stage]]
            query_ids, gallery_ids = [[row['id'] for row in data[name]] for name in names]
            analyzed[stage] = scores(query, gallery, query_ids, gallery_ids)
            if stage == 'final':
                correct, count, mean_ap = cross_view_metrics(query, query_ids, [row['camera'] for row in data[names[0]]],
                    gallery, gallery_ids, [row['camera'] for row in data[names[1]]])
        points = []
        for budget in [.01, .05]:
            calibration = calibrate(analyzed['calibration'], budget)
            final = statistics(analyzed['final'], calibration['threshold'], calibration['margin'], confidence=True)
            points.append(dict(empirical_calibration_budget=budget, calibration=calibration, final=final))
        results.append(dict(model=title, signature=signature, rank1=correct/count, map=mean_ap,
            queries=count, gallery=len(data['gallery']), operating_points=points,
            extraction_seconds=seconds, extraction_hardware=extraction_hardware, analysis_device=str(device),
            timing_scope='crop specialist extraction only; no detector, tracking, or video output'))
        print(f'{title}: final Rank-1 {correct/count:.2%}, mAP {mean_ap:.2%}', flush=True)
        del network
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    report = dict(protocol='VRAI local identity-disjoint train/calibration/final split, not official test',
        open_set='one first-manifest gallery reference per known identity; every fifth sorted ID withheld as unknown',
        calibration='threshold/margin maximize correct known acceptance under empirical unknown-FAR and wrong-known budgets; final identities never tune thresholds',
        models=results)
    (output / 'cross_camera.yaml').write_text(yaml.safe_dump(report))
    rows = []
    for result in results:
        for point in result['operating_points']:
            final = point['final']
            rows.append(dict(model=result['model'], rank1=100*result['rank1'], map=100*result['map'],
                calibration_budget=point['empirical_calibration_budget'], **final))
    with (output / 'cross_camera.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return report


if __name__ == '__main__':
    run(yaml.safe_load((ROOT / 'configs/vehicle_benchmark.yaml').read_text()))
