"""Calibration and tracking comparisons with schedule and budget checks."""
import csv
from html import escape
from .settings import ARMS, read


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def render(settings_reader=read, arms=ARMS, feature_comparison=False, crop_comparison=False):
    settings, root = settings_reader(arms[0])
    cfg = settings['matched_reid']; output = root/cfg['output']
    histories = {arm: rows(output/arm/'training.csv') for arm in arms}
    for arm, history in histories.items():
        if not any(r['event'] == 'calibration' and int(r['epoch']) == cfg['epochs'] for r in history):
            raise ValueError(f'{arm} is incomplete; finish both arms before comparing')
    schedules = [{(r['epoch'], r['step']): r['batch_key'] for r in history if r['event'] == 'train'}
                 for history in histories.values()]
    if schedules[0] != schedules[1]:
        raise ValueError('Logged batch schedules differ; comparison is not matched')
    summaries = []
    for arm, history in histories.items():
        calibration = [r for r in history if r['event'] == 'calibration']
        # Last record for each epoch handles logs replayed after a lost save.
        calibration = list({int(r['epoch']): r for r in calibration}.values())
        calibration.sort(key=lambda r: int(r['epoch']))
        best = max(calibration, key=lambda r: float(r['calibration_map']))
        initial, final = calibration[0], calibration[-1]
        record = {'arm': arm, 'selected_epoch': int(best['epoch']),
                  'calibration_Rank1': 100*float(best['calibration_rank1']),
                  'calibration_mAP': 100*float(best['calibration_map']),
                  'final_epoch_Rank1': 100*float(final['calibration_rank1']),
                  'final_epoch_mAP': 100*float(final['calibration_map']),
                  'final_mAP_change_pp': 100*(float(final['calibration_map'])-float(initial['calibration_map'])),
                  'scheduled_updates': 2*cfg['epochs']*cfg['steps_per_epoch'],
                  'skipped_updates': int(calibration[-1]['skipped_updates'])}
        retrieval = output/arm/'calibration.csv'
        if retrieval.exists():
            for result in rows(retrieval):
                record[result['pooling']+'_mAP'] = float(result['mAP'])
        tracking = output/arm/'kitti_mots/summary.csv'
        if tracking.exists():
            result = rows(tracking)[0]
            for key in ('HOTA', 'IDF1', 'sMOTSA', 'IDSW'):
                record[key] = float(result[key])
        if crop_comparison:
            policy = settings_reader(arm)[0]['model']
            record['crop_geometry'] = policy['crop_geometry']
            record['crop_size'] = policy['crop_size']
        if feature_comparison:
            record['added_adapter_parameters'] = 1180160 if settings_reader(arm)[0]['model'].get('reid_spatial_adapter') else 0
        summaries.append(record)
    fields = list(dict.fromkeys(k for row in summaries for k in row))
    with (output/'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(summaries)
    def table(records):
        keys = list(dict.fromkeys(k for row in records for k in row))
        return '<table><tr>'+''.join(f'<th>{escape(k)}</th>' for k in keys)+'</tr>'+''.join(
            '<tr>'+''.join(f'<td>{row.get(k, ""):.2f}</td>' if isinstance(row.get(k), float)
                          else f'<td>{escape(str(row.get(k, "")))}</td>' for k in keys)+'</tr>' for row in records)+'</table>'
    skipped = any(row['skipped_updates'] for row in summaries)
    note = ('AMP skipped updates occurred; successful update budgets may differ. Inspect counts before drawing conclusions.'
            if skipped else 'Both arms completed the same number of successful scheduled optimizer updates.')
    fallback = [row['arm'] for row in summaries if row['selected_epoch'] == 0]
    if fallback:
        note += ' Epoch-zero fallback selected for: ' + ', '.join(fallback) + '. Selected retrieval and tracking scores for these arms measure the original starting checkpoint; inspect final-epoch scores and history to assess training effects.'
    curves = [{'arm': arm, 'epoch': r['epoch'], 'Rank1': 100*float(r['calibration_rank1']),
               'mAP': 100*float(r['calibration_map'])}
              for arm, history in histories.items() for r in history if r['event'] == 'calibration']
    title = 'Raw encoder ROI versus dedicated ReID adapter' if feature_comparison else 'ReID-only versus joint continuation'
    description = ('Both use raw shared encoder ROIs. The adapter arm adds identity-initialized spatial processing '
        'before the same predicted-mask pooling and projection. Both retain joint KITTI auxiliary losses; no new distillation. '
        'Only this adapter differs: initialization of common weights, input schedule and optimizer update budget match.'
        if feature_comparison else
        'Identical origin, architecture, KITTI/VRAI identity data, batches, augmentation seeds, trainable layers and scheduled updates. '
        'Only joint adds detection/mask auxiliary losses. No new teacher distillation in either arm.')
    caveat = ('The adapter adds trainable capacity and compute. This is not a comparison of pre-mask versus post-mask features: '
        'the original mask_feats already contains raw encoder ROI features. One seed is preliminary evidence.'
        if feature_comparison else
        'ReID-only retains the architecture and predicted masks; its mask branch is updated through identity gradients, '
        'without direct mask supervision. It is not a separate specialist architecture. Equal input/update budgets do not imply '
        'equal wall time or FLOPs: both execute the KITTI forward, but backpropagate different objectives. One seed is preliminary evidence.')
    extra = ''
    if crop_comparison:
        title = 'Matched identity-crop training: aspect 384 versus square 256'
        description = ('Same starting weights, joint KITTI losses, VRAI identity batches, augmentation seeds and update budget. '
                       'Only identity-crop geometry/size differs. Full-frame preprocessing is unchanged. No new distillation.')
        caveat = ('Each arm selects on calibration with its own inference policy. Epoch-zero rows already differ due to preprocessing; '
                  'compare improvement over each origin and use the cross-policy grid to compare selected weights under the same inference policy. '
                  'Compute per update differs with image size. This combines geometry and resolution, and one seed is preliminary evidence.')
        grid = output/'cross_policy.csv'
        if grid.exists():
            extra = '<h2>Selected weights evaluated under both crop policies</h2>'+table(rows(grid))
    (output/'summary.html').write_text('<!doctype html><meta charset="utf-8"><title>Matched ReID continuation</title>'
        '<style>body{font:16px system-ui;margin:32px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccc}</style>'
        f'<h1>{escape(title)}</h1><p>{escape(description)} Final VRAI test is not evaluated.</p><p>'+escape(note)+'</p>'
        +table(summaries)+extra+'<h2>Calibration history</h2>'+table(curves)+
        '<p>Calibration selects checkpoints: these are development results, not blind test scores. KITTI is conventional validation, '
        'using the same baseline tracker.</p><p>'+escape(caveat)+'</p>')
    print(f'Report: {output / "summary.html"}', flush=True)
