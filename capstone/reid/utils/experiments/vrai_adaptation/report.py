"""Single-run learning curve and joint-system preservation summary."""
import csv
from html import escape
from utils.teacher.evaluation import write_csv
from .settings import read


def render():
    settings, root = read()
    cfg = settings['matched_reid']; output = root/cfg['output']
    with (root/settings['training']['log']).open(newline='') as file:
        history = list(csv.DictReader(file))
    calibration = {int(row['epoch']): row for row in history if row['event'] == 'calibration'}
    if cfg['epochs'] not in calibration:
        raise ValueError('Complete adaptation before rendering its report')
    initial, final = calibration[0], calibration[cfg['epochs']]
    selected = max(calibration, key=lambda epoch: float(calibration[epoch]['calibration_map']))
    summary = [{'checkpoint': name, 'epoch': int(row['epoch']),
                'Rank1': 100*float(row['calibration_rank1']), 'mAP': 100*float(row['calibration_map'])}
               for name, row in [('origin', initial), ('selected', calibration[selected]), ('last', final)]]
    write_csv(output/'summary.csv', summary)
    def table(rows):
        keys = list(rows[0])
        return '<table><tr>'+''.join(f'<th>{escape(k)}</th>' for k in keys)+'</tr>'+''.join(
            '<tr>'+''.join(f'<td>{value:.3f}</td>' if isinstance(value, float) else f'<td>{escape(str(value))}</td>'
                         for value in row.values())+'</tr>' for row in rows)+'</table>'
    curves = [{'epoch': epoch, 'Rank1': 100*float(row['calibration_rank1']),
               'mAP': 100*float(row['calibration_map']), 'skipped_updates': row['skipped_updates']}
              for epoch, row in sorted(calibration.items())]
    with (output/'vrai_only/kitti_mots/summary.csv').open(newline='') as file:
        tracking = list(csv.DictReader(file))
    (output/'adaptation.html').write_text('<!doctype html><meta charset="utf-8"><title>VRAI-only adaptation</title>'
        '<style>body{font:16px system-ui;margin:32px}td,th{padding:9px;border:1px solid #ccc}table{border-collapse:collapse}</style>'
        '<h1>VRAI-only adaptation of the combined model</h1><p>No KITTI training images, direct box/mask losses, or new distillation. '
        'The shared identity pathway and last four backbone blocks adapt using VRAI classification and triplet losses. '
        'Predicted masks still guide pooling and receive identity gradients.</p>'+table(summary)+
        f'<p>Scheduled optimizer updates: {cfg["epochs"]*cfg["steps_per_epoch"]}; skipped: {final["skipped_updates"]}. '
        f'Selected epoch: {selected}. Epoch zero is the origin, not newly trained weights.</p>'+
        '<h2>Calibration learning curve</h2>'+table(curves)+'<h2>Selected checkpoint on KITTI validation</h2>'+table(tracking)+
        '<p>Calibration selects checkpoints and is development data, not a blind final test. This run has a different '
        'identity exposure/update budget from earlier joint runs, so improvement does not isolate the causal effect of '
        'removing KITTI. Shared features and masks can drift; inspect KITTI preservation before deployment. '
        'The architecture remains the combined detector/mask/embedding model; this does not train a separate specialist.</p>'+
        '<p><a href="vrai_only/summary.html">Detailed selected-model retrieval, KITTI and timing report</a></p>')
    print(f'Report: {output / "adaptation.html"}', flush=True)
