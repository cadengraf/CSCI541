"""Adapter to pinned, unmodified TrackEval KITTI MOTS mask evaluation."""

from contextlib import contextmanager
import csv
from html import escape
from pathlib import Path
import numpy as np

TRACKEVAL_REVISION = '12c8791b303e0a0b50f753af204249e622d0281a'


@contextmanager
def _numpy_compatibility():
    # TrackEval uses legacy dtype aliases removed by modern NumPy. Restore only
    # missing aliases for the duration of evaluation; metric formulas stay intact.
    aliases = {'float': float, 'int': int, 'bool': bool}
    missing = [name for name in aliases if name not in np.__dict__]
    for name in missing:
        setattr(np, name, aliases[name])
    try:
        yield
    finally:
        for name in missing:
            delattr(np, name)


def evaluate(gt_root, trackers_root, names, sequences, output, print_results=True):
    try:
        import trackeval
        from pycocotools import mask  # noqa: F401
    except ImportError as error:
        raise RuntimeError('Install benchmark dependencies: python -m pip install -r requirements-mots.txt') from error
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with _numpy_compatibility():
        dataset = trackeval.datasets.KittiMOTS({
            'GT_FOLDER': str(gt_root), 'GT_LOC_FORMAT': '{gt_folder}/instances_txt/{seq}.txt',
            'TRACKERS_FOLDER': str(trackers_root), 'OUTPUT_FOLDER': str(output),
            'TRACKERS_TO_EVAL': list(names), 'CLASSES_TO_EVAL': ['car'],
            'SEQ_INFO': sequences, 'PRINT_CONFIG': False})
        evaluator = trackeval.Evaluator({
            'USE_PARALLEL': False, 'PRINT_RESULTS': print_results,
            'PRINT_ONLY_COMBINED': True, 'PRINT_CONFIG': False,
            'TIME_PROGRESS': False, 'DISPLAY_LESS_PROGRESS': True,
            'PLOT_CURVES': False, 'OUTPUT_SUMMARY': True,
            'OUTPUT_DETAILED': True, 'LOG_ON_ERROR': None})
        metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({'PRINT_CONFIG': False}),
                   trackeval.metrics.Identity({'PRINT_CONFIG': False})]
        results, messages = evaluator.evaluate([dataset], metrics)
    rows = []
    for name in names:
        if messages['KittiMOTS'][name] != 'Success':
            raise RuntimeError(f'TrackEval failed for {name}: {messages["KittiMOTS"][name]}')
        combined = results['KittiMOTS'][name]['COMBINED_SEQ']['car']
        clear = combined['CLEAR']
        # With mask similarities, TrackEval's sMOTA/MOTA/MOTP are respectively
        # sMOTSA/MOTSA/MOTSP. All score columns below are percentages.
        row = {'model': name, 'HOTA': 100 * float(np.mean(combined['HOTA']['HOTA'])),
               'DetA': 100 * float(np.mean(combined['HOTA']['DetA'])),
               'AssA': 100 * float(np.mean(combined['HOTA']['AssA'])),
               'IDF1': 100 * float(combined['Identity']['IDF1']),
               'sMOTSA': 100 * float(clear['sMOTA']), 'MOTSA': 100 * float(clear['MOTA']),
               'MOTSP': 100 * float(clear['MOTP']), 'IDSW': int(clear['IDSW']),
               'TP': int(clear['CLR_TP']), 'FP': int(clear['CLR_FP']), 'FN': int(clear['CLR_FN'])}
        rows.append(row)
    temporary = output / 'summary.partial.csv'
    with temporary.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / 'summary.csv')
    header = ''.join(f'<th>{escape(key)}</th>' for key in rows[0])
    body = ''.join('<tr>' + ''.join(f'<td>{value:.2f}</td>' if isinstance(value, float)
                                  else f'<td>{escape(str(value))}</td>' for value in row.values())
                   + '</tr>' for row in rows)
    (output / 'summary.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>KITTI MOTS vehicle benchmark</title>'
        '<style>body{font:16px system-ui;margin:40px;color:#172033}table{border-collapse:collapse}'
        'th,td{padding:12px;border-bottom:1px solid #ccd;text-align:right}th:first-child,td:first-child{text-align:left}</style>'
        '<h1>KITTI MOTS vehicle benchmark</h1><p>Conventional validation split; TrackEval mask evaluation. '
        'Scores are percentages. Lower IDSW, FP, and FN are better.</p>'
        f'<table><tr>{header}</tr>{body}</table>'
        '<p>Compare only methods evaluated on the same sequences. External training data and initialization '
        'must be reported separately. This table makes no state-of-the-art claim.</p>'
        f'<p>TrackEval revision: {TRACKEVAL_REVISION}</p>')
    return rows
