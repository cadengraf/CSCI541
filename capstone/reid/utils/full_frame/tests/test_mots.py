"""Behavior checks for identity continuity, masks, leakage, and MOTS metrics."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from ..mots_benchmark import write_frame
from ..mots_evaluation import evaluate
from ..protocol import load, check_checkpoint
from ..tracker import VehicleTracker, disjoint_predictions
from .. import mots_benchmark


class MOTSPipelineTests(unittest.TestCase):
    def setUp(self):
        self.config, _ = load()
        self.settings = self.config['tracking']

    def test_reordered_detections_and_one_frame_occlusion_keep_ids(self):
        tracker = VehicleTracker(self.settings)
        boxes = torch.tensor([[0., 0., 20., 20.], [60., 0., 80., 20.]])
        features = torch.eye(2)
        first = tracker.update({'boxes': boxes, 'embeddings': features}, 0)
        tracker.update({'boxes': boxes[:0], 'embeddings': features[:0]}, 1)
        later = tracker.update({'boxes': boxes.flip(0) + 2, 'embeddings': features.flip(0)}, 2)
        self.assertEqual(later, first[::-1])
        far = tracker.update({'boxes': boxes[:1] + 10000, 'embeddings': features[:1]}, 3)
        self.assertNotIn(far[0], first)

    def test_export_masks_do_not_overlap(self):
        prediction = {'boxes': torch.tensor([[0., 0., 8., 8.]] * 2),
                      'scores': torch.tensor([0.9, 0.8]), 'embeddings': torch.eye(2),
                      'masks': torch.ones((2, 1, 8, 8))}
        result = disjoint_predictions(prediction, self.settings)
        self.assertEqual(len(result['masks']), 1)
        self.assertTrue((result['masks'].sum(0) <= 1).all())

    def test_old_split_checkpoint_is_rejected(self):
        saved = {'settings': dict(self.config['full_frame']), 'epoch': 8}
        saved['settings']['train_sequences'] = list(range(16))
        with self.assertRaises(ValueError):
            check_checkpoint(saved)

    def test_trackeval_perfect_identity_switch_and_ignore_region(self):
        masks = torch.zeros((2, 12, 12), dtype=torch.bool)
        masks[0, 1:5, 1:5] = True
        masks[1, 7:11, 7:11] = True
        with tempfile.TemporaryDirectory(prefix='mots_metrics_test_') as temporary:
            root = Path(temporary)
            gt = root / 'gt/instances_txt'
            gt.mkdir(parents=True)
            with (gt / '0002.txt').open('w') as file:
                for frame in range(2):
                    write_frame(file, frame, {'masks': masks[:1]}, [1001])
                    buffer = StringIO()
                    write_frame(buffer, frame, {'masks': masks[1:]}, [10000])
                    file.write(buffer.getvalue().replace(' 1 12 12 ', ' 10 12 12 '))
            for name, ids in (('perfect', [1, 1]), ('switch', [1, 2])):
                folder = root / 'trackers' / name / 'data'
                folder.mkdir(parents=True)
                with (folder / '0002.txt').open('w') as file:
                    for frame, identity in enumerate(ids):
                        write_frame(file, frame, {'masks': masks}, [identity, 99])
            with redirect_stdout(StringIO()):
                rows = evaluate(root / 'gt', root / 'trackers', ['perfect', 'switch'],
                                {'0002': 2}, root / 'metrics', print_results=False)
            perfect, switch = rows
            self.assertAlmostEqual(perfect['HOTA'], 100)
            self.assertAlmostEqual(perfect['IDF1'], 100)
            self.assertAlmostEqual(perfect['sMOTSA'], 100)
            self.assertEqual(perfect['FP'], 0)  # prediction inside ignore region removed
            self.assertEqual(switch['IDSW'], 1)
            self.assertAlmostEqual(switch['sMOTSA'], 50)
            self.assertLess(switch['IDF1'], perfect['IDF1'])

    def test_full_export_cache_and_report_with_known_predictions(self):
        calls = []

        class KnownPredictionModel(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()

            def load_state_dict(self, state):
                pass

            def forward(self, images):
                calls.append(1)
                return [{'boxes': torch.tensor([[0., 0., 12., 12.]]),
                         'scores': torch.tensor([0.99]), 'embeddings': torch.ones((1, 128)),
                         'masks': torch.ones((1, 1, 12, 12))}]

        with tempfile.TemporaryDirectory(prefix='mots_export_test_') as directory:
            root = Path(directory)
            settings = self.config['full_frame']
            checkpoint = root / settings['checkpoint']
            checkpoint.parent.mkdir(parents=True)
            torch.save({'settings': settings, 'epoch': settings['epochs'],
                        'model': {'id_classifier.weight': torch.zeros((1, 128))}}, checkpoint)
            for sequence in settings['val_sequences']:
                sequence = f'{sequence:04d}'
                folder = root / settings['data_root'] / 'training/image_02' / sequence
                folder.mkdir(parents=True)
                truth = root / settings['data_root'] / 'instances_txt' / f'{sequence}.txt'
                truth.parent.mkdir(parents=True, exist_ok=True)
                with truth.open('w') as file:
                    for frame in range(2):
                        Image.new('RGB', (12, 12)).save(folder / f'{frame:06d}.png')
                        write_frame(file, frame, {'masks': torch.ones((1, 12, 12), dtype=torch.bool)}, [1001])
            with patch.object(mots_benchmark, 'load', return_value=(self.config, root)), \
                    patch.object(mots_benchmark, 'FullFrameVehicleNet', KnownPredictionModel), \
                    redirect_stdout(StringIO()):
                mots_benchmark.run()
                first_calls = len(calls)
                mots_benchmark.run()
                self.assertEqual(len(calls), first_calls)  # completed sequences reused
                mots_benchmark.run(score_only=True)
            self.assertEqual(first_calls, 18)
            import csv
            report = root / self.settings['metrics_root']
            with (report / 'summary.csv').open() as file:
                row = next(csv.DictReader(file))
            self.assertAlmostEqual(float(row['sMOTSA']), 100)
            self.assertTrue((report / 'summary.html').is_file())
            self.assertTrue((report / 'timing.csv').is_file())


if __name__ == '__main__':
    unittest.main()
