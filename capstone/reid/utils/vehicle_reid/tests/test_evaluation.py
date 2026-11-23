"""Check that diagnostics expose false reconnects and missed returns."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from pycocotools import mask as rle
from utils.vehicle_reid.evaluation import Diagnostics
from utils.vehicle_reid.retrieval import open_set


class DiagnosticTests(unittest.TestCase):
    def test_return_detector_miss_and_delayed_recovery(self):
        mask = np.ones((10, 10), dtype=bool)
        encoded = rle.encode(np.asfortranarray(mask.astype(np.uint8)))['counts'].decode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'truth.txt'
            path.write_text(''.join(f'{frame} 1001 1 10 10 {encoded}\n' for frame in (0, 3, 4)))
            evaluator = Diagnostics(dict(truth=str(path), min_mask_iou=.5, target_truth_id=1001))
            decision = dict(identity=1, status='matched')
            evaluator.update(0, [mask], [decision], 1)
            evaluator.update(1, [], [], 1)
            evaluator.update(2, [], [], 1)
            evaluator.update(3, [], [], 1)
            evaluator.update(4, [mask], [dict(identity=1, status='recovered')], 1)
            self.assertEqual(evaluator.counts['missed_on_return'], 1)
            self.assertEqual(evaluator.counts['delayed_recoveries'], 1)
            self.assertEqual(evaluator.counts['correct_reconnection'], 1)
            self.assertEqual(evaluator.counts['target_missed_visible'], 1)

    def test_wrong_object_reconnection_is_counted(self):
        mask = np.ones((10, 10), dtype=bool)
        encoded = rle.encode(np.asfortranarray(mask.astype(np.uint8)))['counts'].decode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'truth.txt'
            path.write_text(f'0 1001 1 10 10 {encoded}\n1 1002 1 10 10 {encoded}\n')
            evaluator = Diagnostics(dict(truth=str(path), min_mask_iou=.5, target_truth_id=1001))
            evaluator.update(0, [mask], [dict(identity=1, status='new')], 1)
            evaluator.update(1, [mask], [dict(identity=1, status='recovered')], 1)
            self.assertEqual(evaluator.counts['false_reconnection'], 1)
            self.assertEqual(evaluator.counts['target_false_match'], 1)

    def test_unknown_rejection_and_known_wrong_match_are_distinct(self):
        gallery = torch.eye(5)
        query = torch.stack([gallery[0], gallery[1], gallery[2], gallery[3]])
        result = open_set(query, gallery, ['0', '1', '1', '3'], list('01234'), .8, .1)
        self.assertEqual(result['correct_unknown_rejections'], 1)
        self.assertEqual(result['wrong_known_matches'], 1)
        self.assertEqual(result['correct_matches'], 2)


if __name__ == '__main__':
    unittest.main()
