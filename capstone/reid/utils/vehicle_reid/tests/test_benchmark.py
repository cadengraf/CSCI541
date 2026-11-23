"""Benchmark validity checks, including real-input upstream tracker resume."""
from pathlib import Path
import pickle
import tempfile
import unittest
import numpy as np
from pycocotools import mask as rle
import yaml
from utils.tracking.common import ROOT, load, decode
from utils.vehicle_reid.baselines import Method, imports
from utils.vehicle_reid.recovery import Recovery
from utils.vehicle_reid.operating_points import calibrate, statistics


class BenchmarkTests(unittest.TestCase):
    def test_actual_trackers_resume_identically_on_real_detections(self):
        config = yaml.safe_load((ROOT/'configs/vehicle_benchmark.yaml').read_text())
        try:
            imports(config)
        except ImportError as error:
            self.skipTest(str(error))
        from yolox.tracker.basetrack import BaseTrack
        from trackers.ocsort_tracker.ocsort import KalmanBoxTracker
        paths = [ROOT/config['cache']/'0002'/f'{frame:06d}.pt' for frame in range(70)]
        if not all(path.exists() for path in paths):
            self.skipTest('Real cached input not installed')
        def methods():
            return {name: Method(name, config, {}, {}) for name in ('bytetrack', 'ocsort')}
        def run(algorithms, begin, end):
            result = {name: [] for name in algorithms}
            for frame in range(begin, end):
                raw = decode(load(paths[frame]))
                for name, algorithm in algorithms.items():
                    emitted = algorithm.update(raw, None, None, frame)
                    result[name].append([(index, value['identity']) for index, value in emitted])
            return result
        full = run(methods(), 0, 70)
        partial = methods()
        first = run(partial, 0, 25)
        payload = pickle.dumps((partial, BaseTrack._count, KalmanBoxTracker.count))
        methods()  # Simulate a fresh process initializing empty algorithms.
        restored, byte_count, oc_count = pickle.loads(payload)
        BaseTrack._count, KalmanBoxTracker.count = byte_count, oc_count
        rest = run(restored, 25, 70)
        for name in full:
            self.assertEqual(full[name], first[name]+rest[name])
            self.assertEqual(restored[name].mapping_drops, 0)

    def test_no_prior_reference_cannot_inflate_recovery_rate(self):
        mask = np.ones((10,10), dtype=bool)
        counts = rle.encode(np.asfortranarray(mask.astype(np.uint8)))['counts'].decode()
        with tempfile.TemporaryDirectory() as directory:
            truth = Path(directory)/'truth.txt'
            truth.write_text(f'0 1001 1 10 10 {counts}\n5 1001 1 10 10 {counts}\n')
            metric = Recovery(truth)
            metric.update(0, [], [])
            for frame in range(1,5): metric.update(frame, [], [])
            metric.update(5, [mask], [dict(identity=1)])
            self.assertEqual(metric.counts['visibility_returns'], 1)
            self.assertEqual(metric.counts['returns_without_prior_match'], 1)
            self.assertEqual(metric.counts['recovered'], 0)

    def test_temporal_recovery_delay_counts_only_the_old_identity(self):
        mask = np.ones((10,10), dtype=bool)
        counts = rle.encode(np.asfortranarray(mask.astype(np.uint8)))['counts'].decode()
        with tempfile.TemporaryDirectory() as directory:
            truth = Path(directory)/'truth.txt'
            truth.write_text(''.join(f'{frame} 1001 1 10 10 {counts}\n' for frame in [0,5,6,7]))
            metric = Recovery(truth)
            metric.update(0, [mask], [dict(identity=1)])
            for frame in range(1,5): metric.update(frame, [], [])
            metric.update(5, [mask], [dict(identity=None)])
            metric.update(6, [mask], [dict(identity=2)])
            metric.update(7, [mask], [dict(identity=1)])
            self.assertEqual(metric.counts['recovered'], 1)
            self.assertEqual(metric.events[0]['delay_frames'], 2)
            self.assertEqual(metric.events[0]['gap'], 4)

    def test_calibration_budget_does_not_become_a_final_guarantee(self):
        calibration = dict(similarity=np.array([.9,.8,.7,.3]), margin=np.ones(4),
            known=np.array([True,True,False,False]), correct=np.array([True,True,False,False]), identities=np.array(['a','b','c','d']))
        selected = calibrate(calibration, .01)
        self.assertEqual(selected['unknown_far'], 0)
        final = dict(calibration)
        final['similarity'] = np.array([.9,.8,.99,.99])
        result = statistics(final, selected['threshold'], selected['margin'])
        self.assertEqual(result['unknown_far'], 1.)


if __name__ == '__main__':
    unittest.main()
