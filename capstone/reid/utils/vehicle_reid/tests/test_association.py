"""Identity errors are costly: test rejection, persistence, and memory protection."""
import unittest
from pathlib import Path
import numpy as np
import yaml
from utils.vehicle_reid.association import Association


class AssociationTests(unittest.TestCase):
    def setUp(self):
        self.spec = yaml.safe_load((Path(__file__).resolve().parents[3] / 'configs/vehicle_reid.yaml').read_text())['association']
        self.tracker = Association(self.spec)
        self.box = np.array([10, 10, 90, 90], dtype=np.float32)
        self.feature = np.array([1., 0., 0.], dtype=np.float32)

    def step(self, frame, feature=None, box=None, reliable=True):
        return self.tracker.update([self.box if box is None else box],
                                   [self.feature if feature is None else feature], [reliable], frame)[0]

    def confirmed(self):
        for frame in range(4):
            result = self.step(frame)
        self.assertEqual(result['identity'], 1)
        self.assertTrue(self.tracker.identities[1].confirmed)

    def test_return_requires_repeated_evidence_without_memory_write(self):
        self.confirmed()
        before = len(self.tracker.identities[1].recent)
        self.tracker.update([], [], [], 20)
        distant = self.box+1000
        self.assertEqual(self.step(21, box=distant)['status'], 'pending_recovery')
        self.assertIsNone(self.step(22, box=distant)['identity'])
        result = self.step(23, box=distant)
        self.assertEqual((result['identity'], result['status']), (1, 'recovered'))
        self.assertEqual(len(self.tracker.identities[1].recent), before)

    def test_similar_competitors_rejected(self):
        self.confirmed()
        from utils.vehicle_reid.association import Identity
        self.tracker.identities[2] = Identity(2, [self.feature.copy()], confirmed=True)
        self.tracker.reset_view()
        result = self.step(0)
        self.assertIsNone(result['identity'])
        self.assertEqual(result['status'], 'ambiguous')

    def test_two_similar_detections_cannot_claim_one_identity(self):
        self.confirmed()
        self.tracker.reset_view()
        results = self.tracker.update([self.box, self.box+200], [self.feature, self.feature], [True, True], 0)
        self.assertTrue(all(result['identity'] is None for result in results))

    def test_uncertain_observation_never_updates_memory(self):
        self.confirmed()
        before = [value.copy() for value in self.tracker.identities[1].recent]
        for frame in range(4, 8):
            self.assertFalse(self.step(frame, reliable=False)['memory_update'])
        self.assertEqual(len(before), len(self.tracker.identities[1].recent))
        for a, b in zip(before, self.tracker.identities[1].recent):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(self.tracker.identities[1].references[0], self.feature)

    def test_camera_change_discards_spatial_state(self):
        self.confirmed()
        self.tracker.reset_view()
        self.assertIsNone(self.tracker.identities[1].box)
        self.assertIsNone(self.step(0, box=self.box+1000)['identity'])
        self.step(1, box=self.box+1000)
        self.assertEqual(self.step(2, box=self.box+1000)['identity'], 1)

    def test_unrelated_vehicle_gets_distinct_identity(self):
        self.confirmed()
        self.assertEqual(self.step(4, feature=[0., 1., 0.], box=self.box+100)['identity'], 2)

    def test_intermediate_score_stays_unknown(self):
        self.confirmed()
        self.assertIsNone(self.step(20, feature=[.75, np.sqrt(1-.75**2), 0])['identity'])
        self.assertEqual(len(self.tracker.identities), 1)

    def test_nonmonotonic_frame_and_invalid_embedding_fail(self):
        self.step(0)
        with self.assertRaises(ValueError):
            self.step(0)
        with self.assertRaises(ValueError):
            self.step(1, feature=[0, 0, 0])

    def test_interrupted_pending_does_not_confirm(self):
        self.confirmed()
        self.step(20)
        self.tracker.update([], [], [], 21)
        self.assertEqual(self.step(22)['status'], 'pending_recovery')
        self.assertEqual(self.step(23)['status'], 'pending_recovery')


if __name__ == '__main__':
    unittest.main()
