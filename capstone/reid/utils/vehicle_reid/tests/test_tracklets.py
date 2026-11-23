"""Continuity must survive global rejection without authorizing a false recovery."""
import pickle
from pathlib import Path
import unittest
import numpy as np
import yaml
from utils.vehicle_reid.tracklets import TrackletAssociation, tracking_identity


class TrackletTests(unittest.TestCase):
    def setUp(self):
        self.spec = yaml.safe_load((Path(__file__).resolve().parents[3]/'configs/vehicle_reid.yaml').read_text())['association']
        self.tracker = TrackletAssociation(self.spec)
        self.box = np.array([10,10,90,90], dtype=np.float32)
        self.feature = np.array([1,0,0], dtype=np.float32)

    def step(self, frame, feature=None, box=None, reliable=True):
        return self.tracker.update([self.box if box is None else box],
            [self.feature if feature is None else feature], [reliable], frame)[0]

    def enroll(self):
        for frame in range(4): self.step(frame)
        self.assertTrue(self.tracker.identities[1].confirmed)

    def test_identical_vehicles_keep_local_continuity(self):
        boxes = [self.box, self.box+np.array([400,0,400,0])]
        for frame in range(12):
            result = self.tracker.update(boxes, [self.feature,self.feature], [True,True], frame)
            self.assertEqual([row['identity'] for row in result], [1,2])
            self.assertEqual([row['tracklet_id'] for row in result], [1,2])
            self.assertTrue(all(not row['memory_update'] for row in result))

    def test_global_rejection_retains_stable_local_id(self):
        self.enroll()
        self.tracker.reset_view()
        feature = np.array([.75,np.sqrt(1-.75**2),0], dtype=np.float32)
        output_ids = []
        for frame in range(20):
            result = self.step(frame, feature=feature, box=self.box+1000)
            self.assertIsNone(result['identity'])
            output_ids.append(tracking_identity(result))
        self.assertEqual(len(set(output_ids)), 1)
        self.assertEqual(len(self.tracker.identities), 1)

    def test_unreliable_observations_keep_local_ids_without_enrollment(self):
        for frame in range(12):
            result = self.step(frame, reliable=False)
            self.assertEqual(result['tracklet_id'], 1)
            self.assertIsNone(result['identity'])
        self.assertFalse(self.tracker.identities)

    def test_active_identity_cannot_be_recovered_by_another_visible_object(self):
        self.enroll()
        result = self.tracker.update([self.box,self.box+1000], [self.feature,self.feature], [True,True], 4)
        self.assertEqual([row['identity'] for row in result], [1,2])

    def test_global_recovery_requires_consistent_evidence_and_does_not_write(self):
        self.enroll()
        before = [vector.copy() for vector in self.tracker.identities[1].recent]
        self.tracker.reset_view()
        for frame in range(2):
            result = self.step(frame, box=self.box+1000)
            self.assertIsNone(result['identity'])
            self.assertEqual(result['status'], 'pending_recovery')
        result = self.step(2, box=self.box+1000)
        self.assertEqual((result['identity'], result['status']), (1,'recovered'))
        self.assertFalse(result['memory_update'])
        self.assertEqual(len(before), len(self.tracker.identities[1].recent))

    def test_ambiguous_global_candidates_do_not_erase_local_id(self):
        self.enroll()
        from utils.vehicle_reid.association import Identity
        self.tracker.identities[2] = Identity(2, [self.feature.copy()], confirmed=True)
        self.tracker.next_id = 3
        self.tracker.reset_view()
        for frame in range(8):
            result = self.step(frame)
            self.assertIsNone(result['identity'])
            self.assertEqual(result['tracklet_id'], 1)
            self.assertEqual(result['status'], 'ambiguous')

    def test_short_occlusion_preserves_binding_without_global_reconnection(self):
        self.enroll()
        for frame in range(4,10): self.tracker.update([],[],[],frame)
        result = self.step(10)
        self.assertEqual(result['identity'], 1)
        self.assertEqual(result['tracklet_id'], 1)
        self.assertNotEqual(result['status'], 'recovered')
        self.assertFalse(result['memory_update'])

    def test_owner_transfer_removes_stale_local_binding(self):
        self.enroll()
        self.tracker.update([],[],[],4)
        for frame in range(5,8): result = self.step(frame, box=self.box+1000)
        self.assertEqual(result['identity'], 1)
        self.assertEqual(result['tracklet_id'], 2)
        self.assertEqual(self.tracker.bindings, {2:1})
        result = self.tracker.update([self.box,self.box+1000], [self.feature,self.feature], [True,True], 8)
        self.assertEqual(len({row['identity'] for row in result}),2)

    def test_memory_ignores_uncertain_active_features(self):
        self.enroll()
        before = pickle.dumps(self.tracker.identities[1].recent)
        for frame in range(4,8):
            result = self.step(frame, reliable=False)
            self.assertEqual(result['identity'],1)
            self.assertFalse(result['memory_update'])
        self.assertEqual(before,pickle.dumps(self.tracker.identities[1].recent))

    def test_late_global_enrollment_keeps_the_original_output_id(self):
        first = self.step(0,reliable=False)
        initial = tracking_identity(first)
        self.assertIsNone(first['identity'])
        for frame in range(1,4):
            self.assertEqual(tracking_identity(self.step(frame,reliable=False)),initial)
        enrolled = self.step(4,reliable=True)
        self.assertEqual(enrolled['identity'],initial)
        self.assertEqual(tracking_identity(enrolled),initial)

    def test_uncertain_observation_cannot_update_local_appearance(self):
        self.enroll()
        before = self.tracker.local.tracks[0].feature.clone().numpy()
        self.step(4, feature=[.9,.1,0], reliable=False)
        np.testing.assert_array_equal(self.tracker.local.tracks[0].feature.numpy(),before)

    def test_checkpoint_preserves_pending_recovery(self):
        self.enroll()
        self.tracker.reset_view()
        self.step(0,box=self.box+1000)
        restored = pickle.loads(pickle.dumps(self.tracker))
        for frame in range(1,5):
            expected = self.step(frame,box=self.box+1000)
            actual = restored.update([self.box+1000],[self.feature],[True],frame)[0]
            self.assertEqual(expected,actual)

    def test_invalid_dimensions_and_frames_fail_without_advancing_state(self):
        self.step(0)
        with self.assertRaises(ValueError): self.step(0)
        with self.assertRaises(ValueError): self.step(1,feature=[0,1])
        self.assertEqual(self.tracker.last_frame,0)


if __name__ == '__main__':
    unittest.main()
