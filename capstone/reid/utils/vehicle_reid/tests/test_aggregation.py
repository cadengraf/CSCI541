"""Temporal evidence must be causal, resumable, and resistant to contamination."""
import pickle
import unittest
from pathlib import Path
import numpy as np
import yaml
from utils.vehicle_reid.aggregation import TrackletEvidenceAssociation, Sample


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.spec=yaml.safe_load((Path(__file__).resolve().parents[3]/'configs/vehicle_reid.yaml').read_text())['association']
        self.spec.update(recovery_hits=1,aggregation=dict(min_samples=3,max_samples=8,
            sample_gap=2,min_span=4,coherence=.85,pool_source=True,quarantine_recovered_memory=True))
        self.actor=TrackletEvidenceAssociation(self.spec)
        self.box=np.array([10,10,90,90],dtype=np.float32)
        self.feature=np.array([1,0,0],dtype=np.float32)

    def step(self,frame,good=True):
        return self.actor.update([self.box],[self.feature],[good],frame)[0]

    def enroll(self):
        for frame in range(6):self.step(frame)
        self.assertIsNotNone(self.actor.identities[1].enrollment_pool)

    def test_spaced_evidence_delays_recovery_and_survives_resume(self):
        self.enroll()
        source=pickle.dumps(self.actor.identities[1].references)
        self.actor.reset_view()
        for frame in range(4):
            self.assertIsNone(self.step(frame)['identity'])
        restored=pickle.loads(pickle.dumps(self.actor))
        expected=self.step(4)
        actual=restored.update([self.box],[self.feature],[True],4)[0]
        self.assertEqual(expected,actual)
        self.assertEqual((expected['identity'],expected['status']),(1,'recovered'))
        self.assertEqual(source,pickle.dumps(self.actor.identities[1].references))
        before=pickle.dumps(self.actor.identities[1].recent)
        for frame in range(5,18):self.assertFalse(self.step(frame)['memory_update'])
        self.assertEqual(before,pickle.dumps(self.actor.identities[1].recent))

    def test_uncertain_frame_resets_query_evidence(self):
        self.enroll();self.actor.reset_view()
        for frame in range(4):self.step(frame)
        self.step(4,False)
        for frame in range(5,9):self.assertIsNone(self.step(frame)['identity'])
        self.assertEqual(self.step(9)['identity'],1)

    def test_recovery_cannot_create_missing_source_consensus(self):
        for frame in range(3):self.step(frame)
        self.assertFalse(hasattr(self.actor.identities[1],'enrollment_pool'))
        self.actor.reset_view()
        for frame in range(12):self.assertIsNone(self.step(frame)['identity'])
        self.assertFalse(hasattr(self.actor.identities[1],'enrollment_pool'))

    def test_contradictory_features_reject_pool_without_crashing(self):
        samples=[Sample(0,self.feature,1),Sample(2,-self.feature,2),Sample(4,self.feature,1)]
        self.assertIsNone(self.actor._pool(samples))
        samples[1].weight=1
        self.assertIsNone(self.actor._pool(samples))


if __name__=='__main__':unittest.main()
