"""Error budgets cannot be satisfied by hiding local errors or unscored claims."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from pycocotools import mask as rle
import yaml
from utils.tracking.common import ROOT,load
from utils.full_frame.tracker import VehicleTracker
from utils.vehicle_reid.reference import ReferenceAssociation
from utils.vehicle_reid.recovery import Recovery
from utils.vehicle_reid.budget import validate_development,select_point


class RecoveryBudgetTests(unittest.TestCase):
    def test_default_reference_matches_original_on_real_inputs(self):
        pipeline=yaml.safe_load((ROOT/'configs/vehicle_reid.yaml').read_text())
        reference=ReferenceAssociation(pipeline['association'])
        baseline=VehicleTracker(pipeline['association']['local'])
        bank=ROOT/'data/vehicle_benchmark/specialist/0002/state.pt'
        if not bank.exists(): self.skipTest('Real observations not installed')
        for frame,obs in enumerate(load(bank)['observations'][:100]):
            expected=baseline.update(dict(boxes=obs['boxes'],embeddings=obs['embeddings']),frame)
            actual=reference.update(obs['boxes'].numpy(),obs['embeddings'].numpy(),[frame%2==0]*len(expected),frame)
            self.assertEqual([d['identity'] for d in actual],expected)
            for a,b in zip(reference.local.tracks,baseline.tracks):
                torch.testing.assert_close(a.feature,b.feature,rtol=0,atol=0)
                torch.testing.assert_close(a.box,b.box,rtol=0,atol=0)
                torch.testing.assert_close(a.velocity,b.velocity,rtol=0,atol=0)

    def test_calibration_excludes_validation(self):
        c=yaml.safe_load((ROOT/'configs/recovery_budget.yaml').read_text())
        validate_development(c)
        c['sequences']=[2]
        with self.assertRaises(ValueError): validate_development(c)

    def test_low_support_or_unscored_claims_cannot_pass_budget(self):
        choices=dict(a=dict(recovery_similarity=.8,recovery_margin=.1,recovery_hits=3))
        budget=dict(max_false_reconnection_rate=.01,minimum_scored_accepts=20,require_no_unscored_claims=True)
        values=dict(a=dict(global_scored=2,global_correct=2,global_false=0,global_false_rate=0.,identity_any_false_upper95=.78))
        self.assertIsNone(select_point(values,choices,budget))
        values['a'].update(global_scored=30,global_correct=30,global_unscored=1)
        self.assertIsNone(select_point(values,choices,budget))
        values['a']['global_unscored']=0
        self.assertEqual(select_point(values,choices,budget),'a')
        self.assertFalse(values['a']['identity_risk_bound_within_budget'])

    def test_local_and_global_false_actions_are_separate(self):
        mask=np.ones((10,10),dtype=bool)
        encoded=rle.encode(np.asfortranarray(mask.astype(np.uint8)))['counts'].decode()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'truth.txt'
            path.write_text(f'0 1001 1 10 10 {encoded}\n2 1002 1 10 10 {encoded}\n4 1003 1 10 10 {encoded}\n')
            metric=Recovery(path)
            metric.update(0,[mask],[dict(identity=1,association_kind='local_birth')])
            metric.update(1,[],[])
            metric.update(2,[mask],[dict(identity=1,association_kind='local_reacquisition')])
            metric.update(3,[],[])
            metric.update(4,[mask],[dict(identity=1,association_kind='global_recovery',status='recovered')])
            result=metric.action_summary()
            self.assertEqual(result['local_false'],1)
            self.assertEqual(result['global_false'],1)
            self.assertEqual(result['global_scored'],1)
            self.assertEqual(result['global_false_rate'],1.)

    def test_missing_mask_label_is_unscored_instead_of_correct(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'truth.txt'; path.write_text('')
            metric=Recovery(path)
            metric.update(0,[np.ones((2,2),dtype=bool)],
                [dict(identity=1,status='recovered',association_kind='global_recovery')])
            result=metric.action_summary()
            self.assertEqual(result['global_unscored'],1)
            self.assertEqual(result['global_scored'],0)
            self.assertIsNone(result['global_false_rate'])

    def test_prepared_labels_cannot_be_reused_for_another_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'truth.txt'; path.write_text('')
            metric=Recovery(path)
            prepared=metric.prepare_frame(0,[])
            with self.assertRaises(ValueError): metric.update(1,[],[],prepared=prepared)

    def test_legacy_audit_is_explicitly_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'truth.txt'; path.write_text('')
            metric=Recovery(path)
            self.assertFalse(metric.action_summary()['available'])


if __name__=='__main__': unittest.main()
