from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from pycocotools import mask as rle
import torch
from utils.vehicle_reid.appearance_encoder import export
from utils.vehicle_reid.size_audit import SizeAudit,return_sizes


class EncoderBenchmarkTests(unittest.TestCase):
    def test_inference_export_preserves_weights_without_optimizer_or_rng(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source=root/'last.pt'
            weights={'feature':torch.arange(8).reshape(2,4)}
            state=dict(finished=True,epoch=3,step=0,model=weights,settings={'model':{'image_size':256}},
                signature=dict(name='degradation',variant=dict(augmentation=True,hard_sampling=False,metric='triplet')),
                identity_mapping={'a':0},optimizer={'unused':'large'},numpy_rng='unused')
            torch.save(state,source)
            with patch('utils.vehicle_reid.appearance_encoder.ROOT',root):
                manifest=export('last.pt','export/selection.yaml')
                saved=torch.load(root/manifest['candidate']['checkpoint'],weights_only=False)
                self.assertTrue(torch.equal(saved['model']['feature'],weights['feature']))
                self.assertNotIn('optimizer',saved);self.assertNotIn('numpy_rng',saved)
                self.assertEqual(manifest,export('last.pt','export/selection.yaml'))

    def test_clean_selected_starting_checkpoint_cannot_be_exported_as_candidate(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);torch.save(dict(finished=False,epoch=0,step=0),root/'best.pt')
            with patch('utils.vehicle_reid.appearance_encoder.ROOT',root):
                with self.assertRaises(ValueError):export('best.pt','export/selection.yaml')

    def test_size_audit_separates_errors_and_unscored_claims(self):
        audit=SizeAudit(48)
        decisions=[dict(identity=1,association_kind='global_recovery'),
                   dict(identity=2,association_kind='global_recovery'),
                   dict(identity=3,global_identity=None)]
        audit.update(np.array([[0,0,20,20],[0,0,80,80],[0,0,20,20]]),decisions,
            dict(matched={0:11,2:12}),{1:10,2:20})
        summary=audit.summary()
        self.assertEqual(summary['small']['global_false'],1)
        self.assertEqual(summary['small']['unresolved_global_observations'],1)
        self.assertEqual(summary['larger']['global_unscored'],1)

    def test_return_size_uses_gt_at_return_and_counts_missing_reference(self):
        array=np.zeros((100,100),dtype=np.uint8);array[10:30,10:30]=1
        encoded=rle.encode(np.asfortranarray(array))
        class Metric:
            truth={0:[(1,encoded),(2,encoded)],2:[(1,encoded),(2,encoded)]}
            events=[dict(truth_id=1,returned_at=2,result='recovered')]
            pending={}
        result=return_sizes(Metric(),48)
        self.assertEqual(result['small']['visibility_returns'],2)
        self.assertEqual(result['small']['recovered'],1)
        self.assertEqual(result['small']['missed_returns'],1)


if __name__=='__main__':unittest.main()
