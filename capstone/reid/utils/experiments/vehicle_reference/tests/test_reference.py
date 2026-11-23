import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from utils.tracking.common import load
from utils.experiments.vehicle_reference.inference import extract,reuse_target
from utils.experiments.vehicle_reference.command import freeze_points


class Encoder(torch.nn.Module):
    def __init__(self,dimension,fail_after=None):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(1));self.dimension=dimension;self.calls=0;self.fail_after=fail_after
    def forward(self,batch):
        if self.fail_after is not None and self.calls>=self.fail_after:raise RuntimeError('interrupted')
        self.calls+=1
        return torch.ones(len(batch),self.dimension)*self.weight


class ReferenceTests(unittest.TestCase):
    def test_different_embedding_dimensions_and_committed_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);rows=[]
            for i in range(5):
                file=path/f'{i}.png';Image.new('RGB',(16,16),(i,30,40)).save(file);rows.append({'path':str(file)})
            def conversion(value):return torch.from_numpy(np.asarray(value).copy()).permute(2,0,1).float()/255
            config=dict(batch_size=2,checkpoint_batches=1)
            for dimension in [768,3840]:
                state=dict(parts={},seconds={});destination=path/f'{dimension}.pt';broken=Encoder(dimension,fail_after=1)
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    extract(broken,conversion,{'source':rows},['source'],state,destination,config,dimension)
                state=load(destination);self.assertEqual(sum(map(len,state['parts']['source'])),2)
                model=Encoder(dimension)
                result=extract(model,conversion,{'source':rows},['source'],state,destination,config,dimension)
                self.assertEqual(model.calls,2);self.assertEqual(result['source'].shape,(5,dimension))
                torch.testing.assert_close(result['source'].norm(dim=1),torch.ones(5))
                extract(model,conversion,{'source':rows},['source'],state,destination,config,dimension)
                self.assertEqual(model.calls,2)

    def test_rejects_wrong_feature_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'x.png';Image.new('RGB',(16,16)).save(file)
            with self.assertRaisesRegex(ValueError,'Invalid reference embeddings'):
                extract(Encoder(5),lambda value:torch.zeros(3,16,16),{'query':[{'path':str(file)}]},['query'],
                        dict(parts={},seconds={}),Path(directory)/'state.pt',dict(batch_size=1,checkpoint_batches=1),768)

    def test_target_threshold_freeze_requires_every_source_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            config=dict(output=directory,models=['a','b']);point=dict(threshold=.7,margin=.1)
            incomplete=lambda config,name:({'calibration_complete':name=='a','operating_point':point},None)
            with patch('utils.experiments.vehicle_reference.command.open_state',incomplete):
                with self.assertRaisesRegex(ValueError,'Complete all source fits'):freeze_points(config)
            complete=lambda config,name:({'calibration_complete':True,'operating_point':point},None)
            with patch('utils.experiments.vehicle_reference.command.open_state',complete):freeze_points(config)
            changed=lambda config,name:({'calibration_complete':True,'operating_point':dict(threshold=.8,margin=.1)},None)
            with patch('utils.experiments.vehicle_reference.command.open_state',changed):
                with self.assertRaisesRegex(ValueError,'Frozen thresholds changed'):freeze_points(config)


if __name__=='__main__':unittest.main()
