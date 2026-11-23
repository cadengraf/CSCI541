"""Exercise crop-bank alignment/resume with tiny images and a deterministic fake model."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import torch
from PIL import Image
import yaml

from utils.experiments.joint_evaluation import command, recovery
from utils.tracking.common import load, save


class TinyEncoder(torch.nn.Module):
    calls = 0

    def __init__(self, *args, **kwargs):
        super().__init__()

    def crop_embeddings(self, images):
        type(self).calls += 1
        return torch.stack([torch.stack([image.mean(), image.new_tensor(1.)]) for image in images])


class JointRecoveryTests(unittest.TestCase):
    def test_crop_alignment_empty_frames_and_completed_bank_reuse(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'out').mkdir()
            save({'fixture': True}, root/'out/selection.pt')
            image = root/'frame.jpg'
            Image.new('RGB', (10, 10), (120, 80, 40)).save(image)
            frames = [dict(path=str(image), observations=[dict(box=[0,0,5,5]),dict(box=[5,5,10,10])]),
                      dict(path=str(image), observations=[])]
            selection = dict(cases=dict(example=dict(frames=frames)))
            settings = dict(model=dict(embedding_dim=2), training=dict(mixed_precision=False))
            checkpoint = root/'checkpoint.pt'
            save(dict(model={}, identity_to_label={1:0}, vrai_identity_to_label={2:1}), checkpoint)
            config = dict(output='out', state_root='state', checkpoint_frames=1)
            TinyEncoder.calls = 0
            with patch.object(recovery, 'ROOT', root), patch('utils.teacher.model.CoDINOJoint', TinyEncoder):
                recovery.extract(config, selection, settings, checkpoint, torch.device('cpu'))
            state = load(root/'state/features/example/state.pt')
            self.assertTrue(state['complete'])
            first = load(root/'state/features/example/chunk_0000.pt')[0]
            empty = load(root/'state/features/example/chunk_0001.pt')[0]
            self.assertEqual(first.shape, (2,2))
            self.assertEqual(empty.shape, (0,2))
            self.assertTrue(torch.allclose(first.norm(dim=1), torch.ones(2)))
            self.assertEqual(TinyEncoder.calls, 1)
            with patch.object(recovery, 'ROOT', root), patch('utils.teacher.model.CoDINOJoint', side_effect=AssertionError('must reuse')):
                recovery.extract(config, selection, settings, checkpoint, torch.device('cpu'))
            # Simulate interruption after the first committed frame.
            state.update(complete=False, frames=1, parts=state['parts'][:1])
            save(state, root/'state/features/example/state.pt')
            with patch.object(recovery, 'ROOT', root), patch('utils.teacher.model.CoDINOJoint', TinyEncoder):
                recovery.extract(config, selection, settings, checkpoint, torch.device('cpu'))
            self.assertEqual(TinyEncoder.calls, 1)  # already-committed nonempty frame was not encoded again
            self.assertTrue(load(root/'state/features/example/state.pt')['complete'])

    def test_evaluation_config_preserves_saved_model_and_training_recipe(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original = dict(model=dict(crop_geometry='square',embedding_dim=256),
                            training=dict(epochs=4,learning_rate=1e-5), tracking=dict(name='training'))
            save(dict(settings=original), root/'best.pt')
            spec = dict(checkpoint='best.pt',last_checkpoint='last.pt',device='auto',
                        output='new/results',predictions_root='new/predictions',tracking_name='evaluation')
            (root/'eval.yaml').write_text(yaml.safe_dump(spec))
            with patch.object(command, 'ROOT', root):
                settings = command.checkpoint_settings('eval.yaml')
            self.assertEqual(settings['model'], original['model'])
            self.assertEqual(settings['training']['learning_rate'], original['training']['learning_rate'])
            self.assertEqual(settings['tracking']['metrics_root'], 'new/results/kitti_mots')
            self.assertEqual(load(root/'best.pt')['settings'], original)


if __name__ == '__main__':
    unittest.main()
