"""The simple public command must select Co-DINO's runtime without recursion."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import yaml
from utils.tracking.common import ROOT
from utils.vehicle_reid.pipeline import ensure_codino_runtime
from utils.vehicle_reid.models import Detector


class RuntimeTests(unittest.TestCase):
    def test_current_default_is_codino(self):
        config=yaml.safe_load((ROOT/'configs/vehicle_reid.yaml').read_text())
        self.assertEqual(config['detector']['backend'],'codino')
        self.assertEqual(config['detector']['checkpoint'],'checkpoints/teacher/co_dino_joint.best.pt')
        self.assertEqual(config['device'],'auto')

    def test_removed_backend_fails_before_loading_weights(self):
        with patch('torch.load') as load:
            with self.assertRaisesRegex(ValueError,'codino or cache'):
                Detector(dict(backend='maskrcnn'),torch.device('cpu'))
            load.assert_not_called()

    def test_handoff_preserves_arguments_without_shell_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            environment=Path(directory)
            (environment/'bin').mkdir()
            (environment/'bin/python').touch()
            with patch('utils.teacher.settings.read',return_value=({'environment':str(environment)},ROOT)), \
                 patch('utils.vehicle_reid.pipeline.sys.prefix','/another/environment'), \
                 patch('utils.vehicle_reid.pipeline.subprocess.run') as run:
                args=['--output','data/videos/a space','--resume']
                self.assertTrue(ensure_codino_runtime('codino',args))
                run.assert_called_once_with([str(environment/'bin/python'),str(ROOT/'demo.py'),'vehicle',*args],cwd=ROOT,check=True)

    def test_teacher_and_cache_do_not_relaunch(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('utils.teacher.settings.read',return_value=({'environment':directory},ROOT)), \
                 patch('utils.vehicle_reid.pipeline.sys.prefix',directory), \
                 patch('utils.vehicle_reid.pipeline.subprocess.run') as run:
                self.assertFalse(ensure_codino_runtime('codino',[]))
                self.assertFalse(ensure_codino_runtime('cache',[]))
                run.assert_not_called()


if __name__=='__main__':
    unittest.main()
