"""Verify evaluation-only routing without loading a model or touching real outputs."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from utils.experiments.joint_evaluation import command


class JointEvaluationTests(unittest.TestCase):
    def test_existing_crop_settings_are_used_on_final_partition(self):
        settings = {'saved': 'settings'}
        calls = []
        modules = {
            'utils.experiments.crop_training.settings': SimpleNamespace(
                read=lambda arm: (settings, command.ROOT) if arm == 'square_256' else self.fail(arm)),
            'utils.teacher.evaluation': SimpleNamespace(run=lambda **kwargs: calls.append(kwargs)),
        }
        with patch.dict(sys.modules, modules):
            command.evaluate_selected()
        self.assertEqual(calls, [dict(experiment=settings, calibration_only=False)])

    def test_launches_only_isolated_evaluator_then_refreshes_report(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'configs').mkdir()
            (root/'configs/teacher.yaml').write_text('environment: environment\n')
            python = root/'environment/bin/python'
            python.parent.mkdir(parents=True)
            python.touch()
            checkpoints = root/'checkpoints/crop_training'
            checkpoints.mkdir(parents=True)
            for name in ['square_256.best.pt', 'square_256.last.pt']:
                (checkpoints/name).touch()
            refresh = []
            with patch.object(command, 'ROOT', root), patch.object(command.subprocess, 'run') as launch, patch.dict(
                sys.modules, {'utils.reporting.milestones': SimpleNamespace(run=lambda: refresh.append(True))}
            ):
                command.main([])
            self.assertEqual(launch.call_args.args[0], [str(python), '-m',
                'utils.experiments.joint_evaluation.command', '--worker'])
            self.assertTrue(launch.call_args.kwargs['check'])
            self.assertEqual(refresh, [True])

    def test_failed_evaluation_does_not_refresh_report(self):
        import subprocess
        with patch.object(command.Path, 'is_file', return_value=True), patch.object(
            command.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'evaluation')
        ), patch('utils.reporting.milestones.run') as refresh:
            with self.assertRaises(subprocess.CalledProcessError):
                command.main([])
            refresh.assert_not_called()


if __name__ == '__main__':
    unittest.main()
