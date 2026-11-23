from types import ModuleType
import unittest
from utils.experiments.appearance_clip.model import belongs_to_source


class NamespaceTests(unittest.TestCase):
    def test_namespace_with_no_file_from_expected_source_is_allowed(self):
        module=ModuleType('model');module.__file__=None;module.__path__=['/tmp/clip/model']
        self.assertTrue(belongs_to_source(module,'/tmp/clip'))

    def test_regular_module_from_expected_source_is_allowed(self):
        module=ModuleType('config');module.__file__='/tmp/clip/config/__init__.py'
        self.assertTrue(belongs_to_source(module,'/tmp/clip'))

    def test_unrelated_or_mixed_namespace_remains_rejected(self):
        module=ModuleType('model');module.__path__=['/tmp/other/model']
        self.assertFalse(belongs_to_source(module,'/tmp/clip'))
        module.__path__=['/tmp/clip/model','/tmp/other/model']
        self.assertFalse(belongs_to_source(module,'/tmp/clip'))
        module.__path__=['/tmp/clip_extra/model']
        self.assertFalse(belongs_to_source(module,'/tmp/clip'))


if __name__=='__main__':unittest.main()
