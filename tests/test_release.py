"""Offline portability and privacy checks for the source distribution."""

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_python_sources_parse(self):
        for directory in ('lightrag', 'eve_lightrag'):
            for path in (ROOT / directory).rglob('*.py'):
                with self.subTest(path=path.relative_to(ROOT)):
                    ast.parse(path.read_text(encoding='utf-8'))

    def test_no_absolute_home_paths(self):
        for directory in ('lightrag', 'eve_lightrag'):
            for path in (ROOT / directory).rglob('*.py'):
                self.assertNotIn('/home/', path.read_text(encoding='utf-8'), str(path))

    def test_integration_exports(self):
        from lightrag.api import eve_fittings
        self.assertTrue(callable(eve_fittings.simulate_fitting))
        self.assertTrue(callable(eve_fittings.get_character_skill_data))
        self.assertEqual(eve_fittings.simulate_fitting.__module__, 'eve_lightrag.pyfa_skill')

    def test_binding_template_is_packaged(self):
        self.assertTrue((ROOT / 'eve_lightrag/eve_bind.html').is_file())


if __name__ == '__main__':
    unittest.main()
