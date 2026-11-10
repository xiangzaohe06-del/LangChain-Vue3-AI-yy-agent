import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import yaml

from agents.skills import SkillRegistry, load_skill, skill_registry


class SkillsTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write('sample', 'Use this distinctive workflow.')

    def write(self, name, body, metadata=None):
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        (directory / 'SKILL.md').write_text(
            '---\n' + (metadata or f'name: {name}\ndescription: A sample task workflow.')
            + '\n---\n' + body, encoding='utf-8')

    def test_progressive_loading_and_idempotence(self):
        registry, loaded = SkillRegistry(self.root), {}
        self.assertNotIn('distinctive', registry.prompt(loaded))
        self.assertIn('sample', registry.prompt(loaded))
        self.assertTrue(registry.load('sample', loaded)['success'])
        self.assertIn('distinctive', registry.prompt(loaded))
        registry.load('sample', loaded)
        self.assertEqual(len(loaded), 1)
        self.assertNotIn('distinctive', registry.prompt({}))

    def test_unknown_and_path_names_cannot_read_files(self):
        registry = SkillRegistry(self.root)
        for name in ('../sample', 'sample/SKILL.md', str(self.root / 'sample'), 'unknown'):
            with self.assertRaises(ValueError):
                registry.load(name, {})

    def test_malformed_metadata_and_oversized_body_are_rejected(self):
        for metadata, body in [('name: other\ndescription: wrong', 'body'),
                               ('name: sample', 'body'),
                               ('name: sample\ndescription: valid', 'x' * 6001),
                               ('name: sample\ndescription: !!python/object:bad {}', 'body')]:
            self.write('sample', body, metadata)
            with self.assertRaises((ValueError, yaml.YAMLError)):
                SkillRegistry(self.root)

    def test_body_edit_is_loaded_but_metadata_changes_require_refresh(self):
        registry = SkillRegistry(self.root)
        self.write('sample', 'Updated workflow')
        loaded = {}
        registry.load('sample', loaded)
        self.assertEqual(loaded['sample'], 'Updated workflow')
        self.write('sample', 'body', 'name: sample\ndescription: Changed routing')
        with self.assertRaises(ValueError):
            registry.load('sample', {})

    def test_aggregate_budget_is_atomic(self):
        self.write('sample', 'a' * 5500)
        self.write('second', 'b' * 5500)
        registry, loaded = SkillRegistry(self.root), {}
        registry.load('sample', loaded)
        with self.assertRaises(ValueError):
            registry.load('second', loaded)
        self.assertEqual(list(loaded), ['sample'])

    def test_tool_loads_only_current_runtime_context(self):
        registry = SkillRegistry(self.root)
        first, second = SimpleNamespace(context={}), SimpleNamespace(context={})
        with patch('agents.skills.skill_registry', registry):
            result = json.loads(load_skill.func('sample', first))
            self.assertTrue(result['success'])
            self.assertIn('sample', first.context['loaded_skills'])
            self.assertEqual(second.context, {})
            self.assertFalse(json.loads(load_skill.func('../secret', second))['success'])
            self.assertFalse(json.loads(load_skill.func('sample', SimpleNamespace(context=None)))['success'])
        self.assertNotIn('runtime', load_skill.args)

    def test_bundled_skills_are_discoverable_and_loadable(self):
        self.assertEqual(set(skill_registry.skills), {'recipe-recommendation', 'weekly-meal-prep'})
        loaded = {}
        for name in skill_registry.skills:
            self.assertTrue(skill_registry.load(name, loaded)['success'])
        self.assertLess(sum(map(len, loaded.values())), skill_registry.max_loaded_chars)
