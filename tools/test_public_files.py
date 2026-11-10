"""Exercise publication checks against disposable real Git indexes."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import check_public_files as audit


class PublicationCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / '.gitignore').write_text((audit.ROOT / '.gitignore').read_text(), encoding='utf-8')
        self.secret = 'sk-' + 'synthetic_only_1234567890' * 2
        (self.root / 'dietitian').mkdir()
        (self.root / 'dietitian/.env').write_text('DASHSCOPE_API_KEY=' + self.secret, encoding='utf-8')
        subprocess.run(['git', 'init', '--quiet', str(self.root)], check=True, capture_output=True)

    def check(self):
        output = StringIO()
        with patch.object(audit, 'ROOT', self.root), redirect_stdout(output):
            result = audit.main()
        self.assertNotIn(self.secret, output.getvalue())
        return result, output.getvalue()

    def test_ignored_private_files_are_not_candidates(self):
        for relative in ('.runtime/chat.log', 'dietitian/knowledge/documents/private.md',
                         'dietitian/knowledge/sources.json', 'dietitian/.idea/workspace.xml'):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.secret, encoding='utf-8')
        (self.root / 'README.md').write_text('Public project description', encoding='utf-8')
        self.assertEqual(self.check()[0], 0)

    def test_staged_secret_detected_after_working_copy_is_cleaned(self):
        path = self.root / 'settings.py'
        path.write_text('KEY=' + repr(self.secret), encoding='utf-8')
        subprocess.run(['git', '-C', str(self.root), 'add', 'settings.py'], check=True, capture_output=True)
        path.write_text('KEY=None', encoding='utf-8')
        result, output = self.check()
        self.assertEqual(result, 1)
        self.assertIn('[staged]', output)

    def test_force_added_env_is_rejected(self):
        subprocess.run(['git', '-C', str(self.root), 'add', '-f', 'dietitian/.env'],
                       check=True, capture_output=True)
        result, output = self.check()
        self.assertEqual(result, 1)
        self.assertIn('private/runtime file', output)


if __name__ == '__main__':
    unittest.main()
