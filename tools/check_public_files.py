"""Read-only check of files Git would publish; never prints matched secrets.

Run before git init (uses a temporary index) or after staging. No Git history scan.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, stderr=subprocess.PIPE)


@contextmanager
def public_files():
    if (ROOT / '.git').exists():
        yield git('ls-files', '-z', '--cached', '--others', '--exclude-standard')
    else:
        with TemporaryDirectory(prefix='dietitian-public-check-') as directory:
            git('init', '--quiet', directory)
            yield git('--git-dir=' + str(Path(directory) / '.git'), '--work-tree=' + str(ROOT),
                      'ls-files', '-z', '--cached', '--others', '--exclude-standard')


def local_secrets():
    values = set()
    # Check against actual local credentials without displaying any values.
    for path in (ROOT / '.env', ROOT / 'dietitian/.env'):
        if not path.is_file():
            continue
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            match = re.match(r'\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=\s*(.*?)\s*$', line)
            if not match:
                continue
            name, value = match.groups()
            value = value.strip('\"\'')
            if re.search(r'KEY|SECRET|TOKEN|PASSWORD|BUCKET|APP_ID', name) and len(value) >= 8:
                values.add(value)
            elif name.endswith('_URI') and re.search(r'://[^/]+:[^/]+@', value):
                values.add(value)
    return values


def main():
    try:
        with public_files() as output:
            files = sorted({name.decode('utf-8') for name in output.split(b'\0') if name})
    except (OSError, subprocess.CalledProcessError):
        print('FAIL: Git file enumeration failed. Install Git and run from a local checkout.')
        return 2
    secrets = local_secrets()
    staged = set()
    if (ROOT / '.git').exists():
        try:
            staged = {name.decode('utf-8') for name in git('ls-files', '-z', '--cached').split(b'\0') if name}
        except subprocess.CalledProcessError:
            print('FAIL: Could not read staged file list.')
            return 2
    patterns = {
        'credential-like token': re.compile(r'\b(?:sk-[A-Za-z0-9_-]{20,}|tvly-[A-Za-z0-9_-]{20,}|AKID[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b'),
        'private key material': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
        'URL containing credentials': re.compile(r'[a-z][a-z0-9+.-]*://[^\s/\"\']+:[^\s/\"\']+@', re.I),
        'personal absolute path': re.compile(r'(?:[A-Z]:[\\/]Users[\\/](?!Public\b|<)[^\\/\s\"\']+|/(?:Users|home)/(?!<)[^/\s\"\']+)', re.I),
        'COS account-specific hostname': re.compile(r'\b[a-z0-9-]+-\d{8,}\.cos\.[a-z0-9-]+\.myqcloud\.com\b'),
    }
    findings = []
    binary = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.ico', '.woff', '.woff2', '.ttf'}
    for filename in files:
        path = ROOT / filename
        parts = set(Path(filename).parts)
        basename = path.name
        forbidden = (basename == '.env' or basename.startswith('.env.') and basename not in {'.env.example', '.env.template'}
                     or basename.endswith('.env') or bool(parts & {'.runtime', '.idea', '.aws', '.ssh', 'uploads', 'backups'})
                     or basename == 'credentials.json' or basename.startswith('service-account') and basename.endswith('.json')
                     or filename.startswith('dietitian/knowledge/documents/')
                     or filename == 'dietitian/knowledge/sources.json'
                     or path.suffix.lower() in {'.pem', '.key', '.p12', '.pfx', '.db', '.sqlite', '.sqlite3', '.log'})
        if forbidden:
            findings.append((filename, 0, 'private/runtime file included in Git candidate set'))
            continue
        if path.is_symlink():
            findings.append((filename, 0, 'symlink requires manual review'))
            continue
        if path.suffix.lower() in binary:
            continue  # Image pixels/metadata and other binary content require manual review.
        versions = []
        if path.is_file():
            if path.stat().st_size > 2_000_000:
                findings.append((filename, 0, 'large file requires manual review'))
            else:
                versions.append((filename, path.read_bytes()))
        if filename in staged:
            try:
                if int(git('cat-file', '-s', ':' + filename)) > 2_000_000:
                    findings.append((filename + ' [staged]', 0, 'large file requires manual review'))
                else:
                    versions.append((filename + ' [staged]', git('show', ':' + filename)))
            except subprocess.CalledProcessError:
                findings.append((filename, 0, 'could not inspect staged content'))
        for location, data in versions:
            try:
                text = data.decode('utf-8-sig')
            except UnicodeError:
                findings.append((location, 0, 'non-text file requires manual review'))
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if any(secret in line for secret in secrets):
                    findings.append((location, number, 'matches a local credential/account identifier'))
                # This exact invalid-endpoint fixture contains no real credentials.
                checked_line = line.replace('https://user:secret@example.org/mcp', '[synthetic test URL]')
                for label, pattern in patterns.items():
                    if pattern.search(checked_line):
                        findings.append((location, number, label))
    for filename, number, label in findings:
        print(f'REVIEW {filename}:{number}: {label} (value hidden)')
    print(f'Checked {len(files)} Git candidate files; {len(findings)} finding(s).')
    print('Scope: working-tree and staged text. Review history, images and private knowledge separately.')
    return 1 if findings else 0


if __name__ == '__main__':
    sys.exit(main())
