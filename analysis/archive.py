"""Read checksum-pinned release members without trusting archive extraction paths."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

from verification.schedules import need

PHASES = ['static', 'dynamic', 'public_benchmark', 'return_variance', 'ablations']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Bank:
    def __init__(self, root, phase):
        manifest = json.loads((root / 'results/manifest.json').read_text())
        self.release_root = root
        self.meta = manifest['archives'][phase]
        archive = root / self.meta['path']
        need(digest(archive) == self.meta['sha256'], f'Archive hash mismatch: {phase}')
        self.scratch = tempfile.TemporaryDirectory(prefix='peach-paper-')
        self.root = Path(self.scratch.name)
        expected = self.meta['members']
        found = set()
        with tarfile.open(archive, 'r:gz') as stream:
            for member in stream:
                path = PurePosixPath(member.name)
                need(member.isfile() and not path.is_absolute() and '..' not in path.parts
                     and '\\' not in member.name and str(path) == member.name,
                     f'Unsafe archive member: {member.name}')
                need(member.name in expected and member.name not in found, 'Unexpected/duplicate member')
                meta = expected[member.name]
                need(member.size == meta['bytes'], 'Member size mismatch')
                source = stream.extractfile(member)
                target = self.root.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                with target.open('wb') as output:
                    while data := source.read(1024 * 1024):
                        h.update(data)
                        output.write(data)
                need(h.hexdigest() == meta['sha256'], f'Member hash mismatch: {member.name}')
                found.add(member.name)
        need(found == set(expected) and len(found) == self.meta['member_count'], 'Archive coverage mismatch')
        self.index = self.json('index.json')
        for name, meta in self.index.get('external_files', {}).items():
            path = PurePosixPath(name)
            need(not path.is_absolute() and '..' not in path.parts and '\\' not in name, 'Unsafe external path')
            target = root.joinpath(*path.parts)
            need(target.stat().st_size == meta['bytes'] and digest(target) == meta['sha256'],
                 f'External reference hash mismatch: {name}')

    def json(self, name):
        need(name in self.meta['members'], f'Unlisted member: {name}')
        return json.loads((self.root / name).read_text())

    def jsonl(self, name):
        need(name in self.meta['members'], f'Unlisted member: {name}')
        return [json.loads(line) for line in (self.root / name).read_text().splitlines() if line.strip()]

    def file(self, name):
        if name in self.meta['members']:
            return self.root / name
        need(name in self.index.get('external_files', {}), f'Unlisted external file: {name}')
        return self.release_root / name

    def close(self):
        self.scratch.cleanup()
