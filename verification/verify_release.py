"""Check public release artifact bytes and portable archive members, without inference."""
import argparse
import hashlib
import re
import tarfile
from urllib.parse import unquote, urlsplit
import json
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.archive import Bank, digest, PHASES
from verification.schedules import need


def checked_path(root, name):
    path = PurePosixPath(name)
    need(not path.is_absolute() and '..' not in path.parts and '\\' not in name,
         f'Unsafe release path: {name}')
    return root.joinpath(*path.parts)


def verify_sources(root, models):
    configs = json.loads((root / 'experiments/configs/manifest.json').read_text())['models']
    need(set(configs) == set(models), 'Configuration/model coverage differs')
    for name, meta in configs.items():
        need(digest(checked_path(root, meta['path'])) == meta['sha256'],
             f'Configuration hash mismatch: {name}')
        need(meta['source_checkpoint_sha256'] == models[name]['source_sha256'],
             f'Configuration/checkpoint identity differs: {name}')
    sources = json.loads((root / 'experiments/historical_sources.json').read_text())
    meta = sources['archive']
    archive = checked_path(root, meta['path'])
    need(archive.stat().st_size == meta['bytes'] and digest(archive) == meta['sha256'],
         'Historical source archive hash/size mismatch')
    expected = {item['member']: item for version in sources['versions'].values()
                for item in version['files'].values()}
    found = set()
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream:
            checked_path(root, member.name)
            need(member.isfile() and member.name in expected and member.name not in found,
                 f'Unexpected historical source member: {member.name}')
            item = expected[member.name]
            need(member.size == item['bytes'], f'Source size mismatch: {member.name}')
            payload = stream.extractfile(member).read()
            need(hashlib.sha256(payload).hexdigest() == item['sha256'],
                 f'Source hash mismatch: {member.name}')
            found.add(member.name)
    need(found == set(expected) and len(found) == meta['member_count'],
         'Historical source coverage differs')
    print(f'{len(configs)} configurations and {len(found)} historical sources verified', flush=True)


def verify_links(root):
    for document in root.rglob('*.md'):
        if any(part.startswith('.') for part in document.relative_to(root).parts):
            continue
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', document.read_text()):
            url = urlsplit(target)
            if url.scheme or not url.path:
                continue
            need((document.parent / unquote(url.path)).exists(),
                 f'Broken link in {document.relative_to(root)}: {target}')
    print('Local Markdown links verified', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    models = json.loads((args.root / 'models/manifest.json').read_text())['models']
    verify_sources(args.root, models)
    verify_links(args.root)
    identities = set()
    for name, model in models.items():
        path = PurePosixPath(model['path'])
        need(not path.is_absolute() and '..' not in path.parts and '\\' not in str(path), 'Unsafe model path')
        need(digest(args.root.joinpath(*path.parts)) == model['sha256'], f'Model hash mismatch: {name}')
        identities.add(model['source_sha256'])
    members = 0
    for phase in PHASES:
        bank = Bank(args.root, phase)
        try:
            for case in bank.index['cases']:
                record = bank.json(case['member'])
                need(record['fixture_sha256'] == bank.index['fixtures'][case['fixture']]['source_sha256'],
                     'Case fixture source hash differs from index')
                identity = case.get('checkpoint_source_sha256')
                need(record.get('checkpoint_sha256') == identity, 'Case checkpoint source hash differs from index')
                need(identity is None or identity in identities, 'Case checkpoint absent from model manifest')
            for run in bank.index.get('runs', []):
                need(run['checkpoint_source_sha256'] in identities, 'Training checkpoint absent from manifest')
            members += bank.meta['member_count']
            print(f'{phase}: archive/member hashes verified', flush=True)
        finally:
            bank.close()
    print(f'Verified {len(models)} checkpoint files and {members} archive members. '
          'Run analysis.reproduce to recompute schedules and statistics.', flush=True)


if __name__ == '__main__':
    main()
