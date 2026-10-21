"""Replace Git LFS pointer files with their content: python deploy/fetch_lfs.py PATH [PATH ...]

Some build services clone repositories without Git LFS, leaving a small pointer file where
the catalog or index should be. For each path that is still a pointer, this asks GitHub's LFS
batch API (public repositories need no credentials) for the object, downloads it and accepts
it only if its SHA-256 and size match the pointer. Files that are already real are untouched.
Repositories are tried in order from LFS_REPOS (space-separated owner/name).
"""
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

REPOS = os.getenv('LFS_REPOS', 'Akshatb2006/Polaris-Hackathon-Base-MCP '
                  'SpringWinter-Research/Polaris-Hackathon-Base-MCP').split()
POINTER_PREFIX = b'version https://git-lfs.github.com/spec/v1'


def read_pointer(path):
    data = path.read_bytes()[:512]
    if not data.startswith(POINTER_PREFIX):
        return None
    fields = dict(line.split(' ', 1) for line in data.decode().splitlines() if ' ' in line)
    return fields['oid'].removeprefix('sha256:'), int(fields['size'])


def download_url(repo, oid, size):
    body = json.dumps({'operation': 'download', 'transfers': ['basic'],
                       'objects': [{'oid': oid, 'size': size}]}).encode()
    req = urllib.request.Request(f'https://github.com/{repo}.git/info/lfs/objects/batch', data=body, headers={
        'Accept': 'application/vnd.git-lfs+json', 'Content-Type': 'application/vnd.git-lfs+json'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        obj = json.load(resp)['objects'][0]
    return obj.get('actions', {}).get('download', {}).get('href')


def fetch(path):
    pointer = read_pointer(path)
    if pointer is None:
        print(f'{path}: already present ({path.stat().st_size} bytes)')
        return
    oid, size = pointer
    for repo in REPOS:
        try:
            href = download_url(repo, oid, size)
        except OSError as exc:
            print(f'{path}: {repo} lookup failed ({exc})')
            continue
        if not href:
            continue
        tmp = path.with_suffix(path.suffix + '.part')
        digest = hashlib.sha256()
        with urllib.request.urlopen(href, timeout=600) as resp, tmp.open('wb') as out:
            while chunk := resp.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() == oid and tmp.stat().st_size == size:
            tmp.replace(path)
            print(f'{path}: fetched {size} bytes from {repo}, sha256 verified')
            return
        tmp.unlink()
        print(f'{path}: checksum mismatch from {repo}; trying next')
    sys.exit(f'{path}: could not fetch LFS object {oid}')


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        fetch(Path(arg))
