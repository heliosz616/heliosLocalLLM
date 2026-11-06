"""Pinned, resumable download; only promote a GGUF after full SHA256 verification."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = 'bartowski/ukisai_Swift-Qwen3.8-27b-GGUF'
NAME = 'ukisai_Swift-Qwen3.8-27b-IQ4_XS.gguf'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default=REPO)
    parser.add_argument('--filename', default=NAME)
    parser.add_argument('--manifest', default='manifest.json')
    args = parser.parse_args()
    repo, name = args.repo, args.filename
    if Path(name).name != name or Path(args.manifest).name != args.manifest:
        raise ValueError('Filename and manifest must be plain filenames')
    dest = ROOT / 'models'
    dest.mkdir(exist_ok=True)
    manifest = dest / args.manifest
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        if meta['repo'] != repo or meta['filename'] != name:
            raise ValueError('Manifest belongs to a different model; use a distinct --manifest')
    else:
        r = requests.get(f'https://huggingface.co/api/models/{repo}?blobs=true', timeout=60)
        r.raise_for_status()
        info = r.json()
        file = next(x for x in info['siblings'] if x['rfilename'] == name)
        meta = dict(repo=repo, revision=info['sha'], filename=name,
                    size=file['size'], sha256=file['lfs']['sha256'])
        manifest.write_text(json.dumps(meta, indent=2))
    final = dest / name
    part = dest / (name + '.partial')
    url = f'https://huggingface.co/{repo}/resolve/{meta["revision"]}/{name}'
    if not final.exists():
        for attempt in range(30):
            offset = part.stat().st_size if part.exists() else 0
            if offset == meta['size']:
                break
            try:
                with requests.get(url, headers={'Range': f'bytes={offset}-'}, stream=True, timeout=(30, 90)) as r:
                    r.raise_for_status()
                    if offset and (r.status_code != 206 or not r.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                        raise RuntimeError('Server did not honor resume offset')
                    with part.open('ab' if offset else 'wb') as f:
                        last = time.monotonic()
                        for chunk in r.iter_content(8 * 1024 * 1024):
                            f.write(chunk)
                            offset += len(chunk)
                            if time.monotonic() - last > 20:
                                print(f'{offset}/{meta["size"]} ({offset/meta["size"]:.1%})', flush=True)
                                last = time.monotonic()
                        f.flush()
                        os.fsync(f.fileno())
            except requests.RequestException as exc:
                print(f'Retry {attempt}: {type(exc).__name__}', flush=True)
                time.sleep(min(30, 2 + attempt))
        candidate = part
    else:
        candidate = final
    if candidate.stat().st_size != meta['size']:
        raise RuntimeError('Incomplete download; rerun to resume')
    digest = hashlib.file_digest(candidate.open('rb'), 'sha256').hexdigest()
    if digest != meta['sha256']:
        raise RuntimeError('SHA256 mismatch; file retained for diagnosis, not promoted')
    if candidate != final:
        os.replace(candidate, final)
    meta['verified_sha256'] = digest
    manifest.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)

if __name__ == '__main__':
    main()
