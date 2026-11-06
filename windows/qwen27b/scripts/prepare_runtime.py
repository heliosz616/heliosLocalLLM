"""Assemble official, same-release Vulkan and CUDA backends with verified assets."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
TAG = 'b11429'

def main():
    r = requests.get(f'https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{TAG}', timeout=60)
    r.raise_for_status()
    release = r.json()
    names = [f'llama-{TAG}-bin-win-vulkan-x64.zip', f'llama-{TAG}-bin-win-cuda-13.4-x64.zip', 'cudart-llama-bin-win-cuda-13.4-x64.zip']
    assets = [next(a for a in release['assets'] if a['name'] == n) for n in names]
    downloads = ROOT / 'vendor' / 'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    def fetch(a):
        path = downloads / a['name']
        expected = a.get('digest', '').removeprefix('sha256:')
        if not path.exists():
            with requests.get(a['browser_download_url'], stream=True, timeout=(30,120)) as response:
                response.raise_for_status()
                with path.with_suffix('.partial').open('wb') as f:
                    for chunk in response.iter_content(4*1024*1024):
                        f.write(chunk)
            path.with_suffix('.partial').replace(path)
        actual = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        if expected and expected != actual:
            raise RuntimeError(f'Asset checksum mismatch: {path.name}')
        out = downloads / path.stem
        with zipfile.ZipFile(path) as z:
            z.extractall(out)
        print(path.name, actual, flush=True)
        return out, dict(name=path.name, sha256=actual, published_sha256=expected)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, assets))
    runtime = ROOT / 'build' / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    # Common ABI and Vulkan implementation come from the same tagged build.
    for f in results[0][0].rglob('*'):
        if f.is_file() and f.suffix in ('.dll', '.exe'):
            shutil.copy2(f, runtime / f.name)
    for folder, _ in results[1:]:
        for f in folder.rglob('*.dll'):
            if f.name.startswith(('ggml-cuda', 'cublas', 'cudart', 'nvrtc', 'nvJit')):
                shutil.copy2(f, runtime / f.name)
    (ROOT/'reports'/'runtime-assets.json').write_text(json.dumps(dict(tag=TAG, assets=[x[1] for x in results]),indent=2))

if __name__ == '__main__':
    main()
