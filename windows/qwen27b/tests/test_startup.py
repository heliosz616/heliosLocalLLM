"""Exercise real gateway lifespan with isolated files and a simulated engine."""
import asyncio
import copy
import json

import httpx
import pytest

import gateway as g
from model_profiles import ensure_catalog, load_catalog, atomic_json


def test_catalog_upgrade_preserves_single_model_storage(tmp_path):
    cfg = copy.deepcopy(g.CONFIG)
    filename = ensure_catalog(tmp_path, cfg)
    profiles = load_catalog(tmp_path, cfg, filename)
    assert profiles[cfg['model']]['manifest'] == 'manifest.json'
    assert profiles[cfg['model']]['slots'] == 'slots'
    profiles[cfg['model']]['slots'] = 'custom-slots'
    atomic_json(tmp_path/filename, {'models': profiles})
    before = (tmp_path/filename).read_bytes()
    ensure_catalog(tmp_path, cfg)
    assert (tmp_path/filename).read_bytes() == before


@pytest.mark.parametrize('catalog_mode', [False, True])
def test_gateway_full_lifespan_with_published_patch(tmp_path, monkeypatch, catalog_mode):
    cfg = copy.deepcopy(g.CONFIG)
    cfg['model'] = 'swift-qwen3.8-27b'
    for directory in ('state', 'models', 'reports', 'patches'):
        (tmp_path/directory).mkdir()
    (tmp_path/'state/internal-key.txt').write_text('synthetic-internal-test-key')
    atomic_json(tmp_path/'models/manifest.json', {'sha256':'synthetic-model', 'verified_sha256':'synthetic-model'})
    atomic_json(tmp_path/'reports/runtime-assets.json', {})
    atomic_json(tmp_path/'reports'/cfg['runtime_manifest'], {})
    # Use precisely the artifact shipped in the repository, not the obsolete file.
    (tmp_path/'patches/runtime.patch').write_bytes((g.ROOT/'patches/runtime.patch').read_bytes())
    filename = ensure_catalog(tmp_path, cfg) if catalog_mode else None
    profiles = load_catalog(tmp_path, cfg, filename)
    atomic_json(tmp_path/'state/runtime-status.json', {'model':cfg['model'], 'status':'ready'})
    for key, value in dict(ROOT=tmp_path, CONFIG=cfg, MODEL=cfg['model'], PROFILES=profiles,
                           CATALOG_FILE=filename, switching=False, healthy=True, waiting=0,
                           resident=None, prepared_prompt=None, version=None, client=None,
                           stores={}, store=None, active_model=None).items():
        monkeypatch.setattr(g, key, value)
    real_client = httpx.AsyncClient
    calls = []
    async def engine(request):
        calls.append(request.url.path)
        if request.url.path == '/props': return httpx.Response(200, json={'chat_template':'synthetic'})
        if request.url.path == '/slots': return httpx.Response(200, json=[{'is_processing':False}])
        return httpx.Response(200, json={})
    monkeypatch.setattr(g.httpx, 'AsyncClient', lambda **kwargs: real_client(
        transport=httpx.MockTransport(engine), **kwargs))
    async def exercise():
        monkeypatch.setattr(g, 'lock', asyncio.Lock())
        async with g.lifespan(g.app):
            assert g.version and len(g.version) == 64
            assert g.active_model == cfg['model']
            response = await g.health()
            body = response if isinstance(response, dict) else json.loads(response.body)
            assert body['status'] == 'ready'
            assert '/props' in calls and '/slots/0' in calls
    asyncio.run(exercise())


def test_generated_catalog_recovers_same_model_on_next_request(tmp_path, monkeypatch):
    cfg = copy.deepcopy(g.CONFIG)
    filename = ensure_catalog(tmp_path, cfg)
    (tmp_path/'state').mkdir()
    atomic_json(tmp_path/'state/runtime-status.json', {'model':None,'status':'unavailable'})
    for key, value in dict(ROOT=tmp_path, CATALOG_FILE=filename, active_model=None,
                           healthy=False, resident=None, prepared_prompt=None, switching=False).items():
        monkeypatch.setattr(g, key, value)
    activated = []
    async def activate(name): activated.append(name)
    monkeypatch.setattr(g, 'activate_profile', activate)
    async def exercise():
        async with asyncio.timeout(3):
            task = asyncio.create_task(g.ensure_model(cfg['model']))
            path = tmp_path/'state/model-switch-request.json'
            while not path.exists(): await asyncio.sleep(.01)
            request = json.loads(path.read_text())
            assert request['model'] == cfg['model']
            atomic_json(tmp_path/'state/model-switch-result.json',
                        {'id':request['id'], 'ok':True, 'seconds':.5})
            assert await task == .5
    asyncio.run(exercise())
    assert activated == [cfg['model']]
