import asyncio
import copy
import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from model_profiles import load_catalog, atomic_json, prune_shared
import gateway as g


def test_catalog_rejects_shared_cache_and_paths(tmp_path):
    cfg = copy.deepcopy(g.CONFIG)
    profiles={cfg['model']:dict(config=cfg,manifest='manifest.json',slots='slots'),
        'other':dict(config=dict(cfg,model='other'),manifest='other.json',slots='other')}
    path=tmp_path/'catalog.json'
    atomic_json(path,dict(models=profiles))
    assert len(load_catalog(tmp_path,cfg,'catalog.json'))==2
    profiles['other']['slots']='slots'
    atomic_json(path,dict(models=profiles))
    with pytest.raises(ValueError,match='separate'):
        load_catalog(tmp_path,cfg,'catalog.json')
    profiles['other']['slots']='../escape'
    atomic_json(path,dict(models=profiles))
    with pytest.raises(ValueError,match='plain'):
        load_catalog(tmp_path,cfg,'catalog.json')


def test_switch_same_model_fast_path(tmp_path,monkeypatch):
    (tmp_path/'state').mkdir()
    atomic_json(tmp_path/'state/runtime-status.json',dict(model='a',status='ready'))
    monkeypatch.setattr(g,'ROOT',tmp_path)
    monkeypatch.setattr(g,'CATALOG_FILE','catalog.json')
    monkeypatch.setattr(g,'active_model','a')
    monkeypatch.setattr(g,'healthy',True)
    assert asyncio.run(g.ensure_model('a'))==0
    assert not (tmp_path/'state/model-switch-request.json').exists()


@pytest.mark.parametrize('ok',[True,False])
def test_switch_and_failed_load_remain_recoverable(tmp_path,monkeypatch,ok):
    (tmp_path/'state').mkdir()
    atomic_json(tmp_path/'state/runtime-status.json',dict(model='a',status='ready'))
    monkeypatch.setattr(g,'ROOT',tmp_path)
    monkeypatch.setattr(g,'CATALOG_FILE','catalog.json')
    monkeypatch.setattr(g,'active_model','a')
    monkeypatch.setattr(g,'resident',('old','snapshot','digest'))
    monkeypatch.setattr(g,'prepared_prompt',('old',))
    monkeypatch.setattr(g,'switching',False)
    activated=[]
    async def activate(name): activated.append(name)
    monkeypatch.setattr(g,'activate_profile',activate)
    async def exercise():
        task=asyncio.create_task(g.ensure_model('b'))
        path=tmp_path/'state/model-switch-request.json'
        while not path.exists(): await asyncio.sleep(.01)
        assert g.switching and g.resident is None and g.prepared_prompt is None
        request=json.loads(path.read_text())
        atomic_json(tmp_path/'state/model-switch-result.json',dict(id='stale',ok=True))
        await asyncio.sleep(.12)
        assert not task.done()
        atomic_json(tmp_path/'state/model-switch-result.json',dict(id=request['id'],ok=ok,seconds=2))
        if ok:
            assert await task==2
        else:
            with pytest.raises(HTTPException) as error: await task
            assert error.value.status_code==503
        assert not g.switching
    asyncio.run(exercise())
    assert activated==(['b'] if ok else [])


def test_cancelled_switch_waits_for_runtime_completion(tmp_path,monkeypatch):
    (tmp_path/'state').mkdir()
    atomic_json(tmp_path/'state/runtime-status.json',dict(model='a',status='ready'))
    for key,value in dict(ROOT=tmp_path,CATALOG_FILE='catalog.json',active_model='a',resident=None,
                          prepared_prompt=None,switching=False).items():
        monkeypatch.setattr(g,key,value)
    async def activate(name): pass
    monkeypatch.setattr(g,'activate_profile',activate)
    async def exercise():
        task=asyncio.create_task(g.ensure_model('b'))
        path=tmp_path/'state/model-switch-request.json'
        while not path.exists(): await asyncio.sleep(.01)
        task.cancel()
        await asyncio.sleep(.12)
        assert not task.done() and g.switching
        request=json.loads(path.read_text())
        atomic_json(tmp_path/'state/model-switch-result.json',dict(id=request['id'],ok=True,seconds=1))
        with pytest.raises(asyncio.CancelledError): await task
        assert not g.switching
    asyncio.run(exercise())


def test_shared_quota_protects_model_and_session_pair():
    import time
    class FakeStore:
        config={'session_ttl_seconds':3600}
        def __init__(self,size): self.rows=[('same-session',size,time.time())]
        def entries(self): return self.rows
        def delete(self,sid): self.rows=[]
    stores={'a':FakeStore(60),'b':FakeStore(60)}
    prune_shared(stores,100,30,('a','same-session'))
    assert stores['a'].rows and not stores['b'].rows
    with pytest.raises(ValueError,match='previous snapshot preserved'):
        prune_shared(stores,100,50,('a','same-session'))
    assert stores['a'].rows
