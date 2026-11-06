"""Gateway logic only: mock backend results are never hardware acceptance evidence."""
import asyncio
import json
import os
from pathlib import Path
import sys
import httpx
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['LLM_SERVICE_CONFIG']=str(ROOT/'config.example.json')
from unittest.mock import patch
_read_text = Path.read_text
def _test_read_text(path, *args, **kwargs):
    if path == ROOT/'state'/'api-key.txt':
        return 'test-key-that-is-long-enough-for-unit-tests'
    return _read_text(path, *args, **kwargs)
with patch.object(Path, 'read_text', _test_read_text):
    import gateway as g
from store import Store

def run(case,tmp_path,truncate_stream=False,template_status=None):
    async def test():
        g.lock=asyncio.Lock(); g.waiting=0; g.healthy=True; g.version='test'; g.resident=None; g.prepared_prompt=None
        cfg=g.CONFIG.copy(); cfg.update(cache_max_bytes=10000,session_max_bytes=1000)
        g.store=Store(tmp_path,cfg)
        calls=[]; active=0; peak=0
        async def backend(req):
            nonlocal active,peak
            body=json.loads(req.content) if req.content else {}
            calls.append((req.url.path,dict(req.url.params),body))
            if req.url.path=='/slots':
                return httpx.Response(200,json=[{'is_processing':False}])
            if req.url.path=='/slots/0':
                action=req.url.params['action']
                if action=='save':
                    (tmp_path/body['filename']).write_bytes(b'native-checkpoint')
                return httpx.Response(200,json={'n_restored':10,'n_saved':10})
            if req.url.path=='/apply-template':
                if template_status:
                    return httpx.Response(template_status,json={'error':{'message':'Jinja Exception: invalid message' if template_status==500 else 'Backend unavailable'}})
                return httpx.Response(200,json={'prompt':'formatted'})
            if req.url.path=='/tokenize': return httpx.Response(200,json={'tokens':[1,2,3]})
            if req.url.path=='/v1/chat/completions':
                active+=1; peak=max(peak,active)
                await asyncio.sleep(0.02)
                active-=1
                r=dict(choices=[{'message':{'role':'assistant','content':'test'}}],usage={'prompt_tokens':3,'completion_tokens':1},timings={'cache_n':0,'prompt_n':3})
                if body.get('stream'):
                    ending='' if truncate_stream else 'data: [DONE]\n\n'
                    return httpx.Response(200,text='data: '+json.dumps(r)+'\n\n'+ending,headers={'content-type':'text/event-stream'})
                return httpx.Response(200,json=r)
            return httpx.Response(200,json={})
        g.client=httpx.AsyncClient(transport=httpx.MockTransport(backend),base_url='http://engine')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=g.app),base_url='http://gateway',headers={'Authorization':'Bearer '+g.KEY}) as client:
            await case(client,calls,lambda:peak)
        await g.client.aclose(); g.store.db.close()
    asyncio.run(test())

def test_serial_generation_and_session_snapshots(tmp_path):
    async def case(client,calls,peak):
        async def send(sid):
            return await client.post('/v1/chat/completions',json={'messages':[{'role':'user','content':'x'}],'session_id':sid})
        results=await asyncio.gather(send('A'),send('B'))
        assert all(r.status_code==200 for r in results)
        assert peak()==1
        assert g.store.get('A')['filename']!=g.store.get('B')['filename']
        r=await send('A')
        assert r.json()['session_cache']['restore_status']=='cache_miss_after_restore'
        assert any(params.get('action')=='restore' for _,params,_ in calls)
    run(case,tmp_path)

def test_stream_commit_and_stateless_isolation(tmp_path):
    async def case(client,calls,peak):
        r=await client.post('/v1/chat/completions',json={'messages':[],'stream':True,'session_id':'A'})
        assert r.status_code==200 and r.text.endswith('data: [DONE]\n\n')
        assert g.store.get('A')['filename']
        assert not g.lock.locked()
        calls.clear()
        r=await client.post('/v1/chat/completions',json={'messages':[]})
        assert r.status_code==200
        assert not any(params.get('action')=='restore' for _,params,_ in calls)
        assert r.json()['session_cache']['restore_status']=='stateless'
        assert g.resident is None
        assert any(params.get('action')=='erase' for _,params,_ in calls)
    run(case,tmp_path)

def test_auth_path_and_context_rejection(tmp_path):
    async def case(client,calls,peak):
        assert (await client.get('/health',headers={'Authorization':'wrong'})).status_code==401
        assert (await client.get('/slots')).status_code==404
        r=await client.post('/v1/chat/completions',json={'messages':[],'session_id':'../escape'})
        assert r.status_code==400
        r=await client.post('/v1/chat/completions',json={'messages':[],'max_tokens':999999,'stream':True})
        assert r.status_code==400 and not g.lock.locked()
        r=await client.post('/v1/chat/completions',json={'messages':[]})
        assert r.status_code==200
    run(case,tmp_path)

def test_truncated_backend_stream_preserves_previous_snapshot(tmp_path):
    async def case(client,calls,peak):
        assert (await client.post('/v1/chat/completions',json={'messages':[],'session_id':'A'})).status_code==200
        previous=g.store.get('A')['filename']
        calls.clear()
        response=await client.post('/v1/chat/completions',json={'messages':[],'stream':True,'session_id':'A'})
        assert 'server_error' in response.text
        assert g.store.get('A')['filename']==previous
        assert not any(params.get('action')=='save' for _,params,_ in calls)
        assert not g.lock.locked()
        assert g.resident is None
    run(case,tmp_path,truncate_stream=True)

def test_resident_reuse_still_checks_snapshot_integrity(tmp_path):
    async def case(client,calls,peak):
        async def send():
            return await client.post('/v1/chat/completions',json={'messages':[],'session_id':'A'})
        await send()
        calls.clear()
        response=await send()
        # The mock reports zero cached tokens, so this must not be called a hit.
        assert response.json()['session_cache']['restore_status']=='resident_cache_miss'
        assert response.json()['session_cache']['text_cache_match']=='exact'
        assert response.json()['session_cache']['preparation_cache_hit']
        assert not any(path in ('/apply-template','/tokenize') for path,_,_ in calls)
        assert not any(params.get('action') in ('restore','erase') for _,params,_ in calls)
        row=g.store.get('A')
        (tmp_path/row['filename']).write_bytes(b'broken-checkpoint')
        calls.clear()
        response=await send()
        assert response.json()['session_cache']['restore_status']=='restore_failed_integrity'
        assert any(params.get('action')=='erase' for _,params,_ in calls)
        assert not any(params.get('action')=='restore' for _,params,_ in calls)
    run(case,tmp_path)

def test_delete_resident_session_erases_native_state(tmp_path):
    async def case(client,calls,peak):
        await client.post('/v1/chat/completions',json={'messages':[],'session_id':'A'})
        assert g.resident is not None
        calls.clear()
        assert (await client.delete('/sessions/A')).status_code==200
        assert g.resident is None and g.store.get('A') is None
        assert any(params.get('action')=='erase' for _,params,_ in calls)
    run(case,tmp_path)

def test_resident_same_public_id_cannot_cross_principals(tmp_path,monkeypatch):
    original=g.principal
    tokens={'alice-test-token':'alice','bob-test-token':'bob'}
    monkeypatch.setattr(g,'principal',lambda token,*args:tokens.get(token) or original(token,*args))
    async def case(client,calls,peak):
        async def send(token):
            return await client.post('/v1/chat/completions',headers={'Authorization':'Bearer '+token},
                                     json={'messages':[],'session_id':'same-id'})
        assert (await send('alice-test-token')).status_code==200
        alice_resident=g.resident
        calls.clear()
        response=await send('bob-test-token')
        assert response.status_code==200
        assert response.json()['session_cache']['restore_status']=='cache_miss'
        assert g.resident[0]!=alice_resident[0]
        assert any(params.get('action')=='erase' for _,params,_ in calls)
        assert not any(params.get('action')=='restore' for _,params,_ in calls)
    run(case,tmp_path)

def test_off_lan_and_forwarded_ip_cannot_bypass_filter():
    async def case():
        transport=httpx.ASGITransport(app=g.app,client=('203.0.113.10',12345))
        async with httpx.AsyncClient(transport=transport,base_url='http://gateway') as c:
            response=await c.get('/health',headers={'Authorization':'Bearer '+g.KEY,'X-Forwarded-For':'192.0.2.4'})
            assert response.status_code==403
    asyncio.run(case())

def test_agent_defaults_overrides_and_sampler_alias(tmp_path):
    async def case(client,calls,peak):
        async def send(**fields):
            return await client.post('/v1/chat/completions',json={'messages':[],**fields})
        assert (await send(temperature=0.7)).status_code==200
        body=[b for p,_,b in calls if p=='/v1/chat/completions'][-1]
        assert body['reasoning_effort']=='medium' and body['chat_template_kwargs']['enable_thinking'] is True
        for effort,expected in [('none','none'),('minimal','low'),('medium','medium'),('high','xhigh'),('max','xhigh')]:
            assert (await send(reasoning_effort=effort,repetition_penalty=1.1,max_completion_tokens=64)).status_code==200
            body=[b for p,_,b in calls if p=='/v1/chat/completions'][-1]
            assert body['reasoning_effort']==expected
            assert body['chat_template_kwargs']['enable_thinking']==(effort!='none')
            assert body['repeat_penalty']==1.1 and body['max_tokens']==64
        assert (await send(chat_template_kwargs={'enable_thinking':False})).status_code==200
        body=[b for p,_,b in calls if p=='/v1/chat/completions'][-1]
        assert body['chat_template_kwargs']['enable_thinking'] is False
        assert (await send(top_a=0.1)).status_code==400
        assert (await send(max_tokens=64,max_completion_tokens=64)).status_code==400
    run(case,tmp_path)


def test_multi_system_normalized_for_template_and_generation(tmp_path):
    async def case(client,calls,peak):
        messages=[{'role':'system','content':'A'},{'role':'system','content':'B'},
            {'role':'user','content':'q'},{'role':'assistant','content':'answer','reasoning_content':'thought'},
            {'role':'user','content':'followup'}]
        response=await client.post('/v1/chat/completions',json={'messages':messages,'stream':True})
        assert response.status_code==200 and response.headers['x-request-id']
        for path,_,body in calls:
            if path in ('/apply-template','/v1/chat/completions'):
                assert body['messages']==[{'role':'system','content':'A\n\nB'},*messages[2:]]
        calls.clear()
        response=await client.post('/v1/chat/completions',json={'messages':[messages[2],messages[0]]})
        assert response.status_code==400
        assert response.json()['error']['type']=='invalid_request_error'
        assert response.json()['error']['request_id']==response.headers['x-request-id']
        assert not calls and not g.lock.locked()
    run(case,tmp_path)


def test_bad_input_does_not_erase_resident_cache(tmp_path):
    async def case(client,calls,peak):
        assert (await client.post('/v1/chat/completions',json={'messages':[],'session_id':'A'})).status_code==200
        previous=g.resident
        calls.clear()
        invalid=await client.post('/v1/chat/completions',content=b'{broken',headers={'Content-Type':'application/json'})
        assert invalid.status_code==400 and invalid.headers['x-request-id']
        too_long=await client.post('/v1/chat/completions',json={'messages':[],'max_tokens':999999,'stream':True})
        assert too_long.status_code==400
        assert g.resident==previous and not g.lock.locked()
        assert not any(params.get('action')=='erase' for _,params,_ in calls)
    run(case,tmp_path)


@pytest.mark.parametrize('backend_status,expected',[(500,400),(422,400),(503,502)])
def test_template_error_classification(tmp_path,backend_status,expected):
    async def case(client,calls,peak):
        result=await client.post('/v1/chat/completions',json={'messages':[],'stream':True})
        assert result.status_code==expected
        assert result.json()['error']['request_id']==result.headers['x-request-id']
        assert not g.lock.locked()
        assert not any(params.get('action')=='erase' for _,params,_ in calls)
    run(case,tmp_path,template_status=backend_status)
