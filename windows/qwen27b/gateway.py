"""Authenticated single-generation gateway. Inference and token-prefix matching are native."""
import asyncio
from contextlib import asynccontextmanager
import hashlib
import hmac
import ipaddress
import json
import logging
import os
from pathlib import Path
import re
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from store import Store
from auth import principal, session_key
from prompt_cache import preparation_key
from chat_messages import normalize_messages, normalize_tool_choice
from model_profiles import load_catalog, atomic_json, prune_shared

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / os.getenv('LLM_SERVICE_CONFIG', 'config.local.json')).read_text())
keyfile = ROOT/'state'/'api-key.txt'
try:
    KEY = keyfile.read_text().strip()
except FileNotFoundError:
    KEY = ''
if len(KEY) < 24:
    raise RuntimeError('Create state/api-key.txt with at least 24 characters')
NETWORKS = [ipaddress.ip_network(n) for n in CONFIG['allowed_networks']]
MODEL = CONFIG['model']
CATALOG_FILE = os.getenv('LLM_SERVICE_MODEL_CATALOG')
PROFILES = load_catalog(ROOT, CONFIG, CATALOG_FILE)
active_model = MODEL
switching = False
stores = {}
store = None
client = None
lock = asyncio.Lock()
waiting = 0
healthy = True
version = None
resident = None  # (namespaced session ID, committed filename, digest); protected by lock
prepared_prompt = None  # one resident request's text/token preparation, keyed by request-character SHA

def sid_checked(value):
    if value is not None and (not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',value)):
        raise HTTPException(400, 'Invalid session_id')
    return value

async def upstream(path, body=None):
    response = await (client.get(path) if body is None else client.post(path,json=body))
    response.raise_for_status()
    return response.json()

async def idle():
    global healthy
    for _ in range(120):
        slots = await upstream('/slots')
        if len(slots) != 1:
            healthy = False
            raise RuntimeError('Runtime must have exactly one slot')
        if not slots[0]['is_processing']:
            return
        await asyncio.sleep(0.25)
    healthy = False
    raise RuntimeError('Runtime did not become idle; gateway is fail-closed')

async def erase():
    global resident, prepared_prompt
    resident = None
    prepared_prompt = None
    await idle()
    return await upstream('/slots/0?action=erase', {})

async def activate_profile(name):
    global CONFIG, store, version, active_model, resident, prepared_prompt, healthy
    profile = PROFILES[name]
    CONFIG = profile['config']
    store = stores[name]
    manifest = json.loads((ROOT/'models'/profile['manifest']).read_text())
    if manifest.get('verified_sha256') != manifest['sha256']:
        raise RuntimeError('Model has not passed SHA256 verification')
    props = await upstream('/props')
    fingerprint = dict(model=manifest['sha256'], config=CONFIG, template=props.get('chat_template'),
                       runtime=json.loads((ROOT/'reports'/'runtime-assets.json').read_text()),
                       built=json.loads((ROOT/'reports'/CONFIG.get('runtime_manifest','runtime-built.json')).read_text(encoding='utf-8-sig')),
                       patch=hashlib.sha256((ROOT/'patches'/'runtime.patch').read_bytes()).hexdigest(),schema=1)
    version = hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
    resident = prepared_prompt = None
    await erase()
    active_model = name
    healthy = True


async def ensure_model(name):
    """Called only under the generation lock, including the complete load wait."""
    global active_model, switching, resident, prepared_prompt
    if not CATALOG_FILE:
        return 0.0
    status = json.loads((ROOT/'state/runtime-status.json').read_text())
    if active_model == name and healthy and status.get('status') == 'ready' and status.get('model') == name:
        return 0.0
    switching = True
    resident = prepared_prompt = None
    active_model = None
    async def change():
        request_id = uuid.uuid4().hex
        atomic_json(ROOT/'state/model-switch-request.json',dict(id=request_id,model=name))
        deadline = time.monotonic()+660
        while time.monotonic() < deadline:
            path = ROOT/'state/model-switch-result.json'
            if path.exists():
                result = json.loads(path.read_text())
                if result.get('id') == request_id:
                    if not result['ok']:
                        raise HTTPException(503,'Model could not load; gateway remains available, retry or select another model')
                    await activate_profile(name)
                    return result['seconds']
            await asyncio.sleep(0.1)
        raise HTTPException(503,'Model switch timed out')
    task = asyncio.create_task(change())
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Never release the scheduling lock while the supervisor is changing models.
        try:
            await task
        except Exception:
            pass
        raise
    finally:
        switching = False


@asynccontextmanager
async def lifespan(app):
    global client, stores
    stores = {name: Store(ROOT/'state'/profile['slots'],profile['config']) for name,profile in PROFILES.items()}
    internal_key=(ROOT/'state'/'internal-key.txt').read_text().strip()
    client = httpx.AsyncClient(base_url=CONFIG['upstream'],timeout=CONFIG['generation_timeout'],trust_env=False,
                              headers={'Authorization':'Bearer '+internal_key})
    await activate_profile(MODEL)
    yield
    await client.aclose()
    for item in stores.values():
        item.db.close()

app = FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)
logger = logging.getLogger('llm_service.gateway')
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    diagnostic_handler = logging.StreamHandler()
    diagnostic_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(diagnostic_handler)


@app.exception_handler(HTTPException)
async def api_error(request, exc):
    return JSONResponse({'error': {'message': str(exc.detail),
        'type': 'invalid_request_error' if exc.status_code < 500 else 'server_error',
        'code': exc.status_code, 'request_id': getattr(request.state, 'request_id', None)}},
        status_code=exc.status_code, headers=exc.headers)


async def request_diagnostics(request, call_next):
    # Server-generated ID; never log request bodies, credentials or upstream error text.
    request.state.request_id = uuid.uuid4().hex
    safe_path = '/sessions/{id}' if request.url.path.startswith('/sessions/') else (
        request.url.path if request.url.path in ('/health','/sessions','/v1/models','/v1/chat/completions','/v1/completions') else '<unmatched>')
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error('request_id=%s method=%s path=%s exception=%s',
            request.state.request_id, request.method, safe_path, type(exc).__name__)
        return JSONResponse({'error': {'message': 'Internal server error', 'type': 'server_error',
            'request_id': request.state.request_id}}, 500,
            headers={'X-Request-ID': request.state.request_id})
    response.headers['X-Request-ID'] = request.state.request_id
    logger.log(logging.WARNING if response.status_code >= 400 else logging.INFO,
        'request_id=%s method=%s path=%s status=%s headers_ms=%.1f',
        request.state.request_id, request.method, safe_path, response.status_code,
        (time.perf_counter()-started)*1000)
    return response

@app.middleware('http')
async def authenticate(request, call_next):
    peer = ipaddress.ip_address(request.client.host)
    if not any(peer in net for net in NETWORKS):
        return JSONResponse({'error':{'message':'Network not allowed','type':'permission_error'}},403)
    authorization=request.headers.get('authorization','')
    who=principal(authorization[7:],KEY,ROOT/'state'/'users.json') if authorization.startswith('Bearer ') else None
    if who is None:
        return JSONResponse({'error':{'message':'Invalid API key','type':'authentication_error'}},401)
    request.state.principal=who
    request.state.auth_token=authorization[7:]
    return await call_next(request)

app.middleware('http')(request_diagnostics)

@app.get('/health')
async def health():
    if CATALOG_FILE:
        status = json.loads((ROOT/'state/runtime-status.json').read_text())
        result = dict(status='switching' if switching else status['status'],gateway='online',
            model=active_model,loading_model=status.get('model') if switching else None,
            waiting=waiting,active=int(lock.locked()),context=CONFIG['context'])
        if not switching and (status['status'] != 'ready' or not healthy):
            result['status']='unavailable'
            return JSONResponse(result,503)
        return result
    try:
        await upstream('/health')
        ok = healthy
    except Exception:
        ok = False
    return JSONResponse(dict(status='ready' if ok else 'unavailable',waiting=waiting,active=int(lock.locked()),context=CONFIG['context']),200 if ok else 503)

@app.get('/v1/models')
async def models():
    return {'object':'list','data':[{'id':name,'object':'model','owned_by':'local'} for name in PROFILES]}

def session_store(request, model=None):
    name = model or request.query_params.get('model') or MODEL
    if name not in PROFILES:
        raise HTTPException(404,'Unknown model')
    return stores.get(name,store), name

@app.post('/sessions')
async def create_session(request: Request):
    body = await request.json()
    sid = sid_checked(body.get('session_id')) or uuid.uuid4().hex
    target, name = session_store(request,body.get('model'))
    await asyncio.to_thread(target.create,session_key(request.state.principal,sid))
    return {'session_id':sid,'model':name}

@app.get('/sessions/{sid}')
async def get_session(sid: str, request: Request):
    target, name = session_store(request)
    row = await asyncio.to_thread(target.get,session_key(request.state.principal,sid_checked(sid)))
    if not row:
        raise HTTPException(404,'Session not found')
    return {k:row[k] for k in ('size','version','accessed')} | {'id':sid,'last_request':row['metadata'].get('metrics',{})}

@app.delete('/sessions/{sid}')
async def delete_session(sid: str, request: Request):
    sid_checked(sid)
    async with lock:
        target, name = session_store(request)
        key = session_key(request.state.principal,sid)
        if active_model == name and resident and resident[0] == key:
            await erase()
        await asyncio.to_thread(target.delete,key)
    return {'deleted':True,'session_id':sid}

async def acquire(request):
    global waiting
    if not healthy and not CATALOG_FILE:
        raise HTTPException(503,'Runtime state uncertain; restart service')
    if waiting >= CONFIG['queue_limit']:
        raise HTTPException(429,'Generation queue full')
    waiting += 1
    acquired = asyncio.create_task(lock.acquire())
    delivered = False
    try:
        deadline = time.monotonic()+CONFIG['queue_timeout']
        while True:
            await asyncio.wait({acquired},timeout=0.25)
            if await request.is_disconnected():
                raise HTTPException(499,'Client disconnected')
            if acquired.done():
                acquired.result()
                if principal(request.state.auth_token,KEY,ROOT/'state'/'users.json') != request.state.principal:
                    raise HTTPException(401,'API key revoked while queued')
                delivered = True
                return
            if time.monotonic() >= deadline:
                raise HTTPException(503,'Generation queue timeout')
    finally:
        waiting -= 1
        if not delivered:
            if not acquired.done():
                acquired.cancel()
            try:
                if await acquired:
                    lock.release()
            except asyncio.CancelledError:
                pass

async def post_with_disconnect(request, body):
    pending = asyncio.create_task(client.post(request.url.path,json=body))
    deadline = time.monotonic()+CONFIG['generation_timeout']
    try:
        while True:
            await asyncio.wait({pending},timeout=0.25)
            if pending.done():
                return pending.result()
            if await request.is_disconnected():
                raise HTTPException(499,'Client disconnected')
            if time.monotonic() >= deadline:
                raise HTTPException(504,'Generation timeout')
    finally:
        if not pending.done():
            pending.cancel()
            try:
                await pending
            except asyncio.CancelledError:
                pass

async def prepare(sid, prompt_text):
    if not sid:
        await erase()
        return 'stateless',0,'none'
    row, status = await asyncio.to_thread(store.restore_candidate,sid,version)
    if row is None:
        await erase()
        return status,0,'none'
    fingerprint = row['metadata'].get('prompt_fingerprint')
    text_match = 'legacy_untracked'
    if fingerprint:
        chars = fingerprint['characters']
        matches = len(prompt_text)>=chars and hashlib.sha256(prompt_text[:chars].encode('utf-8')).hexdigest()==fingerprint['sha256']
        text_match = ('exact' if len(prompt_text)==chars else 'prefix') if matches else 'edited'
    # Text hashes identify candidate prefix reuse; native token IDs remain the
    # authority. Immutable-file size/mtime checks do not read KV contents.
    if resident == (sid,row['filename'],row['digest']):
        await idle()
        return 'resident_unverified',0,text_match
    await erase()
    start = time.perf_counter()
    try:
        result = await upstream('/slots/0?action=restore',{'filename':row['filename']})
        if result.get('n_restored',0) <= 0:
            raise ValueError('Empty restore')
    except Exception:
        await erase()
        return 'restore_failed',time.perf_counter()-start,text_match
    # This is not called a hit until actual inference reports cached tokens.
    return 'restored_unverified',time.perf_counter()-start,text_match

async def save(sid, body, replies, metrics, input_tokens, prompt_fingerprint, preparation):
    global resident, prepared_prompt
    if not sid:
        return False
    await idle()
    tmp = uuid.uuid4().hex+'.tmp'
    start = time.perf_counter()
    try:
        native = await upstream('/slots/0?action=save',{'filename':tmp})
        metrics['save_seconds'] = time.perf_counter()-start
        metrics['snapshot_bytes'] = (store.root/tmp).stat().st_size
        # Native snapshot contains the exact effective token IDs AND recurrent checkpoints.
        metadata = dict(request=body,responses=replies,metrics=metrics,native_save=native,input_token_ids=input_tokens,
                        token_prefix_location='native_snapshot_server_tokens_payload',prompt_fingerprint=prompt_fingerprint)
        if CATALOG_FILE:
            await asyncio.to_thread(prune_shared,stores,min(p['config']['cache_max_bytes'] for p in PROFILES.values()),
                metrics['snapshot_bytes'],(active_model,sid))
        size = await asyncio.to_thread(store.commit,sid,tmp,version,metadata)
        metrics['snapshot_bytes'] = size
        committed = await asyncio.to_thread(store.get,sid)
        resident = (sid,committed['filename'],committed['digest'])
        prepared_prompt = preparation
        return True
    finally:
        (store.root/tmp).unlink(missing_ok=True)

def collect_metrics(replies,status,restore_seconds,start):
    timings = {}
    usage = {}
    for r in replies:
        timings.update(r.get('timings',{}))
        usage.update(r.get('usage') or {})
    cached = timings.get('cache_n',usage.get('prompt_tokens_details',{}).get('cached_tokens',0))
    total = usage.get('prompt_tokens', cached+timings.get('prompt_n',0))
    result = dict(restore_status=status,restore_seconds=restore_seconds,prompt_tokens=total,
                  cached_tokens=cached,prefill_tokens=timings.get('prompt_n',total-cached),
                  hit_ratio=cached/total if total else 0,timings=timings,
                  generation_seconds=time.perf_counter()-start)
    if status == 'restored_unverified':
        result['restore_status'] = 'prefix_hit' if cached else 'cache_miss_after_restore'
    elif status == 'resident_unverified':
        result['restore_status'] = 'resident_prefix_hit' if cached else 'resident_cache_miss'
    return result

async def cleanup(preserve=False):
    global healthy
    try:
        if preserve and resident is not None:
            await idle()
        else:
            await erase()
    except Exception:
        healthy = False
    finally:
        lock.release()

@app.post('/v1/chat/completions')
@app.post('/v1/completions')
async def completion(request: Request):
    received = time.perf_counter()
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(400,'Request body must be valid JSON') from exc
    if not isinstance(body,dict):
        raise HTTPException(400,'Expected a JSON object')
    if request.url.path == '/v1/chat/completions':
        try:
            body['messages'] = normalize_messages(body.get('messages'))
            normalize_tool_choice(body)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    sid_body = body.pop('session_id',None)
    sid_head = request.headers.get('x-session-id')
    if sid_body and sid_head and sid_body != sid_head:
        raise HTTPException(400,'Conflicting session IDs')
    sid = session_key(request.state.principal,sid_checked(sid_head or sid_body))
    requested_model = body.get('model',MODEL)
    if not isinstance(requested_model,str) or requested_model not in PROFILES:
        raise HTTPException(404,'Unknown model')
    if body.get('n',1) != 1 or body.get('best_of',1) != 1:
        raise HTTPException(400,'Only one generation per request is supported')
    if request.url.path == '/v1/completions' and isinstance(body.get('prompt'),list) and any(isinstance(x,(str,list)) for x in body['prompt']):
        raise HTTPException(400,'Batched prompts are not supported')
    # Prevent callers selecting slots, loading files, or overriding lifecycle controls.
    for key in ('id_slot','slot_id','cache_prompt','cache_id','cache_reuse','prompt_cache','save_prompt','n_keep','n_discard','n_predict'):
        body.pop(key,None)
    body.update(model=requested_model,id_slot=0,cache_prompt=True)
    if body.get('max_tokens') is not None and body.get('max_completion_tokens') is not None:
        raise HTTPException(400,'Use only one of max_tokens and max_completion_tokens')
    if 'repetition_penalty' in body:
        body['repeat_penalty'] = body.pop('repetition_penalty')
    # This backend has no Top-A sampler; never silently pretend it was applied.
    if body.pop('top_a',0) not in (None,0):
        raise HTTPException(400,'top_a is not supported by this runtime; leave it blank or set 0')
    if 'max_completion_tokens' in body:
        body['max_tokens'] = body.pop('max_completion_tokens')
    body.setdefault('max_tokens',512)
    if not isinstance(body['max_tokens'],int) or body['max_tokens'] <= 0:
        raise HTTPException(400,'max_tokens must be positive')
    kwargs = body.setdefault('chat_template_kwargs',{})
    if not isinstance(kwargs,dict):
        raise HTTPException(400,'chat_template_kwargs must be an object')
    effort = body.get('reasoning_effort') or 'medium'
    if effort not in ('none','minimal','low','medium','high','xhigh','max'):
        raise HTTPException(400,'Invalid reasoning_effort')
    # Map adjacent client levels to supported reasoning presets: low/medium/xhigh.
    body['reasoning_effort'] = {'minimal':'low','high':'xhigh','max':'xhigh'}.get(effort,effort)
    kwargs.setdefault('enable_thinking',effort != 'none')
    queue_start = time.perf_counter()
    await acquire(request)
    queue_wait_seconds = time.perf_counter()-queue_start
    committed = False
    backend_state_touched = False
    try:
        model_switch_seconds = await ensure_model(requested_model)
        if await request.is_disconnected():
            raise HTTPException(499,'Client disconnected during model load')
        preparation_start = time.perf_counter()
        request_hash = preparation_key(request.url.path,body)
        preparation_hit = bool(sid and resident and prepared_prompt and prepared_prompt[:2]==(sid,request_hash))
        if preparation_hit:
            _,_,prompt_text,input_tokens,prompt_fingerprint = prepared_prompt
        else:
            prompt = body.get('prompt')
            if request.url.path == '/v1/chat/completions':
                try:
                    prompt = (await upstream('/apply-template',body))['prompt']
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code in (400, 422) or 'Jinja Exception' in exc.response.text:
                        raise HTTPException(400,'Messages cannot be rendered by this model: check message roles, content and tool-call structure') from exc
                    raise
            input_tokens = (await upstream('/tokenize',{'content':prompt,'add_special':True,'parse_special':True}))['tokens']
            prompt_text = prompt if isinstance(prompt,str) else (await upstream('/detokenize',{'tokens':input_tokens}))['content']
            prompt_fingerprint = {'characters':len(prompt_text),'sha256':hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()}
        if len(input_tokens)+body['max_tokens'] > CONFIG['context']:
            raise HTTPException(400,'Prompt plus max_tokens exceeds configured context; history was not truncated')
        preparation = (sid,request_hash,prompt_text,input_tokens,prompt_fingerprint)
        text_processing_seconds = time.perf_counter()-preparation_start
        cache_check_start = time.perf_counter()
        backend_state_touched = True
        status, restore_seconds, text_match = await prepare(sid,prompt_text)
        cache_prepare_seconds = time.perf_counter()-cache_check_start
        preparation_metrics = dict(text_processing_seconds=text_processing_seconds,
            cache_prepare_seconds=cache_prepare_seconds,text_cache_match=text_match,preparation_cache_hit=preparation_hit,
            queue_wait_seconds=queue_wait_seconds,model_switch_seconds=model_switch_seconds)
        start = time.perf_counter()
        if not body.get('stream',False):
            response = await post_with_disconnect(request,body)
            if response.is_error:
                return JSONResponse(response.json(),response.status_code)
            result = response.json()
            metrics = collect_metrics([result],status,restore_seconds,start)
            metrics.update(preparation_metrics)
            committed = await save(sid,body,[result],metrics,input_tokens,prompt_fingerprint,preparation)
            result['session_cache'] = metrics
            return JSONResponse(result)
        body['stream_options'] = dict(body.get('stream_options') or {},include_usage=True)
        upstream_response = await client.send(client.build_request('POST',request.url.path,json=body),stream=True)
        if upstream_response.is_error:
            await upstream_response.aread()
            result = JSONResponse(upstream_response.json(),upstream_response.status_code)
            await upstream_response.aclose()
            return result
    except httpx.HTTPError:
        raise HTTPException(502,'Inference backend failed')
    finally:
        if not body.get('stream') or 'upstream_response' not in locals() or upstream_response.is_error:
            if backend_state_touched:
                await asyncio.shield(cleanup(preserve=committed))
            else:
                lock.release()

    async def events():
        replies = []
        first_token_seconds = None
        first_content_seconds = None
        completed = False
        committed = False
        try:
            async with asyncio.timeout(CONFIG['generation_timeout']):
                async for line in upstream_response.aiter_lines():
                    if line.startswith('data: '):
                        payload = line[6:]
                        if payload == '[DONE]':
                            completed = True
                            break
                        item = json.loads(payload)
                        if item.get('error'):
                            raise RuntimeError('Backend stream reported an error')
                        for choice in item.get('choices',[]):
                            delta = choice.get('delta') or {}
                            content = delta.get('content') or choice.get('text')
                            token = content or delta.get('reasoning_content') or delta.get('tool_calls')
                            if token and first_token_seconds is None:
                                first_token_seconds = time.perf_counter()-received
                            if content and first_content_seconds is None:
                                first_content_seconds = time.perf_counter()-received
                        replies.append(item)
                    yield (line+'\n').encode()
                if not completed:
                    raise RuntimeError('Backend stream ended before completion')
                metrics = collect_metrics(replies,status,restore_seconds,start)
                metrics.update(preparation_metrics)
                metrics.update(first_token_seconds=first_token_seconds,first_content_seconds=first_content_seconds)
                committed = await save(sid,body,replies,metrics,input_tokens,prompt_fingerprint,preparation)
                yield b'data: [DONE]\n\n'
        except asyncio.CancelledError:
            raise
        except Exception:
            yield b'data: {"error":{"message":"Generation or durable snapshot commit failed","type":"server_error"}}\n\ndata: [DONE]\n\n'
        finally:
            try:
                await upstream_response.aclose()
            finally:
                await asyncio.shield(cleanup(preserve=committed))
    return StreamingResponse(events(),media_type='text/event-stream',headers={'Cache-Control':'no-cache'})
