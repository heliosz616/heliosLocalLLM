"""Supervised, single-execution OpenAI LAN gateway for the two pinned models."""
from contextlib import asynccontextmanager
from pathlib import Path
import argparse
import asyncio
import hashlib
import hmac
import json
import logging
import os
import signal
import socket
import sqlite3
import time
import uuid

import httpx
from anyio import CancelScope, create_task_group
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
import legacy_chat
import reasoning_controls
from request_timing import StreamTiming
from addresses import discover as discover_addresses
from cache_budget import cap_native_cache, GIB
from cache_retention import cache_roots, prune_snapshots, enforce_capacity
from reasoning_history import local_reasoning_messages
from starlette.requests import ClientDisconnect
from starlette.exceptions import HTTPException
from starlette.datastructures import Headers
import uvicorn
from settings import load_json

from profiles import ALIASES, Q5, ROOT, profile, verify_artifacts

LOG = logging.getLogger('swift.serving')
STATE = ROOT / 'state/serving'


def api_error(message, status=400, param=None, code=None):
    return JSONResponse({'error': {'message': message,
                                  'type': 'invalid_request_error' if status < 500 else 'server_error',
                                  'param': param, 'code': code}}, status_code=status)


class ProtocolError(ValueError):
    def __init__(self, message, status=400, param=None):
        super().__init__(message)
        self.status, self.param = status, param


class ClientGone(Exception):
    pass


class AuthenticationMiddleware:
    """Pure ASGI: preserve the original receive channel for disconnect checks."""
    def __init__(self, app, credentials):
        self.app, self.credentials = app, credentials

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        state = scope.setdefault('state', {})
        rid = 'req_' + uuid.uuid4().hex
        state['request_id'] = rid
        async def traced_send(message):
            if message['type'] == 'http.response.start':
                message['headers'] = list(message.get('headers', [])) + [(b'x-request-id', rid.encode())]
            await send(message)
        if scope['path'] not in ('/health', '/ready'):
            authorization = Headers(scope=scope).get('authorization', '')
            token = authorization[7:] if authorization.lower().startswith('bearer ') else ''
            if not any(hmac.compare_digest(token.encode(), value.encode()) for value in self.credentials):
                response = api_error('Invalid API key', 401, code='invalid_api_key')
                response.headers['WWW-Authenticate'] = 'Bearer'
                return await response(scope, receive, traced_send)
            state['owner'] = hashlib.sha256(token.encode()).hexdigest()
        return await self.app(scope, receive, traced_send)


def notify(message):
    address = os.environ.get('NOTIFY_SOCKET')
    if not address:
        return
    if address.startswith('@'):
        address = '\0' + address[1:]
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
        channel.sendto(message.encode(), address)


class ResponseStore:
    """Atomic durable response chains, scoped to the bearer credential."""
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS responses '
                        '(id TEXT PRIMARY KEY, owner TEXT NOT NULL, model TEXT NOT NULL, '
                        'input TEXT NOT NULL, response TEXT NOT NULL, created REAL NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS items '
                        '(id TEXT NOT NULL, owner TEXT NOT NULL, model TEXT NOT NULL, item TEXT NOT NULL, '
                        'PRIMARY KEY(id, owner, model))')
        self.db.execute('CREATE TABLE IF NOT EXISTS response_items '
                        '(response_id TEXT, item_id TEXT, owner TEXT, model TEXT, '
                        'PRIMARY KEY(response_id,item_id,owner,model))')
        if self.db.execute('PRAGMA user_version').fetchone()[0] < 1:
            for rid, owner, model, inputs, response in self.db.execute('SELECT id,owner,model,input,response FROM responses').fetchall():
                for item in json.loads(inputs) + json.loads(response).get('output', []):
                    if isinstance(item, dict) and isinstance(item.get('id'), str):
                        self.db.execute('INSERT OR IGNORE INTO response_items VALUES (?,?,?,?)', (rid, item['id'], owner, model))
            self.db.execute('PRAGMA user_version=1')
        self.db.commit()

    def get(self, rid, owner, model=None):
        row = self.db.execute('SELECT owner,model,input,response FROM responses WHERE id=?', (rid,)).fetchone()
        if row is None or not hmac.compare_digest(row[0], owner):
            raise ProtocolError('Response not found', 404, 'previous_response_id')
        if model is not None and row[1] != model:
            raise ProtocolError('A response chain must use the same model', 400, 'model')
        return row[1], json.loads(row[2]), json.loads(row[3])

    def save(self, owner, model, inputs, response):
        rid = response.get('id')
        if not isinstance(rid, str) or not rid:
            raise RuntimeError('Backend completed without a response ID')
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?)',
                            (rid, owner, model, json.dumps(inputs, ensure_ascii=False),
                             json.dumps(response, ensure_ascii=False), time.time()))
            for item in inputs + response.get('output', []):
                if isinstance(item, dict) and isinstance(item.get('id'), str):
                    self.db.execute('INSERT OR REPLACE INTO items VALUES (?,?,?,?)',
                                    (item['id'], owner, model, json.dumps(item, ensure_ascii=False)))
                    self.db.execute('INSERT OR IGNORE INTO response_items VALUES (?,?,?,?)',
                                    (rid, item['id'], owner, model))

    def delete(self, rid, owner):
        self.get(rid, owner)
        with self.db:
            self.db.execute('DELETE FROM responses WHERE id=? AND owner=?', (rid, owner))
            self.db.execute('DELETE FROM response_items WHERE response_id=? AND owner=?', (rid, owner))
            self.db.execute('DELETE FROM items WHERE owner=? AND NOT EXISTS '
                            '(SELECT 1 FROM response_items r WHERE r.item_id=items.id AND r.owner=items.owner AND r.model=items.model)', (owner,))

    def prepare(self, payload, owner, model):
        body = dict(payload)
        inputs = body.get('input', [])
        if inputs is None:
            inputs = []
        if isinstance(inputs, str):
            inputs = [{'role': 'user', 'content': inputs}]
        if not isinstance(inputs, list) or any(not isinstance(item, dict) for item in inputs):
            raise ProtocolError('input must be a string or array of objects', param='input')
        previous = body.pop('previous_response_id', None)
        if previous is not None and not isinstance(previous, str):
            raise ProtocolError('previous_response_id must be a string or null', param='previous_response_id')
        if previous:
            # Response history is protocol data. Retokenize it with the chosen
            # model; do not impose a same-model gate or share physical KV state.
            _, history, response = self.get(previous, owner)
            inputs = history + response.get('output', []) + inputs
        expanded = []
        for item in inputs:
            if item.get('type') == 'item_reference':
                row = self.db.execute('SELECT item FROM items WHERE id=? AND owner=? ORDER BY (model=?) DESC, rowid DESC LIMIT 1',
                                      (item.get('id'), owner, model)).fetchone()
                if row is None and isinstance(item.get('id'), str) and '_input_' in item['id']:
                    # Input-listing IDs also identify persisted ordinary user
                    # messages, which did not arrive with an upstream item ID.
                    rid, _, position = item['id'].rpartition('_input_')
                    if position.isdecimal():
                        try:
                            _, original, _ = self.get(rid, owner)
                        except ProtocolError:
                            original = []
                        index = int(position)
                        if index < len(original):
                            row = (json.dumps(original[index], ensure_ascii=False),)
                if row is None:
                    raise ProtocolError('Referenced input item not found', 404, 'input')
                item = json.loads(row[0])
            expanded.append(item)
        # Preserve the exact public item history for storage and item references.
        # The local native converter otherwise renders a separate reasoning item
        # and its answer as two assistant turns, changing conversation prefixes.
        body['input'] = local_reasoning_messages(expanded)
        return body, expanded

    def close(self):
        self.db.close()


class BackendManager:
    def __init__(self, state, startup_timeout=900, ssd_disk_reserve_bytes=32 * GIB, kv_cache_maximum_bytes=64_000_000_000):
        self.state = state
        self.startup_timeout = startup_timeout
        self.ssd_disk_reserve_bytes = ssd_disk_reserve_bytes
        self.kv_cache_maximum_bytes = kv_cache_maximum_bytes
        self.lock = asyncio.Lock()
        self.process = None
        self.log_file = None
        self.active_model = None
        self.url = None
        self.ready = False
        self.queued = 0
        self.running = False
        self.switches = 0
        self.failures = 0
        self.started = time.time()

    def status(self):
        alive = self.process is not None and self.process.returncode is None
        return {'status': 'ok', 'active_model': self.active_model, 'model_ready': self.ready and alive,
                'request_running': self.running, 'queued_requests': self.queued,
                'model_switches': self.switches, 'backend_failures': self.failures,
                'uptime_seconds': round(time.time() - self.started, 1)}

    async def stop(self):
        self.ready = False
        if self.process is not None and self.process.returncode is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.process.wait(), timeout=45)
            except asyncio.TimeoutError:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await self.process.wait()
        self.process = None
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        self.active_model = self.url = None
        (self.state / 'backend.json').unlink(missing_ok=True)

    async def ensure(self, model, client):
        if self.active_model == model and self.process is not None and self.process.returncode is None and self.ready:
            try:
                health = await client.get(self.url + '/health', timeout=3)
                if health.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
        await asyncio.to_thread(verify_artifacts, model, self.state)
        await self.stop()
        command, env, url = profile(model, self.state)
        cache_budget = await asyncio.to_thread(cap_native_cache, env, self.ssd_disk_reserve_bytes,
                                                   roots=cache_roots(self.state, ROOT),
                                                   total_maximum_bytes=self.kv_cache_maximum_bytes)
        if cache_budget is not None:
            cache_budget['model'] = model
            temporary = self.state / 'cache-budget.json.part'
            temporary.write_text(json.dumps(cache_budget, indent=2) + '\n')
            temporary.replace(self.state / 'cache-budget.json')
            LOG.info('SSD cache model=%s limit=%.2fGiB free=%.2fGiB reserve=%.2fGiB', model,
                     cache_budget['effective_maximum_bytes'] / GIB,
                     cache_budget['free_at_activation_bytes'] / GIB,
                     cache_budget['disk_reserve_bytes'] / GIB)
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        log_path = self.state / 'logs' / f'{model}-{stamp}-{uuid.uuid4().hex[:6]}.log'
        self.log_file = log_path.open('wb')
        self.active_model, self.url = model, url
        LOG.info('loading model=%s log=%s', model, log_path)
        began = time.monotonic()
        try:
            spawn = asyncio.create_task(asyncio.create_subprocess_exec(
                *command, cwd=ROOT, env=env, stdout=self.log_file,
                stderr=asyncio.subprocess.STDOUT, start_new_session=True))
            try:
                self.process = await asyncio.shield(spawn)
            except asyncio.CancelledError:
                self.process = await spawn
                raise
            record = {'pid': self.process.pid, 'model': model, 'url': url, 'log': str(log_path), 'command': command}
            temporary = self.state / 'backend.json.part'
            temporary.write_text(json.dumps(record, indent=2) + '\n')
            temporary.replace(self.state / 'backend.json')
            while time.monotonic() - began < self.startup_timeout:
                if self.process.returncode is not None:
                    raise RuntimeError(f'Backend exited during startup; see {log_path.name}')
                try:
                    response = await client.get(url + '/health', timeout=2)
                    if response.status_code == 200:
                        self.ready = True
                        self.switches += 1
                        LOG.info('ready model=%s load_seconds=%.3f', model, time.monotonic() - began)
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(.5)
            raise RuntimeError(f'Backend startup timed out; see {log_path.name}')
        except BaseException:
            self.failures += 1
            await asyncio.shield(self.stop())
            raise

    async def acquire(self, request=None):
        self.queued += 1
        task = asyncio.create_task(self.lock.acquire())
        acquired = False
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=.25)
                if request is not None and await request.is_disconnected():
                    raise ClientGone()
            acquired = await task
            self.running = True
        except BaseException:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            elif not task.cancelled() and task.exception() is None and task.result():
                self.lock.release()
            raise
        finally:
            self.queued -= 1

    def release(self):
        self.running = False
        self.lock.release()


async def connected(request, awaitable):
    task = asyncio.create_task(awaitable)
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=.25)
            if await request.is_disconnected():
                raise ClientGone()
        return await task
    except BaseException:
        if not task.done():
            task.cancel()
        try:
            await task
        except BaseException:
            pass
        raise


async def with_keepalive(chunks, seconds):
    """Keep an SSE connection active without manufacturing model output."""
    iterator = chunks.__aiter__()
    pending = None
    try:
        while True:
            pending = asyncio.create_task(iterator.__anext__())
            while not pending.done():
                await asyncio.wait({pending}, timeout=seconds)
                if not pending.done():
                    yield b': keep-alive\n\n'
            try:
                chunk = await pending
            except StopAsyncIteration:
                break
            pending = None
            yield chunk
    finally:
        if pending is not None:
            if not pending.done():
                pending.cancel()
            try:
                await pending
            except (asyncio.CancelledError, StopAsyncIteration, httpx.HTTPError):
                pass
        close = getattr(iterator, 'aclose', None)
        if close is not None:
            await close()


async def close_stream(chunks, cleanup):
    # ASGI cancellation scopes otherwise interrupt socket closure itself.
    with CancelScope(shield=True):
        try:
            close = getattr(chunks, 'aclose', None)
            if close is not None:
                await close()
        finally:
            await cleanup()


class ManagedStreamingResponse(StreamingResponse):
    def __init__(self, *args, cleanup, disconnect, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup = cleanup
        self.disconnect = disconnect

    async def __call__(self, scope, receive, send):
        try:
            # Observe disconnects directly. Some ASGI2.4 servers silently
            # discard sends after disconnect, so send errors are insufficient.
            async with create_task_group() as group:
                async def body():
                    try:
                        await self.stream_response(send)
                    except (ClientDisconnect, OSError):
                        pass
                    finally:
                        group.cancel_scope.cancel()
                group.start_soon(body)
                await self.listen_for_disconnect(receive)
                # Close the network stream before cancelling its reader.
                # A cancelled HTTPX iterator can mark a response closed before
                # its asynchronous transport close has actually completed.
                with CancelScope(shield=True):
                    await self.disconnect()
                group.cancel_scope.cancel()
        finally:
            # Close the suspended generator and upstream on every response exit.
            await close_stream(self.body_iterator, self.cleanup)


def create_app(settings, manager=None, transport=None):
    state = Path(settings['state_directory'])
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state, 0o700)
    (state / 'logs').mkdir(exist_ok=True, mode=0o700)
    keys_path = Path(settings['api_key_file'])
    key = keys_path.read_text().strip()
    if not key:
        raise RuntimeError('API key file is empty')
    credentials = [key]
    if settings.get('additional_api_keys_file'):
        credentials += list(json.loads(Path(settings['additional_api_keys_file']).read_text()).values())

    @asynccontextmanager
    async def lifespan(app):
        store = ResponseStore(state / 'responses.sqlite3')
        backend = manager or BackendManager(state, settings.get('startup_timeout_seconds', 900),
                                            int(settings.get('ssd_disk_reserve_gib', 32) * GIB),
                                            int(settings.get('kv_cache_maximum_bytes', 64_000_000_000)))
        timeout = httpx.Timeout(settings.get('request_timeout_seconds', 3600), connect=10, pool=30)
        limits = httpx.Limits(max_keepalive_connections=settings.get('native_keepalive_connections', 20))
        async with httpx.AsyncClient(timeout=timeout, trust_env=False, transport=transport,
                                     limits=limits, headers={'Accept-Encoding': 'identity'}) as client:
            app.state.store, app.state.backend, app.state.client = store, backend, client
            async def warm():
                await backend.acquire()
                try:
                    await backend.ensure(settings.get('default_model', Q5), client)
                except Exception:
                    LOG.exception('initial model load failed; next request may retry')
                finally:
                    backend.release()
            warm_task = asyncio.create_task(warm()) if settings.get('warm_start', False) else None
            async def watchdog():
                while True:
                    notify('WATCHDOG=1')
                    await asyncio.sleep(10)
            heartbeat = asyncio.create_task(watchdog())
            async def maintain_cache():
                roots = cache_roots(state, ROOT)
                retention = float(settings.get('kv_cache_retention_hours', 24)) * 3600
                interval = float(settings.get('kv_cache_cleanup_interval_seconds', 300))
                while True:
                    # Maintenance waits for an idle inference stream. It never
                    # unloads the model or snapshots the hot GPU session.
                    if not backend.lock.locked() and not backend.queued:
                        async with backend.lock:
                            try:
                                work = asyncio.create_task(asyncio.to_thread(prune_snapshots, roots, retention, lock_path=state / '.kv-quota.lock'))
                                try:
                                    result = await asyncio.shield(work)
                                except asyncio.CancelledError:
                                    await work
                                    raise
                                result['capacity'] = await asyncio.to_thread(
                                    enforce_capacity, roots, int(settings.get('kv_cache_maximum_bytes', 64_000_000_000)),
                                    lock_path=state / '.kv-quota.lock')
                                temporary = state / 'cache-retention.json.part'
                                temporary.write_text(json.dumps(result, indent=2) + '\n')
                                temporary.replace(state / 'cache-retention.json')
                                if result['deleted_files']:
                                    LOG.info('KV retention expired %d files; released %.3fGiB',
                                             len(result['deleted_files']), result['removed_allocated_bytes'] / GIB)
                                if result['errors']:
                                    LOG.warning('KV retention errors: %s', result['errors'])
                            except Exception:
                                LOG.exception('KV retention maintenance failed')
                    await asyncio.sleep(max(1, interval))
            cache_maintenance = asyncio.create_task(maintain_cache())
            notify('READY=1')
            try:
                yield
            finally:
                notify('STOPPING=1')
                # The worker is shielded and awaited on cancellation, so no
                # filesystem work survives into native shutdown/persistence.
                cache_maintenance.cancel()
                try:
                    await cache_maintenance
                except asyncio.CancelledError:
                    pass
                heartbeat.cancel()
                try:
                    await heartbeat
                except asyncio.CancelledError:
                    pass
                if warm_task is not None and not warm_task.done():
                    warm_task.cancel()
                    try:
                        await warm_task
                    except asyncio.CancelledError:
                        pass
                await backend.stop()
                store.close()

    app = FastAPI(title='Swift inference service', lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return api_error(str(error.detail), error.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        return api_error('Invalid request parameters', 400)

    @app.exception_handler(Exception)
    async def unhandled_error(request, error):
        LOG.error('unhandled request failure id=%s', getattr(request.state, 'request_id', 'unknown'), exc_info=error)
        return api_error('Internal server error', 500, code='internal_error')

    app.add_middleware(AuthenticationMiddleware, credentials=credentials)

    @app.get('/health')
    async def health():
        return app.state.backend.status()

    @app.get('/ready')
    async def ready():
        status = app.state.backend.status()
        if not settings.get('warm_start', False) and status['active_model'] is None and not status['request_running']:
            status.update(status='idle', request_ready=True, load_on_demand=True)
            return JSONResponse(status)
        if status['model_ready']:
            try:
                response = await app.state.client.get(app.state.backend.url + '/health', timeout=3)
                status['model_ready'] = response.status_code == 200
            except httpx.HTTPError:
                status['model_ready'] = False
        status['request_ready'] = status['model_ready']
        status['load_on_demand'] = not settings.get('warm_start', False)
        return JSONResponse(status, status_code=200 if status['model_ready'] else 503)

    @app.get('/v1/models')
    async def models():
        return {'object': 'list', 'data': [{'id': model, 'object': 'model', 'created': 0,
                                           'owned_by': 'local'} for model in sorted(set(ALIASES.values()))]}

    @app.get('/v1/models/{name:path}')
    async def model_details(name: str):
        model = ALIASES.get(name)
        if model is None:
            return api_error('Model not found', 404, 'model')
        return {'id': model, 'object': 'model', 'created': 0, 'owned_by': 'local'}

    @app.post('/v1/models/{name}/load')
    async def load_without_prompt(name: str, request: Request):
        model = ALIASES.get(name)
        if model is None:
            return api_error('Model not found', 404, 'model')
        backend = app.state.backend
        held = False
        try:
            await backend.acquire(request)
            held = True
            await connected(request, backend.ensure(model, app.state.client))
            return {'model': model, 'loaded': True}
        except ClientGone:
            return api_error('Client disconnected', 499)
        except Exception:
            LOG.exception('Model-only activation failed')
            return api_error('Inference backend unavailable; retry the request', 503, code='backend_unavailable')
        finally:
            if held:
                backend.release()

    async def payload_for(request):
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > settings.get('max_request_bytes', 256 * 1024**2):
                raise ProtocolError('Request exceeds the configured byte capacity', 413)
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeError) as error:
            raise ProtocolError('Malformed JSON request') from error
        if not isinstance(body, dict):
            raise ProtocolError('Request body must be an object')
        requested = body.get('model')
        if requested is None:
            previous = body.get('previous_response_id')
            if isinstance(previous, str) and previous:
                requested = app.state.store.get(previous, request.state.owner)[0]
            else:
                requested = settings.get('default_model', Q5)
        if not isinstance(requested, str):
            raise ProtocolError('model must be a string', param='model')
        model = ALIASES.get(requested)
        if model is None:
            raise ProtocolError(f'Model {requested!r} is not available', 404, 'model')
        body['model'] = model
        return body, model

    async def proxy(request, path):
        backend, client = app.state.backend, app.state.client
        held, upstream, stream_chunks = False, None, None
        began = time.monotonic()
        model = None
        timing = StreamTiming(began)
        phases = {}
        async def close_upstream():
            nonlocal upstream
            with CancelScope(shield=True):
                if upstream is not None:
                    current, upstream = upstream, None
                    await current.aclose()
        async def finish_cleanup():
            nonlocal held, upstream
            with CancelScope(shield=True):
                try:
                    await close_upstream()
                finally:
                    if held:
                        held = False
                        backend.release()
                        LOG.info('request_done id=%s model=%s seconds=%.3f timing=%s', request.state.request_id, model, time.monotonic() - began, json.dumps(dict(phases, first_token_ms=timing.first_token_ms, first_text_ms=timing.first_text_ms), separators=(',', ':')))
        async def cleanup():
            nonlocal stream_chunks
            current, stream_chunks = stream_chunks, None
            # Background cleanup also owns the reader: ASGI can disconnect
            # while the body generator is suspended at yield.
            await close_stream(current, finish_cleanup)
        try:
            body, model = await payload_for(request)
            if path in ('/v1/chat/completions', '/v1/responses'):
                try:
                    body = reasoning_controls.prepare(body, responses=path == '/v1/responses')
                except ValueError as error:
                    raise ProtocolError(str(error), param='reasoning_effort' if path == '/v1/chat/completions' else 'reasoning') from error
            legacy = False
            if path == '/v1/chat/completions':
                try:
                    body, legacy = legacy_chat.prepare(body)
                except ValueError as error:
                    raise ProtocolError(str(error)) from error
            inputs = None
            if path == '/v1/responses':
                body, inputs = app.state.store.prepare(body, request.state.owner, model)
            phases['body_ready_ms'] = (time.monotonic() - began) * 1000
            phases['stream'] = bool(body.get('stream'))
            await backend.acquire(request)
            phases['queue_ms'] = (time.monotonic() - began) * 1000 - phases['body_ready_ms']
            held = True
            ensure_start = time.monotonic()
            await connected(request, backend.ensure(model, client))
            phases['model_ready_ms'] = (time.monotonic() - ensure_start) * 1000
            outgoing = client.build_request('POST', backend.url + path, json=body,
                                             headers={'x-request-id': request.state.request_id})
            upstream = await connected(request, client.send(outgoing, stream=True))
            headers = {'content-type': upstream.headers.get('content-type', 'application/json'),
                       'cache-control': 'no-store', 'x-accel-buffering': 'no'}
            status = upstream.status_code
            if not body.get('stream') or status >= 400:
                data = await connected(request, upstream.aread())
                if legacy and status < 400:
                    data = json.dumps(legacy_chat.response(json.loads(data)), ensure_ascii=False).encode()
                if inputs is not None and status < 400 and body.get('store', True):
                    app.state.store.save(request.state.owner, model, inputs, json.loads(data))
                await cleanup()
                return Response(data, status_code=status, headers=headers)

            async def stream():
                nonlocal stream_chunks
                pending = b''
                chunks = None
                try:
                    chunks = upstream.aiter_bytes()
                    if legacy:
                        chunks = legacy_chat.stream(chunks)
                    keepalive = settings.get('stream_keepalive_seconds', 0)
                    if keepalive > 0:
                        chunks = with_keepalive(chunks, keepalive)
                    stream_chunks = chunks
                    async for chunk in chunks:
                        timing.observe(chunk)
                        if inputs is not None and body.get('store', True):
                            pending = (pending + chunk).replace(b'\r\n', b'\n')
                            while b'\n\n' in pending:
                                event, pending = pending.split(b'\n\n', 1)
                                value = b'\n'.join(line[5:].lstrip() for line in event.split(b'\n') if line.startswith(b'data:'))
                                if value and value != b'[DONE]':
                                    parsed = json.loads(value)
                                    if parsed.get('type') in ('response.completed', 'response.incomplete', 'response.failed'):
                                        app.state.store.save(request.state.owner, model, inputs, parsed['response'])
                        yield chunk
                except httpx.HTTPError:
                    LOG.exception('stream backend failure id=%s', request.state.request_id)
                    error = {'error': {'message': 'Backend stream interrupted', 'type': 'server_error', 'code': 'backend_interrupted'}}
                    yield ('event: error\ndata: ' + json.dumps(error) + '\n\n').encode()
                finally:
                    await cleanup()
            return ManagedStreamingResponse(stream(), status_code=status, headers=headers,
                                            cleanup=cleanup, disconnect=close_upstream)
        except ProtocolError as error:
            await cleanup()
            return api_error(str(error), error.status, error.param)
        except ClientGone:
            await cleanup()
            return Response(status_code=499)
        except httpx.TimeoutException:
            if held:
                await asyncio.shield(backend.stop())
            await cleanup()
            return api_error('Inference request timed out', 504, code='backend_timeout')
        except (httpx.HTTPError, RuntimeError, OSError):
            LOG.exception('backend request failed id=%s', request.state.request_id)
            await cleanup()
            return api_error('Inference backend unavailable; retry the request', 503, code='backend_unavailable')
        except BaseException:
            await cleanup()
            raise

    @app.post('/v1/chat/completions')
    async def chat(request: Request):
        return await proxy(request, '/v1/chat/completions')

    @app.post('/v1/completions')
    async def completion(request: Request):
        return await proxy(request, '/v1/completions')

    @app.post('/v1/responses')
    async def responses(request: Request):
        return await proxy(request, '/v1/responses')

    @app.get('/v1/responses/{rid}')
    async def get_response(rid: str, request: Request):
        try:
            return app.state.store.get(rid, request.state.owner)[2]
        except ProtocolError as error:
            return api_error(str(error), error.status)

    @app.delete('/v1/responses/{rid}')
    async def delete_response(rid: str, request: Request):
        try:
            app.state.store.delete(rid, request.state.owner)
            return {'id': rid, 'object': 'response.deleted', 'deleted': True}
        except ProtocolError as error:
            return api_error(str(error), error.status)

    @app.get('/v1/responses/{rid}/input_items')
    async def response_inputs(rid: str, request: Request, limit: int = 20, order: str = 'desc', after: str = None, before: str = None):
        try:
            items = app.state.store.get(rid, request.state.owner)[1]
            if limit < 1 or limit > 100 or order not in ('asc', 'desc'):
                raise ProtocolError('limit must be 1..100 and order must be asc or desc')
            items = [dict(item, id=item.get('id', f'{rid}_input_{i}'), type=item.get('type', 'message')) for i, item in enumerate(items)]
            for item in items:
                if item.get('type') == 'message' and isinstance(item.get('content'), str):
                    content = {'type': 'output_text' if item.get('role') == 'assistant' else 'input_text',
                               'text': item['content']}
                    if content['type'] == 'output_text':
                        content['annotations'] = []
                    item['content'] = [content]
            if order == 'desc':
                items.reverse()
            if after:
                ids = [item['id'] for item in items]
                if after not in ids:
                    raise ProtocolError('after item not found', 404)
                items = items[ids.index(after) + 1:]
            if before:
                ids = [item['id'] for item in items]
                if before not in ids:
                    raise ProtocolError('before item not found', 404)
                items = items[:ids.index(before)]
            selected = items[:limit]
            return {'object': 'list', 'data': selected, 'has_more': len(items) > limit,
                    'first_id': selected[0]['id'] if selected else None, 'last_id': selected[-1]['id'] if selected else None}
        except ProtocolError as error:
            return api_error(str(error), error.status)

    @app.get('/metrics')
    async def metrics():
        status = app.state.backend.status()
        text = ''.join(f'swift_gateway_{key} {int(value)}\n' for key, value in status.items() if isinstance(value, (int, bool)))
        return Response(text, media_type='text/plain; version=0.0.4')

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.local.json')
    args = parser.parse_args()
    settings = load_json(args.config)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    for label, url in discover_addresses(settings.get('port', 8080), settings.get('host', '0.0.0.0')).items():
        LOG.info('API URL (%s): %s', label, url)
    uvicorn.run(create_app(settings), host=settings.get('host', '0.0.0.0'), port=settings.get('port', 8080),
                workers=1, timeout_keep_alive=30, timeout_graceful_shutdown=60, access_log=False)


if __name__ == '__main__':
    main()
