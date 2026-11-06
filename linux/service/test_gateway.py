"""Protocol/state/error tests independent of model quality and GPU timing."""
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import json
import sys
import socket
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from anyio import CancelScope
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gateway import BackendManager, ClientGone, ProtocolError, ResponseStore, create_app, with_keepalive, close_stream
from profiles import Q5, FLASH


class FakeManager(BackendManager):
    async def ensure(self, model, client):
        self.active_model, self.url = model, 'http://backend'
        self.process = SimpleNamespace(returncode=None)
        self.ready = True

    async def stop(self):
        self.ready = False
        self.process = None


@asynccontextmanager
async def fixture(tmp_path, extra_settings=None):
    key = tmp_path / 'key'
    key.write_text('test-secret')
    extra = tmp_path / 'extra'
    extra.write_text(json.dumps({'second': 'second-secret'}))
    seen = []
    fake = FastAPI()

    @fake.get('/health')
    async def health():
        return {'status': 'ok'}

    @fake.post('/v1/chat/completions')
    async def chat(request: Request):
        body = await request.json()
        seen.append(body)
        if body.get('test_backend_error'):
            return JSONResponse({'error': {'message': 'Unsupported parameter', 'type': 'invalid_request_error'}}, status_code=400)
        if body.get('test_legacy_response'):
            value = {'id': 'chat_legacy', 'choices': [{'index': 0,
                'message': {'role': 'assistant', 'tool_calls': [{'id': 'call_a', 'type': 'function',
                    'function': {'name': 'increment', 'arguments': '{"value":17}'}}]}, 'finish_reason': 'tool_calls'}]}
            if body.get('stream'):
                value['choices'][0]['delta'] = value['choices'][0].pop('message')
                async def events():
                    raw = ('data: ' + json.dumps(value) + '\n\ndata: [DONE]\n\n').encode()
                    for pos in range(0, len(raw), 11):
                        yield raw[pos:pos + 11]
                return StreamingResponse(events(), media_type='text/event-stream')
            return value
        return {'id': 'chat_' + str(len(seen)), 'model': body['model'], 'object': 'chat.completion',
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'OK'}, 'finish_reason': 'stop'}]}

    @fake.post('/v1/responses')
    async def responses(request: Request):
        body = await request.json()
        seen.append(body)
        response = {'id': 'resp_' + str(len(seen)), 'model': body['model'], 'object': 'response',
                    'status': 'completed', 'output': [{'id': 'msg_' + str(len(seen)), 'type': 'message',
                    'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'OK', 'annotations': []}]}]}
        if not body.get('stream'):
            return response
        async def events():
            yield b'event: response.created\ndata: {"type":"response.created"}\n\n'
            text = 'event: response.completed\ndata: ' + json.dumps({'type': 'response.completed', 'response': response}) + '\n\n'
            # Deliberate splits across event and JSON boundaries.
            for pos in range(0, len(text), 13):
                yield text[pos:pos + 13].encode()
        return StreamingResponse(events(), media_type='text/event-stream')

    manager = FakeManager(tmp_path)
    settings = {'state_directory': str(tmp_path), 'api_key_file': str(key),
                'additional_api_keys_file': str(extra), 'warm_start': False}
    settings.update(extra_settings or {})
    app = create_app(settings, manager, httpx.ASGITransport(app=fake))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway',
                                     headers={'Authorization': 'Bearer test-secret'}) as client:
            yield client, seen, manager, app


@pytest.mark.asyncio
async def test_cache_expiration_waits_for_idle_backend(tmp_path):
    import os
    async with fixture(tmp_path, {'kv_cache_cleanup_interval_seconds': 1}) as (_, _, manager, _):
        await manager.acquire()
        namespace = tmp_path / 'q5-ssd-cache' / ('a' * 64)
        namespace.mkdir(parents=True)
        meta = namespace / '123-1.json'
        payload = namespace / '123-1.bin'
        meta.write_text(json.dumps({'version': 1, 'identity': namespace.name, 'bytes': 7}))
        payload.write_bytes(b'payload')
        os.utime(meta, (0, 0))
        try:
            await asyncio.sleep(1.1)
            assert meta.exists() and payload.exists()
        finally:
            manager.release()
        for _ in range(30):
            if not meta.exists():
                break
            await asyncio.sleep(.1)
        assert not meta.exists() and not payload.exists()
        audit = json.loads((tmp_path / 'cache-retention.json').read_text())
        assert audit['retention_seconds'] == 86400 and not audit['errors']


@pytest.mark.asyncio
async def test_auth_models_and_bad_requests_recover(tmp_path):
    async with fixture(tmp_path) as (client, seen, manager, app):
        assert (await client.get('/health', headers={'Authorization': ''})).status_code == 200
        assert (await client.get('/v1/models', headers={'Authorization': ''})).status_code == 401
        assert {item['id'] for item in (await client.get('/v1/models')).json()['data']} == {Q5, FLASH}
        assert (await client.get('/v1/models/' + Q5)).json()['id'] == Q5
        assert (await client.get('/v1/models/missing')).status_code == 404
        for retired in ('unknown-model', 'unsupported-model', 'missing-alias', 'unregistered'):
            assert (await client.get('/v1/models/' + retired)).status_code == 404
            assert (await client.post('/v1/chat/completions', json={'model': retired, 'messages': []})).status_code == 404
        for body, code in (([], 400), ({'model': {}}, 400), ({'model': 'missing'}, 404)):
            # Empty dict follows the documented default model; use nonempty for a bad type.
            if body == {'model': {}}:
                body = {'model': {'bad': 1}}
            response = await client.post('/v1/chat/completions', json=body)
            assert response.status_code == code
            assert 'error' in response.json()
        response = await client.post('/v1/chat/completions', content=b'{broken')
        assert response.status_code == 400
        assert (await client.get('/unknown')).json()['error']
        response = await client.post('/v1/chat/completions', json={'model': Q5, 'messages': [{'role': 'user', 'content': 'hello'}]})
        assert response.status_code == 200 and not manager.lock.locked()


@pytest.mark.asyncio
async def test_lazy_start_and_request_selected_model(tmp_path):
    async with fixture(tmp_path) as (client, seen, manager, app):
        ready = (await client.get('/ready')).json()
        assert ready['request_ready'] and ready['status'] == 'idle' and not ready['model_ready']
        assert manager.active_model is None
        assert (await client.get('/v1/models')).status_code == 200
        assert manager.active_model is None
        for model in (FLASH, Q5, FLASH):
            response = await client.post('/v1/chat/completions', json={'model': model, 'messages': [{'role': 'user', 'content': 'hi'}]})
            assert response.status_code == 200 and manager.active_model == model
            assert seen[-1]['model'] == model


@pytest.mark.asyncio
async def test_request_reasoning_controls_both_APIs_and_profiles(tmp_path):
    async with fixture(tmp_path) as (client, seen, manager, app):
        for model in (Q5, FLASH):
            for responses in (False, True):
                for effort, expected in [('none', 'none'), ('minimal', 'low'), ('low', 'low'),
                                          ('medium', 'medium'), ('high', 'xhigh'), ('xhigh', 'xhigh')]:
                    body = {'model': model, 'metadata': {'keep': 'exact'}}
                    if responses:
                        body.update(input='hi', reasoning={'effort': effort, 'summary': 'auto'})
                    else:
                        body.update(messages=[{'role': 'user', 'content': 'hi'}], reasoning_effort=effort)
                    path = '/v1/responses' if responses else '/v1/chat/completions'
                    assert (await client.post(path, json=body)).status_code == 200
                    kwargs = seen[-1]['chat_template_kwargs']
                    assert kwargs == {'reasoning_effort': expected, 'enable_thinking': effort != 'none'}
                    assert seen[-1]['metadata'] == body['metadata']
                    if responses:
                        assert seen[-1]['reasoning'] == {'effort': expected, 'summary': 'auto'}
        body = {'messages': [{'role': 'user', 'content': 'hi'}], 'reasoning_effort': 'high',
                'chat_template_kwargs': {'enable_thinking': False, 'preserve_thinking': False}}
        assert (await client.post('/v1/chat/completions', json=body)).status_code == 200
        assert seen[-1]['chat_template_kwargs'] == {'enable_thinking': False, 'preserve_thinking': False, 'reasoning_effort': 'xhigh'}
        body.update(reasoning_effort='none', chat_template_kwargs={'enable_thinking': True})
        assert (await client.post('/v1/chat/completions', json=body)).status_code == 200
        assert not seen[-1]['chat_template_kwargs']['enable_thinking']
        body.update(reasoning_effort='ultra')
        assert (await client.post('/v1/chat/completions', json=body)).status_code == 400
        assert not manager.lock.locked()


@pytest.mark.asyncio
async def test_preserves_roles_order_fields_and_tool_ids(tmp_path):
    async with fixture(tmp_path) as (client, seen, manager, app):
        messages = [{'role': role, 'content': f' {i}\n'} for i, role in enumerate(('system', 'user', 'system', 'developer', 'user', 'assistant', 'assistant'))]
        messages += [{'role': 'assistant', 'tool_calls': [{'id': 'a', 'type': 'function', 'function': {'name': 'f', 'arguments': '{"x":1}'}},
                                                         {'id': 'b', 'type': 'function', 'function': {'name': 'f', 'arguments': '{"x":2}'}}]},
                     {'role': 'tool', 'tool_call_id': 'b', 'content': 'second'},
                     {'role': 'tool', 'tool_call_id': 'a', 'content': 'first'}]
        body = {'model': 'swift-q5', 'messages': messages, 'metadata': {'agent': 'run'}, 'parallel_tool_calls': True,
                'response_format': {'type': 'json_object'}}
        assert (await client.post('/v1/chat/completions', json=body)).status_code == 200
        assert seen[-1] == dict(body, model=Q5)
        assert not manager.lock.locked()


@pytest.mark.asyncio
async def test_backend_error_releases_execution_queue(tmp_path):
    async with fixture(tmp_path) as (client, seen, manager, app):
        assert (await client.post('/v1/chat/completions', json={'test_backend_error': True})).status_code == 400
        assert not manager.lock.locked()
        assert (await client.post('/v1/chat/completions', json={})).status_code == 200


@pytest.mark.asyncio
async def test_deprecated_functions_proxy_json_and_fragmented_stream(tmp_path):
    async with fixture(tmp_path) as (client, seen, manager, app):
        body = {'functions': [{'name': 'increment', 'parameters': {}}],
                'function_call': {'name': 'increment'}, 'messages': [{'role': 'user', 'content': 'call it'}],
                'test_legacy_response': True}
        response = await client.post('/v1/chat/completions', json=body)
        assert response.status_code == 200
        assert response.json()['choices'][0]['message']['function_call']['name'] == 'increment'
        assert response.json()['choices'][0]['finish_reason'] == 'function_call'
        assert seen[-1]['tool_choice']['function']['name'] == 'increment'
        response = await client.post('/v1/chat/completions', json=dict(body, stream=True))
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ') and line != 'data: [DONE]']
        assert events[0]['choices'][0]['delta']['function_call']['arguments'] == '{"value":17}'
        assert not manager.lock.locked()
        assert (await client.post('/v1/chat/completions', json={'functions': 1})).status_code == 400


@pytest.mark.asyncio
async def test_response_chain_owner_model_and_instructions(tmp_path):
    async with fixture(tmp_path) as (client, seen, manager, app):
        first = (await client.post('/v1/responses', json={'model': Q5, 'instructions': 'first instructions', 'input': 'first'})).json()
        rid = first['id']
        assert (await client.get('/v1/responses/' + rid)).json() == first
        assert (await client.get('/v1/responses/' + rid, headers={'Authorization': 'Bearer second-secret'})).status_code == 404
        cross = await client.post('/v1/responses', json={'model': FLASH, 'previous_response_id': rid, 'input': 'next'})
        assert cross.status_code == 200 and seen[-1]['model'] == FLASH
        assert seen[-1]['input'][0]['content'] == 'first'
        assert (await client.post('/v1/responses', json={'previous_response_id': {'bad': 1}, 'input': 'next'})).status_code == 400
        continued = await client.post('/v1/responses', json={'model': Q5, 'previous_response_id': rid,
                                                            'instructions': 'new instructions', 'input': 'next'})
        assert continued.status_code == 200
        assert seen[-1]['input'] == [{'role': 'user', 'content': 'first'}, *first['output'], {'role': 'user', 'content': 'next'}]
        assert seen[-1]['instructions'] == 'new instructions'
        assert 'previous_response_id' not in seen[-1]


@pytest.mark.asyncio
async def test_stream_response_persistence_and_opt_out(tmp_path):
    async with fixture(tmp_path) as (client, seen, manager, app):
        stream = await client.post('/v1/responses', json={'input': 'stream', 'stream': True})
        assert stream.status_code == 200 and 'response.completed' in stream.text
        assert (await client.get('/v1/responses/resp_1')).status_code == 200
        assert not manager.lock.locked()
        hidden = (await client.post('/v1/responses', json={'input': 'private', 'store': False})).json()
        assert (await client.get('/v1/responses/' + hidden['id'])).status_code == 404
        assert (await client.delete('/v1/responses/resp_1')).json()['deleted'] is True
        assert (await client.get('/v1/responses/resp_1')).status_code == 404


@pytest.mark.asyncio
async def test_item_references_and_input_pagination(tmp_path):
    async with fixture(tmp_path) as (client, seen, manager, app):
        first = (await client.post('/v1/responses', json={'input': 'a'})).json()
        response = await client.post('/v1/responses', json={'input': [{'type': 'item_reference', 'id': first['output'][0]['id']}, {'role': 'user', 'content': 'b'}]})
        assert response.status_code == 200
        assert seen[-1]['input'][0] == first['output'][0]
        rid = response.json()['id']
        page = (await client.get('/v1/responses/' + rid + '/input_items?limit=1&order=asc')).json()
        assert len(page['data']) == 1 and page['has_more']
        next_page = (await client.get('/v1/responses/' + rid + '/input_items', params={'order': 'asc', 'after': page['last_id']})).json()
        assert len(next_page['data']) == 1 and next_page['data'][0]['content'] == [{'type': 'input_text', 'text': 'b'}]
        original_inputs = (await client.get('/v1/responses/' + first['id'] + '/input_items?order=asc')).json()
        user_id = original_inputs['data'][0]['id']
        referenced = await client.post('/v1/responses', json={'input': [{'type': 'item_reference', 'id': user_id}]})
        assert referenced.status_code == 200 and seen[-1]['input'][0]['content'] == 'a'
        forbidden = await client.post('/v1/responses', json={'input': [{'type': 'item_reference', 'id': user_id}]},
                                      headers={'Authorization': 'Bearer second-secret'})
        assert forbidden.status_code == 404
        assert (await client.get('/v1/responses/' + rid + '/input_items?limit=not_a_number')).status_code == 400


@pytest.mark.asyncio
async def test_cancelled_and_disconnected_waiters_do_not_steal_lock(tmp_path):
    manager = BackendManager(tmp_path)
    await manager.acquire()
    class Gone:
        async def is_disconnected(self):
            return True
    with pytest.raises(ClientGone):
        await manager.acquire(Gone())
    assert manager.lock.locked() and manager.queued == 0
    task = asyncio.create_task(manager.acquire())
    await asyncio.sleep(.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager.lock.locked() and manager.queued == 0
    manager.release()
    await asyncio.wait_for(manager.acquire(), 1)
    manager.release()


def test_store_restart_and_long_chain_without_turn_gate(tmp_path):
    path = tmp_path / 'responses.sqlite3'
    store = ResponseStore(path)
    previous = None
    for turn in range(150):
        body = {'input': f'turn {turn}'}
        if previous:
            body['previous_response_id'] = previous
        _, inputs = store.prepare(body, 'owner', Q5)
        previous = 'r_' + str(turn)
        store.save('owner', Q5, inputs, {'id': previous, 'output': [{'role': 'assistant', 'content': 'OK'}]})
    assert len(inputs) == 299
    store.close()
    store = ResponseStore(path)
    assert len(store.get(previous, 'owner')[1]) == 299
    with pytest.raises(ProtocolError):
        store.get(previous, 'someone-else')
    store.close()


@pytest.mark.asyncio
async def test_sse_keepalive_preserves_frames_and_closes_cancelled_reader():
    closed = asyncio.Event()
    async def delayed():
        try:
            await asyncio.sleep(.035)
            yield b'data: {"content":"hello"}\n\n'
            yield b'data: [DONE]\n\n'
        finally:
            closed.set()
    frames = [frame async for frame in with_keepalive(delayed(), .005)]
    assert b': keep-alive\n\n' in frames
    assert [frame for frame in frames if frame.startswith(b'data:')] == [
        b'data: {"content":"hello"}\n\n', b'data: [DONE]\n\n']
    assert closed.is_set()
    closed.clear()
    reader = with_keepalive(delayed(), .005)
    assert await anext(reader) == b': keep-alive\n\n'
    await reader.aclose()
    assert closed.is_set()


@pytest.mark.asyncio
async def test_cancelled_asgi_scope_finishes_reader_and_socket_cleanup():
    closed = []
    class Reader:
        async def aclose(self):
            await asyncio.sleep(.005)
            closed.append('reader')
    async def cleanup():
        await asyncio.sleep(.005)
        closed.append('socket')
    with CancelScope() as scope:
        scope.cancel()
        await close_stream(Reader(), cleanup)
    assert closed == ['reader', 'socket']


def test_real_socket_disconnect_cancels_native_reader_before_next_request(tmp_path, monkeypatch):
    closed = threading.Event()
    native = FastAPI()
    native_lock = asyncio.Lock()
    emitted = []
    events = []
    import gateway as gateway_module
    class TracedResponse(gateway_module.ManagedStreamingResponse):
        async def listen_for_disconnect(self, receive):
            events.append('gateway-listen')
            async def traced():
                value = await receive()
                events.append('gateway-received-' + value['type'])
                return value
            return await super().listen_for_disconnect(traced)
        async def __call__(self, *args):
            events.append('gateway-response-enter')
            try:
                return await super().__call__(*args)
            finally:
                events.append('gateway-response-exit')
    monkeypatch.setattr(gateway_module, 'ManagedStreamingResponse', TracedResponse)

    @native.get('/health')
    async def health():
        return {'status': 'ok'}

    @native.post('/v1/chat/completions')
    async def completion(request: Request):
        body = await request.json()
        await native_lock.acquire()
        if not body.get('stream'):
            native_lock.release()
            return {'choices': [{'message': {'role': 'assistant', 'content': '42'}}]}
        async def tokens():
            try:
                for number in range(2048):
                    emitted.append(number)
                    yield b'data: {"choices":[{"index":0,"delta":{"content":"x"}}]}\n\n'
                    await asyncio.sleep(.01)
            finally:
                native_lock.release()
                closed.set()
                events.append('native-reader-closed')
        return StreamingResponse(tokens(), media_type='text/event-stream')

    servers = []
    def launch(app):
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(128)
        address = 'http://127.0.0.1:' + str(listener.getsockname()[1])
        server = uvicorn.Server(uvicorn.Config(app, log_level='error', timeout_graceful_shutdown=1))
        thread = threading.Thread(target=lambda: asyncio.run(server.serve(sockets=[listener])), daemon=True)
        servers.append((server, thread, listener))
        thread.start()
        until = time.monotonic() + 5
        while not server.started:
            assert thread.is_alive() and time.monotonic() < until
            time.sleep(.01)
        return address

    try:
        async def native_disconnect_aware(scope, receive, send):
            # The fixture models a native server that observes socket closure.
            # Starlette's ASGI2.4 body-finalization behavior is tested on the
            # gateway side, rather than accidentally duplicated in the fixture.
            if scope['type'] == 'http':
                scope = dict(scope, asgi=dict(scope.get('asgi', {}), spec_version='2.3'))
            await native(scope, receive, send)
        native_url = launch(native_disconnect_aware)
        class Manager(FakeManager):
            async def ensure(self, model, client):
                self.active_model, self.url = model, native_url
                self.process = SimpleNamespace(returncode=None)
                self.ready = True
        keyfile = tmp_path / 'key'
        keyfile.write_text('socket-test-key')
        settings = dict(state_directory=str(tmp_path / 'state'), api_key_file=str(keyfile),
                        default_model=Q5, native_keepalive_connections=0, stream_keepalive_seconds=10)
        manager = Manager(tmp_path / 'state')
        url = launch(create_app(settings, manager))
        with httpx.Client(base_url=url, headers={'Authorization': 'Bearer socket-test-key'},
                          trust_env=False, timeout=3) as client:
            body = dict(model=Q5, messages=[{'role': 'user', 'content': 'count'}], stream=True)
            with client.stream('POST', '/v1/chat/completions', json=body) as reply:
                reply.raise_for_status()
                for line in reply.iter_lines():
                    if line.startswith('data:'):
                        break
            began = time.monotonic()
            body['stream'] = False
            try:
                reply = client.post('/v1/chat/completions', json=body)
            except Exception:
                print('socket cancellation diagnostic:', dict(events=events, emitted=len(emitted),
                      native_closed=closed.is_set(), gateway=manager.status()))
                raise
            reply.raise_for_status()
            assert reply.json()['choices'][0]['message']['content'] == '42'
            assert time.monotonic() - began < 2
            assert closed.wait(1)
            assert len(emitted) < 2048
    finally:
        for server, _, _ in servers:
            server.should_exit = True
        for _, thread, listener in servers:
            thread.join(5)
            listener.close()


@pytest.mark.asyncio
async def test_model_only_activation_never_submits_a_prompt(tmp_path):
    async with fixture(tmp_path) as (client, seen, manager, app):
        assert (await client.post('/v1/models/' + FLASH + '/load', headers={'Authorization':''})).status_code == 401
        for _ in range(2):
            r = await client.post('/v1/models/' + FLASH + '/load', json={})
            assert r.status_code == 200 and r.json() == {'model': FLASH, 'loaded': True}
            assert manager.active_model == FLASH and not manager.lock.locked()
            assert seen == []
        assert (await client.post('/v1/models/' + 'unknown-model' + '/load')).status_code == 404
