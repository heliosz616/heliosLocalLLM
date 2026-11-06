#!/usr/bin/python3
"""Human controls for the deployed user service; no inference-engine changes."""
from pathlib import Path
import argparse
import json
import os
import re
import shutil
from datetime import datetime
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

from addresses import discover as discover_addresses
from settings import load_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config.local.json'
UNIT = 'llm-service.service'
MODELS = {
    'qwen3.8-flash': 'Retained Qwen3.8 Flash-Next UD-IQ4_XS',
    'qwen3.8-27b': 'Swift Q5_K_S',
}
ALIASES = {'qwen-swift': 'qwen3.8-27b', 'qwen-flash-next': 'qwen3.8-flash',
           'swift-1.5-q5ks': 'qwen3.8-27b', 'q5': 'qwen3.8-27b', 'flash': 'qwen3.8-flash',
           **{k: k for k in MODELS}}
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def command(args, timeout=10, check=True):
    result = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or 'Command failed')
    return result.stdout.strip()


def config():
    return load_json(CONFIG)


def api_key():
    return Path(config()['api_key_file']).read_text().strip()


def request(url, authenticated=False):
    headers = {'Authorization': 'Bearer ' + api_key()} if authenticated else {}
    with HTTP.open(urllib.request.Request(url, headers=headers), timeout=5) as response:
        return json.load(response)


def service_properties():
    value = command(['systemctl', '--user', 'show', UNIT,
                     '-p', 'ActiveState', '-p', 'SubState', '-p', 'UnitFileState'], check=False)
    return dict(line.split('=', 1) for line in value.splitlines() if '=' in line)


def status():
    props = service_properties()
    data = {'service': props.get('ActiveState', 'unknown'),
            'autostart': props.get('UnitFileState') in ('enabled', 'enabled-runtime'),
            'default_model': config()['default_model'], 'ready': False}
    if data['service'] in ('active', 'activating'):
        try:
            data['health'] = request(f"http://127.0.0.1:{config()['port']}/ready")
            data['ready'] = bool(data['health'].get('request_ready', data['health'].get('model_ready')))
        except urllib.error.HTTPError as error:
            try:
                data['health'] = json.load(error)
            except (ValueError, OSError):
                data['detail'] = str(error)
        except (OSError, ValueError) as error:
            data['detail'] = str(error)
    return data


def connections():
    value = config()
    return discover_addresses(value['port'], value.get('host', '0.0.0.0'))


def print_connections():
    print('\nCurrent API URLs:')
    for label, url in connections().items():
        print('  ' + label + ': ' + url)


def check_connection(url):
    ids = sorted(item['id'] for item in request(url.rstrip('/') + '/models', True)['data'])
    if not set(MODELS).issubset(ids):
        raise RuntimeError('The API answered, but one of the deployed models is missing.')
    return 'Authenticated API check passed. Registered models: ' + ', '.join(ids)


def print_storage():
    usage = shutil.disk_usage(ROOT)
    def size(value):
        return f'{value / 1e9:.1f} GB ({value / 1024**3:.1f} GiB)'
    print('Disk total:', size(usage.total))
    print('Disk used: ', size(usage.used))
    print('Disk free: ', size(usage.free))
    print('Total SSD KV + recurrent-state limit:', size(config().get('kv_cache_maximum_bytes', 64_000_000_000)))
    if (ROOT / 'runtime/recurrent-checkpoints-active.json').exists():
        print('Periodic recurrent checkpoints: minimum 32K spacing at existing batch boundaries; at most 6 per session')
        print('Hot forward sessions retain GPU KV; periodic checkpoint files contain no ordinary attention KV')
    print('KV expiration:', config().get('kv_cache_retention_hours', 24), 'hours since last SSD use')
    print('KV cleanup interval:', config().get('kv_cache_cleanup_interval_seconds', 300), 'seconds (when idle)')
    state = Path(config()['state_directory'])
    for label, path in [('Model files', ROOT / 'models'),
                        ('Swift KV cache', state / 'q5-ssd-cache'),
                        ('Flash KV cache', state / 'flash-next-ssd-cache')]:
        if path.exists():
            output = command(['du', '-s', '-B1', str(path)], timeout=30, check=False)
            number = output.split(maxsplit=1)[0] if output else ''
            print(label + ':', size(int(number)) if number.isdecimal() else 'unavailable', '—', path)
    try:
        budget = json.loads((state / 'cache-budget.json').read_text())
    except (OSError, ValueError):
        return
    print('Last activated model:', budget['model'])
    print('Its SSD cache allowance:', size(budget['effective_maximum_bytes']))
    print('Disk reserve at activation:', size(budget['disk_reserve_bytes']))


def control(action):
    if action not in ('start', 'stop', 'restart'):
        raise ValueError('Unknown service action')
    command(['systemctl', '--user', action, UNIT], timeout=930)
    return {'start': 'Service started.', 'stop': 'Service stopped.',
            'restart': 'Service restarted.'}[action]


def autostart(enabled):
    command(['systemctl', '--user', 'enable' if enabled else 'disable', UNIT])
    return 'Automatic startup ' + ('enabled.' if enabled else 'disabled.')


def load_model(name):
    """Activate through the normal request queue; preserve an already hot model."""
    model = ALIASES[name]
    if service_properties().get('ActiveState') not in ('active', 'activating'):
        print(control('start'), flush=True)
    value = config()
    address = f"http://127.0.0.1:{value['port']}"
    try:
        ready = request(address + '/ready')
    except urllib.error.HTTPError as error:
        if error.code != 503:
            raise
        # A backend which is loading or recovering must still accept a queued
        # activation request; /ready deliberately reports 503 during that time.
        ready = json.load(error)
    if ready.get('active_model') == model and ready.get('model_ready'):
        return model + ' is already loaded; its current session is preserved.'
    payload = b'{}'  # Load weights only; never synthesize a prompt or replace a session.
    headers = {'Authorization': 'Bearer ' + api_key(), 'Content-Type': 'application/json'}
    timeout = value.get('request_timeout_seconds', 3600) + value.get('startup_timeout_seconds', 900) + 60
    with HTTP.open(urllib.request.Request(address + '/v1/models/' + model + '/load', data=payload,
                                         headers=headers, method='POST'), timeout=timeout) as response:
        response.read()
    ready = request(address + '/ready')
    if ready.get('active_model') != model or not ready.get('model_ready'):
        raise RuntimeError('Activation completed but the requested model is no longer loaded; check llm-service status.')
    return model + ' is loaded. API requests can select another model without changing the default.'


def default_model(model):
    model = ALIASES[model]
    value = config()
    was_running = service_properties().get('ActiveState') in ('active', 'activating')
    if value['default_model'] == model:
        return MODELS[model] + ' is already the default.'
    previous = CONFIG.read_bytes()
    value['default_model'] = model
    mode = CONFIG.stat().st_mode & 0o777
    def save(contents):
        fd, name = tempfile.mkstemp(prefix='.swift-config-', dir=CONFIG.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(contents)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(name, mode)
            os.replace(name, CONFIG)
        finally:
            if os.path.exists(name):
                os.unlink(name)
    save((json.dumps(value, indent=2) + '\n').encode())
    if was_running:
        try:
            control('restart')
        except Exception as original:
            save(previous)
            try:
                control('restart')
            except Exception as recovery:
                raise RuntimeError(f'Model change failed: {original}\nPrevious default restored; recovery failed: {recovery}') from original
            raise RuntimeError(f'Model change failed: {original}\nPrevious default and service restored.') from original
    return MODELS[model] + (' saved as default; gateway restarted. Models load on request.' if was_running else ' saved as default; service remains stopped.')


def main():
    parser = argparse.ArgumentParser(prog='llm-service', description='Manage local Qwen inference from the terminal.')
    sub = parser.add_subparsers(dest='action')
    help_text = {
        'start': 'Start serving', 'stop': 'Stop serving and release GPU memory',
        'restart': 'Restart serving (interrupts active requests)',
        'status': 'Show readiness, loaded model, queue and automatic startup',
        'info': 'Show Tailscale/LAN URLs, model IDs and key location',
        'url': 'Print the preferred API base URL', 'models': 'List model IDs',
        'key': 'Print the API key', 'check': 'Check the authenticated API',
        'storage': 'Show disk space, model/cache sizes and the latest cache allowance',
    }
    for name, description in help_text.items():
        sub.add_parser(name, help=description)
    history = sub.add_parser('history', help='Recent requests with gateway first-token and total time')
    history.add_argument('-n', '--lines', type=int, default=20)
    logs = sub.add_parser('logs', help='Follow live logs; Ctrl+C exits')
    logs.add_argument('-n', '--lines', type=int, default=100, help='Recent log lines to show first (default: 100)')
    logs.add_argument('--no-follow', action='store_true', help='Print recent logs and exit')
    model = sub.add_parser('default-model', help='Optional fallback when a client omits model; clients normally choose per request')
    model.add_argument('model', nargs='?', choices=ALIASES, metavar='{flash,q5}', help='Use flash or q5 (full model IDs also accepted)')
    load = sub.add_parser('load', help='Load a model now through the request queue; preserve an already loaded model')
    load.add_argument('model', nargs='?', choices=ALIASES, help='Model ID or alias; omitted loads the configured default')
    auto = sub.add_parser('autostart', help='Show/change automatic startup without starting or stopping now')
    auto.add_argument('setting', nargs='?', choices=('on', 'off'))
    args = parser.parse_args()
    try:
        if args.action in (None, 'status'):
            value = status()
            print('Service:          ', value['service'])
            print('Ready:            ', 'yes' if value['ready'] else 'no')
            print('Automatic startup:', 'on' if value['autostart'] else 'off')
            print('Default model:    ', value['default_model'])
            print('Load on request:  ', 'yes' if not config().get('warm_start', False) else 'no')
            if value.get('health'):
                print('Loaded model:     ', value['health'].get('active_model') or 'none')
                print('Request running:  ', 'yes' if value['health'].get('request_running') else 'no')
                print('Queued requests:  ', value['health'].get('queued_requests', 0))
            elif value.get('detail'):
                print('Detail:           ', value['detail'])
            if args.action is None:
                print('\nCommands: llm-service start | stop | restart | logs | info | --help')
        elif args.action in ('start', 'stop', 'restart'):
            print({'start': 'Starting service…', 'stop': 'Stopping service…', 'restart': 'Restarting service…'}[args.action], flush=True)
            print(control(args.action))
            if args.action in ('start', 'restart'):
                print_connections()
        elif args.action == 'info':
            for name, url in connections().items():
                print(name + ':', url)
            for model, label in MODELS.items():
                print(label + ':', model)
            print('API key file:', config()['api_key_file'])
            print('Show the key: llm-service key')
        elif args.action == 'url':
            print(next(iter(connections().values())))
        elif args.action == 'models':
            print('\n'.join(MODELS))
        elif args.action == 'key':
            print(api_key())
        elif args.action == 'check':
            url = next(iter(connections().values()))
            print(url)
            print(check_connection(url))
        elif args.action == 'storage':
            print_storage()
        elif args.action == 'load':
            model = args.model or config()['default_model']
            print('Loading ' + ALIASES[model] + ' (waits for any active request)…', flush=True)
            print(load_model(model))
            print_connections()
        elif args.action == 'default-model':
            if args.model:
                print('Saving fallback model; the gateway restarts if running…', flush=True)
                print(default_model(args.model))
                if service_properties().get('ActiveState') == 'active':
                    print_connections()
            else:
                print(config()['default_model'])
        elif args.action == 'autostart':
            if args.setting:
                print(autostart(args.setting == 'on'))
            else:
                print('on' if status()['autostart'] else 'off')
        elif args.action == 'history':
            if not 1 <= args.lines <= 1000:
                raise ValueError('History count must be between 1 and 1000.')
            raw = command(['journalctl', '--user', '-u', UNIT, '--no-pager', '-o', 'json', '--grep=request_done', '-n', str(args.lines)])
            print('Time                  Model                 First token  First text   Total')
            for line in raw.splitlines():
                entry = json.loads(line)
                message = entry.get('MESSAGE', '')
                match = re.search(r'model=(\S+) seconds=([\d.]+)', message)
                if not match:
                    continue
                timing = json.loads(message.split(' timing=', 1)[1]) if ' timing=' in message else {}
                def ms(name):
                    value = timing.get(name)
                    return f'{value:.1f} ms' if value is not None else 'unrecorded'
                stamp = datetime.fromtimestamp(int(entry['__REALTIME_TIMESTAMP']) / 1000000).strftime('%Y-%m-%d %H:%M:%S')
                print(f'{stamp}  {match[1]:21} {ms("first_token_ms"):>11}  {ms("first_text_ms"):>11}  {match[2]} s')
            print('First-token/text times are measured at the gateway; client network transit is excluded.')
        elif args.action == 'logs':
            if args.lines < 0:
                raise ValueError('Log line count must be zero or greater.')
            command_line = ['journalctl', '--user', '-u', UNIT, '-n', str(args.lines), '--no-pager']
            if not args.no_follow:
                command_line.append('-f')
            return subprocess.call(command_line)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
