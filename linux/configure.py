#!/usr/bin/env python3
"""Create local-only settings and credentials; optionally install a user unit."""
from pathlib import Path
import argparse
import json
import os
import secrets
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'service'))
from settings import load_json


def initialize(root=ROOT):
    for name in ('config', 'engine-profiles'):
        destination = root / (name + '.local.json')
        if not destination.exists():
            with destination.open('x') as stream:
                stream.write((root / (name + '.example.json')).read_text())
            destination.chmod(0o600)
    settings = load_json(root / 'config.local.json', root=root)
    state = Path(settings['state_directory'])
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    state.chmod(0o700)
    key = Path(settings['api_key_file'])
    if not key.exists():
        fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            stream.write(secrets.token_urlsafe(48) + '\n')
    print('Local settings prepared; existing settings and credentials preserved.')


def install_unit():
    python = ROOT / '.venv/bin/python'
    if not python.exists():
        raise SystemExit('Create linux/.venv and install service dependencies first.')
    # systemd quoted arguments still expand percent specifiers.
    def quote(value):
        return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%') + '"'
    unit = f'''[Unit]
Description=Local LLM gateway
After=network-online.target
[Service]
Type=notify
NotifyAccess=main
WatchdogSec=30
WorkingDirectory={quote(ROOT)}
ExecStart={quote(python)} {quote(ROOT / 'service/gateway.py')} --config {quote(ROOT / 'config.local.json')}
Environment=PYTHONUNBUFFERED=1
Restart=on-failure
RestartSec=3
KillMode=mixed
TimeoutStopSec=65
UMask=0077
[Install]
WantedBy=default.target
'''
    path = Path.home() / '.config/systemd/user/llm-service.service'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(unit)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    print('User unit installed; service has not been started or enabled.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-systemd', action='store_true')
    args = parser.parse_args()
    initialize()
    if args.install_systemd:
        install_unit()
