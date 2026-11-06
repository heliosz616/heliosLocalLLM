"""Discover current client URLs without storing interface addresses."""
import json
import shutil
import subprocess


def _read(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=3, check=True)
    return json.loads(result.stdout)


def discover(port=8080, host='0.0.0.0'):
    if host not in ('0.0.0.0', '::', ''):
        address = f'[{host}]' if ':' in host else host
        return {'Configured host': f'http://{address}:{port}/v1'}
    urls = {}
    if shutil.which('tailscale'):
        try:
            ts = _read(['tailscale', 'status', '--json'])
            if ts.get('BackendState') == 'Running':
                own = ts.get('Self', {})
                ip = next((x for x in own.get('TailscaleIPs', []) if ':' not in x), None)
                if ip:
                    urls['Tailscale'] = f'http://{ip}:{port}/v1'
                name = own.get('DNSName', '').rstrip('.')
                if name:
                    urls['MagicDNS'] = f'http://{name}:{port}/v1'
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    try:
        for interface in _read(['ip', '-j', '-4', 'address', 'show', 'scope', 'global']):
            if interface['ifname'].startswith(('tailscale', 'docker', 'br-', 'veth', 'virbr')):
                continue
            for addr in interface.get('addr_info', []):
                if addr.get('family') == 'inet':
                    urls.setdefault('LAN', f"http://{addr['local']}:{port}/v1")
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    urls['This computer'] = f'http://127.0.0.1:{port}/v1'
    return urls
