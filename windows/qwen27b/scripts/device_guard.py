"""Validate explicitly selected devices against the runtime inventory."""
import re


def parse_devices(output):
    result = {}
    for line in output.splitlines():
        match = re.match(r'\s*(CUDA\d+|Vulkan\d+):\s+(.+?)\s+\(\d+ MiB,', line)
        if match:
            result[match[1]] = match[2]
    return result


def resolve_devices(requested, available, identities=None, explicit=False):
    identities = identities or {}
    selected, resolved = [], {}
    for old in requested.split(','):
        old = old.strip()
        expected = available.get(old) if explicit else identities.get(old, available.get(old))
        if not expected:
            raise ValueError('Selected device is missing from the runtime inventory')
        family = re.sub(r'\d+$', '', old)
        matches = [device for device, name in available.items()
                   if name == expected and device.startswith(family)]
        if len(matches) != 1:
            raise ValueError('Selected device identity is missing or ambiguous')
        if expected in resolved.values():
            raise ValueError('Duplicate physical device selection')
        selected.append(matches[0])
        resolved[matches[0]] = expected
    return ','.join(selected), resolved
