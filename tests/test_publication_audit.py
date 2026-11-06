"""Synthetic privacy regressions; no live values or machine data in fixtures."""
from pathlib import Path
import hashlib
import importlib.util

path = Path(__file__).resolve().parents[1] / 'tools/audit_release.py'
spec = importlib.util.spec_from_file_location('publication_audit', path)
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def rules(data, path='example.py', allow=(), private=()):
    return {item[2] for item in audit.inspect(path, data, set(allow), private)}


def test_private_runtime_file_rejected_even_when_text():
    assert 'private-runtime-path' in rules(b'innocent text', 'linux/state/example.txt')
    assert 'private-runtime-path' in rules(b'{}', 'linux/config.local.json')


def test_binary_payload_rejected_without_suspicious_extension():
    assert 'binary-or-runtime-payload' in rules(b'header\0binary')


def test_provider_key_and_home_address_detection():
    token = b'github_' + b'pat_' + b'SyntheticFixtureNotARealCredential'
    assert 'provider-token' in rules(token)
    home = b'/' + b'home/' + b'example-account/private'
    assert 'personal-home' in rules(home)
    network = b'.'.join([b'192', b'168', b'12', b'34'])
    assert 'private-address' in rules(network)


def test_loopback_and_documentation_address_are_not_private_inventory():
    assert 'private-address' not in rules(b'http://127.0.0.1:8080')
    assert 'private-address' not in rules(b'http://192.0.2.1:8080')


def test_reviewed_exception_is_exact_and_cannot_hide_known_secret():
    line = b'Example <' + b'upstream' + b'@' + b'example.org>'
    allow = {('notice.h', 'email', hashlib.sha256(line).hexdigest())}
    assert not rules(line, 'notice.h', allow)
    assert 'email' in rules(line+b' changed', 'notice.h', allow)
    assert 'known-private-value' in rules(line, 'notice.h', allow, ['upstream'])


def test_empty_blob_request_is_nonblocking():
    assert audit.blobs([]) == {}


def test_private_values_in_filenames_are_rejected():
    assert 'known-private-value' in rules(b'innocent', 'docs/synthetic-operator.txt', private=['synthetic-operator'])
    path = '/' + 'home/' + 'example-account/file.py'
    assert 'personal-home' in rules(b'innocent', path)
