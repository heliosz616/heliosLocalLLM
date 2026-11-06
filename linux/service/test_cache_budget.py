import json
from pathlib import Path
import tempfile
import unittest

from cache_budget import GIB, cap_native_cache, select_budget


class CacheBudgetTests(unittest.TestCase):
    identity = 'a' * 64

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.namespace = Path(self.temporary.name)

    def snapshot(self, name, size, identity=None):
        # Sparse files exercise accounting without allocating the payload bytes.
        with (self.namespace / (name + '.bin')).open('wb') as stream:
            stream.truncate(size)
        value = dict(version=1, identity=identity or self.identity, bytes=size, checksum='0' * 32)
        (self.namespace / (name + '.json')).write_text(json.dumps(value))

    def budget(self, maximum=64, free=100, reserve=32):
        return select_budget(self.namespace, self.identity, maximum * GIB, reserve * GIB, free_bytes=free * GIB)

    def test_empty_cache_keeps_configured_maximum_when_space_allows(self):
        self.assertEqual(self.budget()['effective_maximum_bytes'], 64 * GIB)

    def test_growth_uses_only_free_space_above_reserve(self):
        self.snapshot('123-1', 10 * GIB)
        report = self.budget(free=40)
        self.assertEqual(report['effective_maximum_bytes'], 18 * GIB)
        self.assertEqual(report['maximum_new_payload_growth_bytes'], 8 * GIB)

    def test_space_reserve_does_not_force_existing_entry_pruning(self):
        self.snapshot('123-1', 12 * GIB)
        report = self.budget(free=20)
        self.assertEqual(report['effective_maximum_bytes'], 12 * GIB)
        self.assertEqual(report['maximum_new_payload_growth_bytes'], 0)

    def test_operator_maximum_still_applies(self):
        self.snapshot('123-1', 10 * GIB)
        self.assertEqual(self.budget(maximum=5)['effective_maximum_bytes'], 5 * GIB)

    def test_orphans_and_invalid_metadata_do_not_inflate_evictable_bytes(self):
        self.snapshot('123-1', 10 * GIB, identity='b' * 64)
        self.snapshot('123-2', 10 * GIB)
        (self.namespace / '123-2.json').unlink()
        self.snapshot('invalid-name', 10 * GIB)
        (self.namespace / '123-3.json').write_text('{bad json')
        self.assertEqual(self.budget(free=40)['effective_maximum_bytes'], 8 * GIB)

    def test_sequential_models_share_remaining_free_space(self):
        self.snapshot('123-1', 10 * GIB)
        first = self.budget(free=100)
        free_after_first = 100 * GIB - first['maximum_new_payload_growth_bytes']
        second = select_budget(self.namespace / 'empty', self.identity, 64 * GIB, free_bytes=free_after_first)
        self.assertEqual(second['effective_maximum_bytes'], 14 * GIB)
        self.assertEqual(free_after_first - second['maximum_new_payload_growth_bytes'], 32 * GIB)

    def test_uncached_profile_is_unchanged(self):
        environment = {'OTHER': 'kept'}
        self.assertIsNone(cap_native_cache(environment))
        self.assertEqual(environment, {'OTHER': 'kept'})

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            self.budget(reserve=-1)
        with self.assertRaises(ValueError):
            select_budget(self.namespace, '../bad', GIB, free_bytes=100 * GIB)


if __name__ == '__main__':
    unittest.main()


def test_total_capacity_subtracts_other_model_cache(tmp_path):
    from cache_retention import cache_bytes
    active = tmp_path / 'active' / ('a' * 64)
    other = tmp_path / 'other' / ('b' * 64)
    active.mkdir(parents=True)
    other.mkdir(parents=True)
    (other / '100-1.bin').write_bytes(b'x' * 1024)
    (other / '100-1.json').write_text(json.dumps(dict(version=1, identity=other.name)))
    env = dict(LLAMA_CACHE_SSD_DIR=str(active.parent), LLAMA_CACHE_SSD_ID=active.name,
               LLAMA_CACHE_SSD_BYTES=str(256 * GIB))
    result = cap_native_cache(env, 0, roots=[active.parent, other.parent], total_maximum_bytes=64_000_000_000)
    assert int(env['LLAMA_CACHE_SSD_BYTES']) + cache_bytes([other.parent]) + 1024**2 == 64_000_000_000
    assert result['total_maximum_bytes'] == 64_000_000_000


def test_recurrent_and_kv_native_writes_share_one_global_reservation_limit(tmp_path):
    active=tmp_path/'flash'/('a'*64)
    active.mkdir(parents=True)
    env=dict(LLAMA_CACHE_SSD_DIR=str(active.parent),LLAMA_CACHE_SSD_ID=active.name,
             LLAMA_CACHE_SSD_BYTES=str(256*GIB),LLAMA_RECURRENT_CHECKPOINTS='1')
    cap_native_cache(env,0,roots=[active.parent,tmp_path/'q5'],total_maximum_bytes=64_000_000_000)
    assert env['LLAMA_CACHE_SSD_SHARED_BYTES']=='64000000000'
    assert env['LLAMA_CACHE_SSD_SHARED_LOCK']==str(tmp_path/'.kv-quota.lock')
    assert json.loads(env['LLAMA_CACHE_SSD_SHARED_ROOTS'])==[str(active.parent),str(tmp_path/'q5')]
    assert int(env['LLAMA_CACHE_SSD_BYTES'])<=64_000_000_000
