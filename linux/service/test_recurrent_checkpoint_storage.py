"""Native storage tests: no model loads, GPU calls or production cache writes."""
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / 'native/checkpoints'

@pytest.fixture(scope='module')
def native_storage_tests(tmp_path_factory):
    build = tmp_path_factory.mktemp('native-checkpoint-storage')
    for name in ('checkpoint-policy', 'checkpoint-bundle', 'shared-cache-quota'):
        subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                        '-I', str(ROOT / 'native/flash-core/vendor'),
                        str(OVERLAY / ('test-' + name + '.cpp')), '-o', str(build / name)],
                       check=True, capture_output=True, text=True)
    return build


def test_six_checkpoint_retention_and_replay_boundaries(native_storage_tests):
    subprocess.run([str(native_storage_tests / 'checkpoint-policy')], check=True)


def test_partial_state_bundle_roundtrip_and_extent_validation(native_storage_tests, tmp_path):
    subprocess.run([str(native_storage_tests / 'checkpoint-bundle'), str(tmp_path / 'checkpoint.bin')], check=True)


def test_combined_quota_counts_pending_writes_and_recovers_crashed_writer(native_storage_tests, tmp_path):
    subprocess.run([str(native_storage_tests / 'shared-cache-quota'), str(tmp_path)], check=True)
