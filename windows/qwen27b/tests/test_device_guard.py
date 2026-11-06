import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pytest
from device_guard import parse_devices, resolve_devices


def test_inventory_and_selection():
    available = parse_devices('CUDA0: Example GPU A (4096 MiB, 2048 MiB free)')
    assert resolve_devices('CUDA0', available)[0] == 'CUDA0'


def test_identity_remapping():
    assert resolve_devices('Vulkan1', {'Vulkan2': 'Example GPU B'},
                           {'Vulkan1': 'Example GPU B'})[0] == 'Vulkan2'


def test_missing_ambiguous_and_duplicate():
    with pytest.raises(ValueError):
        resolve_devices('CUDA0', {})
    with pytest.raises(ValueError):
        resolve_devices('Vulkan0', {'Vulkan0': 'Example GPU', 'Vulkan1': 'Example GPU'})
    with pytest.raises(ValueError):
        resolve_devices('CUDA0,Vulkan0', {'CUDA0': 'Example GPU', 'Vulkan0': 'Example GPU'})
