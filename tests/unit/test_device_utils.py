from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def device_module(monkeypatch):
    import src.utils.device as device

    # Ensure caches are cleared before each test
    device.reset_device_cache()

    yield device

    # Clear caches after test to avoid cross-test pollution
    device.reset_device_cache()


def test_has_cuda_without_torch_returns_false(device_module, monkeypatch):
    monkeypatch.setattr(device_module, "torch", None, raising=False)

    assert device_module.has_cuda() is False


def test_has_cuda_is_cached(device_module, monkeypatch):
    mock_is_available = Mock(return_value=True)
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=mock_is_available))
    monkeypatch.setattr(device_module, "torch", fake_torch, raising=False)

    assert device_module.has_cuda() is True
    assert device_module.has_cuda() is True
    assert mock_is_available.call_count == 1


def test_select_device_respects_env_override(device_module, monkeypatch):
    monkeypatch.setenv("MAIE_DEVICE", "cpu")
    monkeypatch.setattr(device_module, "has_cuda", Mock(return_value=True))

    device_module.select_device.cache_clear()

    assert device_module.select_device() == "cpu"


@pytest.mark.parametrize(
    "cuda_available,mps_available,expected",
    [(True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_select_device_priority(
    device_module, monkeypatch, cuda_available, mps_available, expected
):
    monkeypatch.delenv("MAIE_DEVICE", raising=False)
    monkeypatch.setattr(device_module, "has_cuda", Mock(return_value=cuda_available))
    monkeypatch.setattr(
        device_module, "_mps_available", Mock(return_value=mps_available)
    )

    device_module.select_device.cache_clear()

    assert device_module.select_device() == expected


def test_ensure_cuda_available_raises_when_missing(device_module, monkeypatch):
    monkeypatch.setattr(device_module, "has_cuda", Mock(return_value=False))

    with pytest.raises(RuntimeError) as exc:
        device_module.ensure_cuda_available("GPU required")

    assert "GPU required" in str(exc.value)
