import asyncio
import pytest

from src.processors import base


def _raise_exc():
    raise RuntimeError("boom")


def test_safe_execute_sync_catches_exception():
    result, error = base.safe_execute_sync(_raise_exc)
    assert result is None
    assert error is not None
    assert "boom" in error.get("message", "")
    assert "type" in error


class DummyProcessor(base.Processor):
    def execute(self, *args, **kwargs):
        return "sync-ok"

    def unload(self) -> None:
        pass


def test_async_execute_calls_sync_execute():
    p = DummyProcessor()
    res = asyncio.run(p.async_execute())
    assert res == "sync-ok"


def test_asrresult_error_field():
    r = base.ASRResult(text="hello", error={"message": "err"})
    assert r.error["message"] == "err"


# -- New tests for enhanced base.py features (TDD) --


class UnloadTrackerProcessor(base.Processor):
    """Tracks whether unload() was called when used as a context manager."""
    def __init__(self):
        self.unloaded = False

    def execute(self, *args, **kwargs):
        return "ok"

    def unload(self) -> None:
        self.unloaded = True


def test_context_manager_calls_unload():
    p = UnloadTrackerProcessor()
    with p as ctx:
        # __enter__ should return self
        assert ctx is p
        # unload should not have been called yet
        assert not p.unloaded
    # after exiting, unload must have been called deterministically
    assert p.unloaded is True


# Runtime-checkable protocol tests
class FakeASRBackend:
    def execute(self, audio_data: bytes, **kwargs) -> base.ASRResult:
        return base.ASRResult(text="transcript")

    def unload(self) -> None:
        pass

    def get_version_info(self) -> base.VersionInfo:
        return {"name": "fake-asr", "version": "0.1"}


class FakeLLMBackend:
    def execute(self, text: str, **kwargs) -> base.LLMResult:
        return base.LLMResult(text="response")

    def unload(self) -> None:
        pass

    def get_version_info(self) -> base.VersionInfo:
        return {"name": "fake-llm", "version": "0.2"}


def test_runtime_checkable_protocols():
    asr = FakeASRBackend()
    llm = FakeLLMBackend()

    # Structural runtime isinstance checks should succeed for matching protocols
    assert isinstance(asr, base.ASRBackend)
    assert isinstance(llm, base.LLMBackend)

    # Because Protocol runtime checks are structural and ignore argument/return
    # annotations, a well-formed ASR backend may also satisfy LLMBackend at runtime.
    # Verify negative cases using an intentionally incomplete object.
    class IncompleteBackend:
        def execute(self, *a, **k):
            return None

    incomplete = IncompleteBackend()
    assert not isinstance(incomplete, base.ASRBackend)
    assert not isinstance(incomplete, base.LLMBackend)


def _raise_processor_error():
    raise base.ProcessorError("processor failed")


def test_processor_error_is_captured_by_safe_execute_sync():
    result, error = base.safe_execute_sync(_raise_processor_error)
    assert result is None
    assert error is not None
    assert error.get("type") == "ProcessorError"
    assert "processor failed" in error.get("message", "")


def test_reduce_memory_default_and_override_behavior():
    # default Processor (via DummyProcessor) has no-op reduce_memory
    p = DummyProcessor()
    assert p.reduce_memory() is None

    # override reduce_memory to mutate object state
    class MemProcessor(base.Processor):
        def __init__(self):
            self.cleaned = False

        def execute(self, *a, **k):
            return "done"

        def unload(self) -> None:
            pass

        def reduce_memory(self) -> None:
            # simulate freeing caches
            self.cleaned = True
            return None

    m = MemProcessor()
    assert getattr(m, "cleaned", False) is False
    assert m.reduce_memory() is None
    assert m.cleaned is True


def test_versioninfo_structure_adherence():
    # VersionInfo is TypedDict(total=False) — optional keys allowed
    v1: base.VersionInfo = {"name": "backend-x"}
    assert isinstance(v1, dict)
    assert v1["name"] == "backend-x"
    assert "version" not in v1

    # Fake backends should return a dict compatible with VersionInfo
    asr = FakeASRBackend()
    info = asr.get_version_info()
    assert isinstance(info, dict)
    assert "name" in info
    assert info["name"] == "fake-asr"