# tests/logging/test_loguru_baseline.py
import builtins
import json
from types import SimpleNamespace


import src.processors.llm.config as llm_config
import src.processors.llm.processor as llm_processor
import src.processors.llm.schema_validator as schema_validator
import src.tooling.vllm_utils as vllm_utils
import src.worker.pipeline as pipeline


# Dummy logger class
class DummyLogger:
    def __init__(self):
        self.records = []

    def _record(self, level, message, **kwargs):
        self.records.append({"level": level, "message": message, "kwargs": kwargs})

    def debug(self, message, *args, **kwargs):
        return self._record("DEBUG", message, **kwargs)

    def info(self, message, *args, **kwargs):
        return self._record("INFO", message, **kwargs)

    def warning(self, message, *args, **kwargs):
        return self._record("WARNING", message, **kwargs)

    def error(self, message, *args, **kwargs):
        return self._record("ERROR", message, **kwargs)

    # Provide bind/patch to mirror loguru API used in codebase
    def bind(self, **kwargs):
        return self

    def patch(self, func):
        return self


class FakeRedis:
    def __init__(self):
        self.storage = {}
        self.hset_calls = []

    def hset(self, key, mapping=None, **kwargs):
        # Accept either mapping or keyword args
        if mapping is None:
            mapping = kwargs
        self.hset_calls.append((key, mapping))
        self.storage[key] = mapping


def test_modules_expose_logger_objects():
    # Ensure modules import a `logger` symbol with expected methods
    for mod in (llm_config, schema_validator, llm_processor, vllm_utils, pipeline):
        assert hasattr(mod, "logger"), f"{mod.__name__} missing logger"
        log = getattr(mod, "logger")
        for method in ("debug", "info", "warning", "error"):
            assert callable(getattr(log, method, None)), (
                f"{mod.__name__}.logger missing {method}"
            )


def test_load_model_generation_config_logs_and_parses(tmp_path, monkeypatch):
    dummy = DummyLogger()
    monkeypatch.setattr(llm_config, "logger", dummy)

    # No model_path logs debug and returns empty config
    cfg = llm_config.load_model_generation_config(None)
    assert isinstance(cfg, llm_config.GenerationConfig)
    assert any("No model path provided" in r["message"] for r in dummy.records)

    # Create a fake generation_config.json
    data = {"temperature": 0.5, "max_new_tokens": 42}
    model_dir = tmp_path / "modelA"
    model_dir.mkdir()
    (model_dir / "generation_config.json").write_text(json.dumps(data))

    dummy.records.clear()
    cfg2 = llm_config.load_model_generation_config(model_dir)
    assert cfg2.temperature == 0.5
    assert cfg2.max_tokens == 42
    assert any("Loaded model config" in r["message"] for r in dummy.records)


def test_validate_llm_output_schema_failure_logs(monkeypatch):
    dummy = DummyLogger()
    monkeypatch.setattr(schema_validator, "logger", dummy)

    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }
    # Provide valid JSON that does not satisfy schema (missing 'title')
    parsed, error = schema_validator.validate_llm_output('{"other": "x"}', schema)
    assert parsed is None
    assert error is not None and "Schema validation failed" in error
    assert any(
        r["level"] == "WARNING" and "Schema validation failed" in r["message"]
        for r in dummy.records
    )


def test_apply_overrides_to_sampling_logs_on_missing_vllm(monkeypatch):
    dummy = DummyLogger()
    monkeypatch.setattr(vllm_utils, "logger", dummy)

    base = SimpleNamespace()
    overrides = {"temperature": 0.1}
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "vllm":
            raise ModuleNotFoundError("No module named 'vllm'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    res = vllm_utils.apply_overrides_to_sampling(base, overrides)
    # vLLM is likely not installed in test env; function should return base and log an error
    assert res is base
    assert any(r["level"] == "ERROR" for r in dummy.records)


def test_update_status_uses_redis_and_logs(monkeypatch):
    dummy = DummyLogger()
    monkeypatch.setattr(pipeline, "logger", dummy)
    fake = FakeRedis()
    from src.api.schemas import TaskStatus

    pipeline._update_status(
        fake, "task:123", TaskStatus.PREPROCESSING, details={"meta": {"x": 1}}
    )
    # Redis hset called
    assert fake.hset_calls, "Redis.hset was not called"
    key, mapping = fake.hset_calls[-1]
    assert key == "task:123"
    assert mapping["status"] == TaskStatus.PREPROCESSING.value
    assert "updated_at" in mapping
    # details dict should be serialized to JSON string for nested dicts in original function
    assert isinstance(mapping["meta"], str) or isinstance(mapping["meta"], dict)
    # Logger should have recorded an info
    assert any(
        r["level"] == "INFO" and "Task status updated" in r["message"]
        for r in dummy.records
    )


def test_pipeline_logging_integration_smoke(monkeypatch):
    # Smoke test: simulate a simple pipeline run where audio_path invalid triggers logging and raises
    dummy = DummyLogger()
    monkeypatch.setattr(pipeline, "logger", dummy)
    task_params = {"audio_path": None}
    result = pipeline.process_audio_task(task_params)
    assert result["status"] == "error"
    assert any(
        r["level"] == "ERROR" and "Invalid audio path" in r["message"]
        for r in dummy.records
    )
