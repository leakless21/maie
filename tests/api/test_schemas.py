"""Concise tests for src.api.schemas using RED-style one-line expectations."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.api import schemas


@pytest.fixture
def sample_asr_backend():
    """RED: minimal ASR backend dict containing required keys for ASRBackendSchema."""
    return {
        "name": "whisper",
        "model_variant": "small",
        "model_path": "/models/whisper-small",
        "checkpoint_hash": "abc123",
        "compute_type": "cpu",
        "decoding_params": {"beam_size": 5},
    }


@pytest.fixture
def sample_llm():
    """RED: minimal LLM backend dict containing required keys for LLMSchema."""
    return {
        "name": "qwen",
        "checkpoint_hash": "def456",
        "quantization": "int8",
        "thinking": True,
        "reasoning_parser": None,
        "structured_output": {"type": "json"},
        "decoding_params": {"max_tokens": 200},
    }


def test_enums_have_expected_members():
    """RED: TaskStatus includes 'PENDING' and 'COMPLETE'; Feature includes 'raw_transcript' and 'summary'."""
    members = {m.value for m in schemas.TaskStatus}
    assert "PENDING" in members
    assert "COMPLETE" in members
    features = {f.value for f in schemas.Feature}
    assert "raw_transcript" in features
    assert "summary" in features


def test_process_request_defaults_and_template_validation():
    """RED: minimal ProcessRequestSchema with template_id constructs; SUMMARY without template_id raises ValidationError."""
    req = schemas.ProcessRequestSchema(file="audio.wav", template_id="meeting_notes_v1")
    assert isinstance(req, schemas.ProcessRequestSchema)
    assert schemas.Feature.CLEAN_TRANSCRIPT in req.features

    with pytest.raises(ValidationError):
        schemas.ProcessRequestSchema(
            file="audio.wav", features=[schemas.Feature.SUMMARY]
        )

    req2 = schemas.ProcessRequestSchema(
        file="audio.wav",
        features=[schemas.Feature.SUMMARY],
        template_id="nonexistent_template",
    )
    assert req2.template_id == "nonexistent_template"


def test_process_request_accepts_comma_delimited_features():
    """RED: ProcessRequestSchema splits comma-delimited feature strings into enums."""
    req = schemas.ProcessRequestSchema(
        file="audio.wav",
        features="summary,raw_transcript",
        template_id="meeting_notes_v1",
    )

    assert req.features == [schemas.Feature.SUMMARY, schemas.Feature.RAW_TRANSCRIPT]


def test_text_process_request_accepts_comma_delimited_features():
    """RED: TextProcessRequestSchema splits comma-delimited feature strings into enums."""
    req = schemas.TextProcessRequestSchema(
        text="hello",
        features="summary,clean_transcript",
        template_id="meeting_notes_v1",
    )

    assert req.features == [
        schemas.Feature.SUMMARY,
        schemas.Feature.CLEAN_TRANSCRIPT,
    ]


def test_process_response_and_status_response_roundtrip():
    """RED: ProcessResponse retains task_id; StatusResponseSchema.status serializes to 'PENDING'."""
    tid = uuid4()
    resp = schemas.ProcessResponse(task_id=tid, status="PENDING")
    j = resp.model_dump()
    assert j["task_id"] == tid

    status = schemas.StatusResponseSchema(
        task_id=tid, status=schemas.TaskStatus.PENDING.value
    )
    d = status.model_dump()
    assert d["status"] == "PENDING"


def test_versions_and_nested_models(sample_asr_backend, sample_llm):
    """RED: VersionsSchema preserves pipeline_version and nested asr_backend/llm fields."""
    asr = schemas.ASRBackendSchema(**sample_asr_backend)
    llm = schemas.LLMSchema(**sample_llm)
    versions = schemas.VersionsSchema(
        pipeline_version="1.0.0", asr_backend=asr, llm=llm
    )
    out = versions.model_dump()
    assert out["pipeline_version"] == "1.0.0"
    assert out["asr_backend"]["name"] == "whisper"
    assert out["llm"]["thinking"] is True


def test_models_and_template_responses_defaults_and_fields():
    """RED: ModelInfoSchema.supported_languages defaults to []; TemplatesResponseSchema preserves template id when available."""
    model = schemas.ModelInfoSchema(
        id="m1",
        name="ASR Model",
        description="desc",
        type="ASR",
        version="v1",
    )
    assert model.supported_languages == []
    models_resp = schemas.ModelsResponseSchema(models=[model])
    assert len(models_resp.models) == 1

    if hasattr(schemas, "TemplateInfoSchema"):
        tmpl = schemas.TemplateInfoSchema(
            id="t1",
            name="tmpl",
            description="d",
            schema_url="http://example.com/schema",
        )
        tpl_resp = schemas.TemplatesResponseSchema(templates=[tmpl])
        assert tpl_resp.templates[0].id == "t1"


def test_metrics_and_results_validation():
    """RED: MetricsSchema accepts numeric values; non-numeric input_duration_seconds raises ValidationError."""
    metrics = schemas.MetricsSchema(
        input_duration_seconds=10.0,
        processing_time_seconds=1.5,
        rtf=0.15,
        vad_coverage=0.9,
        asr_confidence_avg=0.85,
    )
    assert metrics.edit_rate_cleaning is None

    with pytest.raises(ValidationError):
        schemas.MetricsSchema(
            input_duration_seconds="ten",
            processing_time_seconds=1.5,
            rtf=0.15,
            vad_coverage=0.9,
            asr_confidence_avg=0.85,
        )


def test_json_schema_generation_includes_features():
    """RED: ProcessRequestSchema.model_json_schema() includes 'features' in properties."""
    js = schemas.ProcessRequestSchema.model_json_schema()
    props = js.get("properties", {})
    assert "features" in props


def test_results_schema_smoke():
    """RED: ResultsSchema stores provided fields; clean_transcript equals provided 'clean'."""
    r = schemas.ResultsSchema(
        raw_transcript="raw", clean_transcript="clean", summary={"k": "v"}
    )
    assert r.clean_transcript == "clean"


def test_enable_diarization_defaults_to_false():
    """RED: ProcessRequestSchema.enable_diarization defaults to False."""
    req = schemas.ProcessRequestSchema(
        file="audio.wav",
        features=[schemas.Feature.RAW_TRANSCRIPT],
    )
    assert req.enable_diarization is False


def test_enable_diarization_accepts_true():
    """RED: ProcessRequestSchema.enable_diarization can be set to True."""
    req = schemas.ProcessRequestSchema(
        file="audio.wav",
        features=[schemas.Feature.RAW_TRANSCRIPT],
        enable_diarization=True,
    )
    assert req.enable_diarization is True


def test_enable_diarization_accepts_false_explicitly():
    """RED: ProcessRequestSchema.enable_diarization can be explicitly set to False."""
    req = schemas.ProcessRequestSchema(
        file="audio.wav",
        features=[schemas.Feature.RAW_TRANSCRIPT],
        enable_diarization=False,
    )
    assert req.enable_diarization is False


def test_enable_diarization_validates_boolean_type():
    """RED: ProcessRequestSchema.enable_diarization coerces string 'yes' to True."""
    req = schemas.ProcessRequestSchema(
        file="audio.wav",
        features=[schemas.Feature.RAW_TRANSCRIPT],
        enable_diarization="yes",  # Pydantic coerces to True
    )
    # Pydantic coerces truthy strings to True
    assert req.enable_diarization is True

    # Test that invalid types like dict raise ValidationError
    with pytest.raises(ValidationError):
        schemas.ProcessRequestSchema(
            file="audio.wav",
            features=[schemas.Feature.RAW_TRANSCRIPT],
            enable_diarization={"invalid": "type"},
        )
