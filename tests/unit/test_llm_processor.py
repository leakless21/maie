"""
Unit tests for LLM processor functionality.

Tests cover model loading, text enhancement, structured summarization,
and resource management with comprehensive mocking.
"""

import json
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.config.model import LlmBackendType, LlmServerSettings
from src.processors.base import LLMResult
from src.processors.llm.processor import LLMProcessor
from src.tooling.llm_client import VllmServerClient


class TestLLMProcessorInitialization:
    """Test LLM processor initialization."""

    def test_init_with_default_model(self):
        """Test initialization with default model path."""
        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.llm_enhance.model = "test-model"
            mock_settings.paths.templates_dir = Path("templates")
            mock_settings.llm_enhance.temperature = 0.7

            processor = LLMProcessor()

            assert processor.model_path == "test-model"
            assert processor.model is None
            assert processor._model_loaded is False
            assert processor.checkpoint_hash is None

    def test_init_with_custom_model(self):
        """Test initialization with custom model path."""
        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.paths.templates_dir = Path("templates")

            processor = LLMProcessor(model_path="custom-model")

            assert processor.model_path == "custom-model"

    def test_init_creates_prompt_renderer(self):
        """Test that prompt renderer is created during initialization."""
        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.paths.templates_dir = Path("templates")

            with patch("src.processors.llm.processor.TemplateLoader") as mock_loader:
                with patch(
                    "src.processors.llm.processor.PromptRenderer"
                ) as mock_renderer:
                    LLMProcessor()

                    mock_loader.assert_called_once()
                    mock_renderer.assert_called_once()


class TestLoadModel:
    """Test model loading functionality."""

    def test_load_model_success(self):
        """Test successful model loading."""
        processor = LLMProcessor()

        mock_model = Mock()
        mock_model.get_default_sampling_params.return_value = Mock()

        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.llm_backend = LlmBackendType.LOCAL_VLLM
            mock_settings.llm_enhance = SimpleNamespace(
                enhance_base_url="http://test-server/v1",
                enhance_api_key=None,
                enhance_model_name="test-model",
                summary_url="http://test-server/v1",
                summary_api_key=None,
                summary_model_name="test-model",
                request_timeout_seconds=60.0,
            )

            with patch("vllm.LLM") as mock_llm_class:
                mock_llm_class.return_value = mock_model

                with patch(
                    "src.processors.llm.processor.calculate_checkpoint_hash"
                ) as mock_hash:
                    mock_hash.return_value = "test-hash"

                    with patch("src.processors.llm.processor.get_model_info") as mock_info:
                        mock_info.return_value = {"model_name": "test-model"}

                        processor._load_model()

                        assert processor.model == mock_model
                        assert processor.checkpoint_hash == "test-hash"
                        assert processor._model_loaded is True
                        mock_llm_class.assert_called_once()

    def test_load_model_advanced_scheduler_args(self):
        """Advanced scheduler overrides should be forwarded to vLLM."""

        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.llm_enhance_model = "test-model"
            mock_settings.paths.templates_dir = Path("templates")
            mock_settings.verbose_components = False
            mock_settings.llm_enhance = SimpleNamespace(
                model="test-model",
                temperature=0.0,
                top_p=None,
                top_k=None,
                max_tokens=None,
                quantization=None,
                gpu_memory_utilization=0.8,
                max_model_len=8192,
                max_num_seqs=4,
                max_num_batched_tokens=2048,
                max_num_partial_prefills=2,
            )
            mock_settings.llm_sum = SimpleNamespace(
                temperature=0.7,
                top_p=0.9,
                top_k=20,
                max_tokens=1000,
            )

            with (
                patch("src.processors.llm.processor.TemplateLoader"),
                patch("src.processors.llm.processor.PromptRenderer"),
            ):
                processor = LLMProcessor()

            mock_model = Mock()
            mock_model.get_default_sampling_params.return_value = Mock()

            with patch("vllm.LLM") as mock_llm_class:
                mock_llm_class.return_value = mock_model
                processor._load_model()

            kwargs = mock_llm_class.call_args.kwargs
        assert kwargs["max_num_seqs"] == 4
        assert kwargs["max_num_batched_tokens"] == 2048
        assert kwargs["max_num_partial_prefills"] == 2

    def test_load_model_vllm_not_installed(self):
        """Test graceful handling when vLLM is not installed."""
        processor = LLMProcessor()

        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.llm_backend = LlmBackendType.VLLM_SERVER
            # Simulate server client initialization error
            with patch("src.processors.llm.processor.VllmServerClient", side_effect=ImportError("vLLM server client not available")):
                with pytest.raises(ImportError):
                    processor._load_model()

            assert processor.client_enhance is None
            assert processor.client_summary is None
            assert "init_error" in processor.checkpoint_hash
            assert processor._model_loaded is False

    def test_load_model_load_error(self):
        """Test handling of model loading errors."""
        processor = LLMProcessor()

        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.llm_backend = LlmBackendType.VLLM_SERVER
            with patch("src.processors.llm.processor.VllmServerClient", side_effect=Exception("Load error")):
                with pytest.raises(Exception, match="Load error"):
                    processor._load_model()

    def test_load_model_server_backend(self):
        """Test loading model with VLLM_SERVER backend."""
        processor = LLMProcessor()

        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.llm_backend = LlmBackendType.VLLM_SERVER
            mock_settings.llm_server.enhance_base_url = "http://test-server/v1"
            mock_settings.llm_server.enhance_api_key = None
            mock_settings.llm_server.enhance_model_name = "test-model"
            mock_settings.llm_server.summary_url = "http://test-server/v1"  # Property
            mock_settings.llm_server.summary_api_key = None
            mock_settings.llm_server.summary_model_name = "test-model"
            mock_settings.llm_server.request_timeout_seconds = 60.0

            with patch(
                "src.processors.llm.processor.VllmServerClient"
            ) as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                processor._load_model()

                # Both clients should be initialized
                assert processor.client_enhance == mock_client
                assert processor.client_summary == mock_client
                assert processor._model_loaded is True
                assert processor.model_info["backend"] == "vllm_server"

                # Should be called twice (enhance + summary)
                assert mock_client_class.call_count == 2

        assert processor.model is None
        assert "load_error" not in processor.checkpoint_hash

    def test_load_model_already_loaded(self):
        """Test that model is not loaded twice."""
        processor = LLMProcessor()
        processor._model_loaded = True

        with patch("vllm.LLM") as mock_llm_class:
            processor._load_model()

            # Should not call LLM constructor
            mock_llm_class.assert_not_called()


class TestExecute:
    """Test basic execution functionality."""

    def test_execute_without_model(self):
        """Test execution when model is not available."""
        processor = LLMProcessor()
        processor.model = None
        processor._model_loaded = True
        with pytest.raises(RuntimeError, match="No LLM client configured"):
            processor.execute("test text")

    def test_execute_with_model(self):
        """Test execution when model is loaded."""
        processor = LLMProcessor()
        processor._model_loaded = True
        processor.client_enhance = Mock()
        processor.client_summary = processor.client_enhance

        # Mock client response
        mock_output = Mock()
        mock_output.outputs = [Mock(text="generated text")]
        mock_output.prompt_token_ids = [1, 2, 3]
        mock_output.outputs[0].token_ids = [4, 5]
        processor.client_enhance.chat.return_value = [mock_output]

        result = processor.execute("test prompt")

        assert result.text == "generated text"
        assert result.model_info["usage"]["prompt_tokens"] == 3
        assert result.model_info["usage"]["completion_tokens"] == 2
        processor.client_enhance.chat.assert_called_once()

    def test_execute_generation_error(self):
        """Test execution when generation fails."""
        processor = LLMProcessor()
        processor._model_loaded = True
        # Simulate server client raising generation error
        processor.client_enhance = Mock()
        processor.client_enhance.chat.side_effect = Exception("Generation error")

        with patch(
            "src.processors.llm.processor.apply_overrides_to_sampling"
        ) as mock_apply:
            mock_apply.return_value = Mock()

            result = processor.execute("test text")

            # Should return original text on error
            assert result.text == "test text"

    def test_execute_loads_model_lazily(self):
        """Test that model is loaded on first execution."""
        processor = LLMProcessor()

        with patch.object(processor, "_load_model") as mock_load:
            # Patch _load_model to set _model_loaded and provide a client
            def _fake_load():
                processor._model_loaded = True
                processor.client_enhance = Mock()
                processor.client_enhance.chat.return_value = [Mock(outputs=[Mock(text="ok")])]

            mock_load.side_effect = _fake_load

            processor.execute("test text")

            mock_load.assert_called_once()

    def test_execute_render_error(self):
        """Test execution when prompt rendering fails."""
        processor = LLMProcessor()
        processor._model_loaded = True
        processor.client_enhance = Mock()
        processor.client_summary = processor.client_enhance

        # Manually mock prompt renderer
        processor.prompt_renderer = Mock()
        processor.prompt_renderer.render.side_effect = Exception("Render error")

        with patch(
            "src.processors.llm.processor.load_template_schema"
        ) as mock_load_schema:
            mock_load_schema.return_value = {"type": "object"}
            result = processor.execute("transcript", task="summary", template_id="test")

        assert result.text == "transcript"
        assert "error" in result.metadata
        assert "Prompt rendering error" in result.metadata["error"]


class TestEnhanceText:
    """Test text enhancement functionality."""

    def test_enhance_text_without_model(self):
        """Test enhancement when model is not available."""
        processor = LLMProcessor()

        # Mock _load_model to simulate failure to load (does not set _model_loaded=True)
        with patch.object(processor, "_load_model") as mock_load:
            # _load_model does nothing, so _model_loaded remains False
            result = processor.enhance_text("test text")

            assert result["enhancement_applied"] is False
            assert result["enhanced_text"] == "test text"
            assert result["edit_distance"] == 0
            mock_load.assert_called_once()

    def test_enhance_text_with_model(self):
        """Test successful text enhancement."""
        processor = LLMProcessor()
        processor._model_loaded = True
        processor.model = Mock()
        processor.model_info = {"model_name": "test"}

        enhanced_result = LLMResult(
            text="Enhanced text with punctuation.",
            tokens_used=5,
            model_info={"model_name": "test"},
            metadata={"task": "enhancement"},
        )

        with patch.object(
            processor, "execute", return_value=enhanced_result
        ) as mock_execute:
            result = processor.enhance_text("test text without punctuation")

        assert result["enhanced_text"] == "Enhanced text with punctuation."
        assert result["enhancement_applied"] is True
        assert result["edit_distance"] > 0
        assert "edit_rate" in result
        mock_execute.assert_called_once_with(
            "test text without punctuation", task="enhancement"
        )

    def test_enhance_text_prompt_render_error(self):
        """Test enhancement when prompt rendering fails."""
        processor = LLMProcessor()
        processor._model_loaded = True
        processor.model = Mock()
        # Provide a server client to avoid no-client error
        processor.client_enhance = Mock()
        processor.model_info = {"model_name": "test"}

        with patch.object(
            processor.prompt_renderer,
            "render",
            side_effect=Exception("Render error"),
        ):
            result = processor.enhance_text("test text")

            # When template rendering fails in execute(), it returns an error in metadata
            # The result should fall back gracefully
            assert (
                "enhanced_text" in result or result.get("enhancement_applied") is False
            )

    def test_enhance_text_loads_model_lazily(self):
        """Test that model is loaded on first enhancement."""
        processor = LLMProcessor()

        with patch.object(processor, "_load_model") as mock_load:
            processor.enhance_text("test text")

            mock_load.assert_called_once()


class TestNeedsEnhancement:
    """Test enhancement decision logic."""

    def test_needs_enhancement_whisper(self):
        """Test that Whisper backend doesn't need enhancement."""
        processor = LLMProcessor()

        assert processor.needs_enhancement("whisper") is False
        assert processor.needs_enhancement("WHISPER") is False

    def test_needs_enhancement_other_backends(self):
        """Test that other backends need enhancement."""
        processor = LLMProcessor()

        assert processor.needs_enhancement("chunkformer") is True
        assert processor.needs_enhancement("other") is True


class TestGenerateSummary:
    """Test structured summarization functionality."""

    def test_generate_summary_without_model(self):
        """Test summarization when model is not available."""
        processor = LLMProcessor()

        # Mock _load_model to simulate failure to load (does not set _model_loaded=True)
        with patch.object(processor, "_load_model") as mock_load:
            # After load, if no client is configured we expect an error (no fallback)
            processor._model_loaded = True
            processor.client_summary = None
            with pytest.raises(RuntimeError, match="No LLM client configured"):
                processor.generate_summary("transcript", "meeting_notes_v1")
            mock_load.assert_called_once()
            mock_load.assert_called_once()

    def test_generate_summary_schema_load_error(self, tmp_path):
        """Test summarization when schema loading fails."""
        processor = LLMProcessor()
        processor._model_loaded = True
        processor.model = Mock()

        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.paths.templates_dir = tmp_path

            result = processor.generate_summary("transcript", "nonexistent_template")

            assert result["summary"] is None
            assert "Template schema error" in result["error"]

    def test_generate_summary_prompt_render_error(self, tmp_path):
        """Test summarization when prompt rendering fails."""
        processor = LLMProcessor()
        processor._model_loaded = True
        processor.model = Mock()

        # Create valid schema file
        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        schema_file = schemas_dir / "meeting_notes_v1.json"
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "tags": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {"type": "string"},
                },
            },
            "required": ["title", "tags"],
        }
        schema_file.write_text(json.dumps(schema))

        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.paths.templates_dir = tmp_path

            # Mock execute to return an error result directly
            # This verifies generate_summary handles errors from execute correctly
            with patch.object(processor, "execute") as mock_execute:
                mock_execute.return_value = LLMResult(
                    text="transcript",
                    tokens_used=None,
                    model_info={},
                    metadata={"error": "Prompt rendering error: Render error"},
                )

                result = processor.generate_summary("transcript", "meeting_notes_v1")

            assert result["summary"] is None
            assert "error" in result
            assert "Prompt rendering error" in result["error"]

    def test_generate_summary_success(self, tmp_path):
        """Test successful summarization."""
        processor = LLMProcessor()
        processor._model_loaded = True

        # Create valid schema file
        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        schema_file = schemas_dir / "meeting_notes_v1.json"
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "tags": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {"type": "string"},
                },
            },
            "required": ["title", "tags"],
        }
        schema_file.write_text(json.dumps(schema))

        # Mock model and outputs
        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = '{"title": "Test Meeting", "tags": ["meeting"]}'

        mock_model = Mock()
        mock_model.get_default_sampling_params.return_value = Mock()
        mock_model.generate.return_value = [mock_output]
        processor.model = mock_model

        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.paths.templates_dir = tmp_path
            mock_settings.llm_sum_temperature = 0.7
            mock_settings.llm_sum_top_p = 0.9
            mock_settings.llm_sum_top_k = 20
            mock_settings.llm_sum_max_tokens = 1000

            with patch.object(processor, "execute") as mock_execute:
                mock_execute.return_value = LLMResult(
                    text='{"title": "Test Meeting", "tags": ["meeting"]}',
                    tokens_used=None,
                    model_info={"model_name": "test"},
                    metadata={},
                )

                with patch.object(processor.prompt_renderer, "render") as mock_render:
                    mock_render.return_value = "rendered prompt"

                    result = processor.generate_summary(
                        "transcript", "meeting_notes_v1"
                    )

                    assert result["summary"] == {
                        "title": "Test Meeting",
                        "tags": ["meeting"],
                    }
                    assert result["retry_count"] == 0
                    assert "model_info" in result

    def test_generate_summary_validation_retry(self, tmp_path):
        """Test summarization with validation retry logic."""
        processor = LLMProcessor()
        processor._model_loaded = True

        # Create valid schema file
        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        schema_file = schemas_dir / "meeting_notes_v1.json"
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "tags": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {"type": "string"},
                },
            },
            "required": ["title", "tags"],
        }
        schema_file.write_text(json.dumps(schema))

        processor.model = Mock()

        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.paths.templates_dir = tmp_path
            mock_settings.llm_sum_temperature = 0.7
            mock_settings.llm_sum_top_p = 0.9
            mock_settings.llm_sum_top_k = 20
            mock_settings.llm_sum_max_tokens = 1000

            # Mock execute to return invalid JSON first, then valid
            call_count = 0

            def mock_execute_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return LLMResult(
                        text="invalid json",
                        tokens_used=None,
                        model_info={},
                        metadata={},
                    )
                else:
                    return LLMResult(
                        text='{"title": "Test", "tags": ["meeting"]}',
                        tokens_used=None,
                        model_info={},
                        metadata={},
                    )

            with patch.object(
                processor, "execute", side_effect=mock_execute_side_effect
            ):
                with patch.object(processor.prompt_renderer, "render") as mock_render:
                    mock_render.return_value = "rendered prompt"

                    result = processor.generate_summary(
                        "transcript", "meeting_notes_v1"
                    )

                    assert result["summary"] == {"title": "Test", "tags": ["meeting"]}
                    assert result["retry_count"] == 1  # One retry

    def test_generate_summary_structured_outputs_toggle(self, tmp_path):
        """Test that structured outputs are added only when enabled in settings."""
        processor = LLMProcessor()
        processor._model_loaded = True

        # Create minimal schema file
        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        schema_file = schemas_dir / "meeting_notes_v1.json"
        schema = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
        }
        schema_file.write_text(json.dumps(schema))

        # Disable structured outputs: execute() should not receive structured_outputs
        with (
            patch("src.processors.llm.processor.settings") as mock_settings,
            patch.object(processor, "execute") as mock_execute,
        ):
            mock_settings.paths.templates_dir = tmp_path
            mock_settings.llm_sum = SimpleNamespace(
                temperature=0.7,
                top_p=0.9,
                top_k=20,
                max_tokens=1000,
                structured_outputs_enabled=False,
            )

            mock_execute.return_value = LLMResult(
                text="{}",
                tokens_used=None,
                model_info={},
                metadata={},
            )

            processor.generate_summary("transcript", "meeting_notes_v1")

            _, called_kwargs = mock_execute.call_args
            assert "structured_outputs" not in called_kwargs

        # Enable structured outputs: execute() should receive structured_outputs
        with (
            patch("src.processors.llm.processor.settings") as mock_settings,
            patch.object(processor, "execute") as mock_execute,
        ):
            mock_settings.paths.templates_dir = tmp_path
            mock_settings.llm_sum = SimpleNamespace(
                temperature=0.7,
                top_p=0.9,
                top_k=20,
                max_tokens=1000,
                structured_outputs_enabled=True,
            )

            mock_execute.return_value = LLMResult(
                text="{}",
                tokens_used=None,
                model_info={},
                metadata={},
            )

            processor.generate_summary("transcript", "meeting_notes_v1")

            _, called_kwargs = mock_execute.call_args
            assert "structured_outputs" in called_kwargs

    def test_generate_summary_loads_model_lazily(self):
        """Test that model is loaded on first summarization."""
        processor = LLMProcessor()

        with patch.object(processor, "_load_model") as mock_load:
            processor.generate_summary("transcript", "template")

            mock_load.assert_called_once()


class TestUnload:
    """Test model unloading functionality."""

    def test_unload_with_model(self):
        """Test unloading when model is loaded."""
        processor = LLMProcessor()
        processor.model = Mock()
        processor._model_loaded = True
        processor.current_template_id = "test"
        processor.current_schema_hash = "hash"

        with (
            patch("src.processors.llm.processor.torch") as mock_torch,
            patch("src.processors.llm.processor.has_cuda", return_value=True),
            patch("src.processors.llm.processor.settings") as mock_settings,
        ):
            mock_settings.llm_backend = LlmBackendType.VLLM_SERVER
            processor.unload()

            assert processor.model is None
            assert processor._model_loaded is False
            assert processor.current_template_id is None
            assert processor.current_schema_hash is None
            # For server backend we should not clear CUDA cache during unload
            mock_torch.cuda.empty_cache.assert_not_called()

    def test_unload_without_model(self):
        """Test unloading when no model is loaded."""
        processor = LLMProcessor()
        processor.model = None

        processor.unload()

        # Should not raise any errors
        assert processor.model is None

    def test_unload_cuda_error(self):
        """Test unloading when CUDA cache clearing fails."""
        processor = LLMProcessor()
        processor.model = Mock()

        with (
            patch("src.processors.llm.processor.torch") as mock_torch,
            patch("src.processors.llm.processor.has_cuda", return_value=True),
        ):
            mock_torch.cuda.empty_cache.side_effect = Exception("CUDA error")

            # Should not raise exception
            processor.unload()

            assert processor.model is None


class TestGetVersionInfo:
    """Test version information functionality."""

    def test_get_version_info_with_model_info(self):
        """Test version info with loaded model info."""
        processor = LLMProcessor()
        processor.model_info = {"model_name": "test-model"}
        processor.checkpoint_hash = "test-hash"
        processor.current_template_id = "template"
        processor.current_schema_hash = "schema-hash"

        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.llm_enhance_model = "default-model"
            mock_settings.llm_enhance = SimpleNamespace(model="default-model")
            mock_settings.llm_sum = SimpleNamespace(
                temperature=0.7,
                top_p=0.9,
                top_k=20,
                max_tokens=1000,
                structured_outputs_enabled=True,
                structured_outputs_backend="guidance",
            )

            info = processor.get_version_info()

            assert info["name"] == "test-model"
            assert info["checkpoint_hash"] == "test-hash"
            assert info["quantization"] == "awq-4bit"
            assert info["structured_output"]["backend"] == "guidance"
            assert info["structured_output"]["schema_id"] == "template"
            assert info["structured_output"]["schema_hash"] == "schema-hash"

    def test_get_version_info_without_model_info(self):
        """Test version info without model info."""
        processor = LLMProcessor()
        processor.model_info = None
        processor.checkpoint_hash = None

        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.llm_enhance_model = "default-model"
            mock_settings.llm_enhance = SimpleNamespace(model="default-model")
            mock_settings.llm_sum = SimpleNamespace(
                temperature=0.7,
                top_p=0.9,
                top_k=20,
                max_tokens=1000,
                structured_outputs_enabled=False,
                structured_outputs_backend="xgrammar",
            )

            info = processor.get_version_info()

            assert info["name"] == "default-model"
            assert info["checkpoint_hash"] == "unknown"
            assert info["structured_output"]["schema_id"] == "none"
            assert info["structured_output"]["schema_hash"] == "none"
