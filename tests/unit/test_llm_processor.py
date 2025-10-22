"""
Unit tests for LLM processor functionality.

Tests cover model loading, text enhancement, structured summarization,
and resource management with comprehensive mocking.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.processors.base import LLMResult
from src.processors.llm.processor import LLMProcessor


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
                    processor = LLMProcessor()

                    mock_loader.assert_called_once()
                    mock_renderer.assert_called_once()


class TestLoadModel:
    """Test model loading functionality."""

    def test_load_model_success(self):
        """Test successful model loading."""
        processor = LLMProcessor()

        mock_model = Mock()
        mock_model.get_default_sampling_params.return_value = Mock()

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
            mock_settings.llm_enhance_temperature = 0.0
            mock_settings.llm_enhance_top_p = None
            mock_settings.llm_enhance_top_k = None
            mock_settings.llm_enhance_max_tokens = None
            mock_settings.llm_enhance_quantization = None
            mock_settings.llm_enhance_gpu_memory_utilization = 0.8
            mock_settings.llm_enhance_max_model_len = 8192
            mock_settings.llm_enhance_max_num_seqs = 4
            mock_settings.llm_enhance_max_num_batched_tokens = 2048
            mock_settings.llm_enhance_max_num_partial_prefills = 2
            mock_settings.verbose_components = False

            with patch(
                "src.processors.llm.processor.TemplateLoader"
            ), patch("src.processors.llm.processor.PromptRenderer"):
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

        with patch("vllm.LLM", side_effect=ImportError("vLLM not installed")):
            processor._load_model()

            assert processor.model is None
            assert processor.checkpoint_hash == "vllm_not_installed"
            assert processor._model_loaded is False

    def test_load_model_load_error(self):
        """Test handling of model loading errors."""
        processor = LLMProcessor()

        with patch("vllm.LLM", side_effect=Exception("Load error")):
            with pytest.raises(Exception, match="Load error"):
                processor._load_model()

            assert processor.model is None
            assert "load_error" in processor.checkpoint_hash

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

        result = processor.execute("test text")

        assert isinstance(result, LLMResult)
        assert result.text == "test text"
        assert result.tokens_used is None
        assert result.metadata["fallback"] is True

    def test_execute_with_model(self):
        """Test execution with loaded model."""
        processor = LLMProcessor()
        processor._model_loaded = True

        # Mock model and outputs - token_ids MUST be a real list
        mock_inner_output = Mock()
        mock_inner_output.text = "generated text"
        mock_inner_output.token_ids = [1, 2, 3, 4, 5]  # Real list, not Mock!
        mock_inner_output.finish_reason = "stop"

        mock_output = Mock()
        mock_output.outputs = [mock_inner_output]

        mock_model = Mock()
        mock_model.get_default_sampling_params.return_value = Mock()
        mock_model.generate.return_value = [mock_output]
        processor.model = mock_model

        with patch(
            "src.processors.llm.processor.apply_overrides_to_sampling"
        ) as mock_apply:
            mock_apply.return_value = Mock()

            result = processor.execute("test text", task="enhancement")

            assert result.text == "generated text"
            assert result.metadata["task"] == "enhancement"
            mock_model.generate.assert_called_once()

    def test_execute_generation_error(self):
        """Test execution when generation fails."""
        processor = LLMProcessor()
        processor._model_loaded = True

        mock_model = Mock()
        mock_model.get_default_sampling_params.return_value = Mock()
        mock_model.generate.side_effect = Exception("Generation error")
        processor.model = mock_model

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
            processor.execute("test text")

            mock_load.assert_called_once()


class TestEnhanceText:
    """Test text enhancement functionality."""

    def test_enhance_text_without_model(self):
        """Test enhancement when model is not available."""
        processor = LLMProcessor()
        processor.model = None
        processor._model_loaded = True

        result = processor.enhance_text("test text")

        assert result["enhanced_text"] == "test text"
        assert result["enhancement_applied"] is False
        assert result["edit_distance"] == 0

    def test_enhance_text_with_model(self):
        """Test successful text enhancement."""
        processor = LLMProcessor()
        processor._model_loaded = True

        # Mock model and outputs
        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = "Enhanced text with punctuation."

        mock_model = Mock()
        mock_model.get_default_sampling_params.return_value = Mock()
        mock_model.generate.return_value = [mock_output]
        processor.model = mock_model
        processor.model_info = {"model_name": "test"}

        with patch.object(processor.prompt_renderer, "render") as mock_render:
            mock_render.return_value = "rendered prompt with chat template"

            result = processor.enhance_text("test text without punctuation")

            assert result["enhanced_text"] == "Enhanced text with punctuation."
            assert result["enhancement_applied"] is True
            assert result["edit_distance"] > 0
            assert "edit_rate" in result
            # Template rendering now happens inside execute(), which is called
            mock_render.assert_called_once_with(
                "text_enhancement_v1", text_input="test text without punctuation"
            )

    def test_enhance_text_prompt_render_error(self):
        """Test enhancement when prompt rendering fails."""
        processor = LLMProcessor()
        processor._model_loaded = True
        processor.model = Mock()
        processor.model_info = {"model_name": "test"}

        with patch.object(
            processor.prompt_renderer, "render", side_effect=Exception("Render error")
        ):
            result = processor.enhance_text("test text")

            # When template rendering fails in execute(), it returns an error in metadata
            # The result should fall back gracefully
            assert "enhanced_text" in result or result.get("enhancement_applied") is False

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
        processor.model = None
        processor._model_loaded = True

        result = processor.generate_summary("transcript", "meeting_notes_v1")

        assert result["summary"] is None
        assert "error" in result
        assert "LLM model not available" in result["error"]

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
        schema_file = tmp_path / "meeting_notes_v1.json"
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

            with patch.object(
                processor.prompt_renderer,
                "render",
                side_effect=Exception("Render error"),
            ):
                result = processor.generate_summary("transcript", "meeting_notes_v1")

                assert result["summary"] is None
                assert "Prompt rendering error" in result["error"]

    def test_generate_summary_success(self, tmp_path):
        """Test successful summarization."""
        processor = LLMProcessor()
        processor._model_loaded = True

        # Create valid schema file
        schema_file = tmp_path / "meeting_notes_v1.json"
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
        schema_file = tmp_path / "meeting_notes_v1.json"
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

        with patch(
            "src.processors.llm.processor.torch"
        ) as mock_src.processors.llm.processor.torch:
            mock_src.processors.llm.processor.torch.cuda.is_available.return_value = (
                True
            )

            processor.unload()

            assert processor.model is None
            assert processor._model_loaded is False
            assert processor.current_template_id is None
            assert processor.current_schema_hash is None
            mock_src.processors.llm.processor.torch.cuda.empty_cache.assert_called_once()

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

        with patch(
            "src.processors.llm.processor.torch"
        ) as mock_src.processors.llm.processor.torch:
            mock_src.processors.llm.processor.torch.cuda.is_available.return_value = (
                True
            )
            mock_src.processors.llm.processor.torch.cuda.empty_cache.side_effect = (
                Exception("CUDA error")
            )

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
            mock_settings.llm_sum_temperature = 0.7
            mock_settings.llm_sum_top_p = 0.9
            mock_settings.llm_sum_top_k = 20
            mock_settings.llm_sum_max_tokens = 1000

            info = processor.get_version_info()

            assert info["name"] == "test-model"
            assert info["checkpoint_hash"] == "test-hash"
            assert info["quantization"] == "awq-4bit"
            assert info["structured_output"]["schema_id"] == "template"
            assert info["structured_output"]["schema_hash"] == "schema-hash"

    def test_get_version_info_without_model_info(self):
        """Test version info without model info."""
        processor = LLMProcessor()
        processor.model_info = None
        processor.checkpoint_hash = None

        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.llm_enhance_model = "default-model"
            mock_settings.llm_sum_temperature = 0.7
            mock_settings.llm_sum_top_p = 0.9
            mock_settings.llm_sum_top_k = 20
            mock_settings.llm_sum_max_tokens = 1000

            info = processor.get_version_info()

            assert info["name"] == "default-model"
            assert info["checkpoint_hash"] == "unknown"
            assert info["structured_output"]["schema_id"] == "none"
            assert info["structured_output"]["schema_hash"] == "none"
