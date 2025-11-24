"""
Integration tests for LLM processor functionality.

Tests cover end-to-end workflows with real vLLM integration where possible,
and comprehensive mocking for CI environments.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.processors.base import LLMResult
from src.processors.llm.processor import LLMProcessor


class TestLLMIntegration:
    """Integration tests for LLM processor workflows."""

    @staticmethod
    def _configure_llm_settings(
        mock_settings,
        templates_dir,
        *,
        enhance_overrides: dict | None = None,
        summary_overrides: dict | None = None,
        verbose: bool = False,
    ) -> None:
        """Configure nested LLM settings for tests."""
        enhance_defaults = {
            "model": "test-model",
            "temperature": 0.0,
            "top_p": 0.9,
            "top_k": 20,
            "max_tokens": 1000,
            "quantization": None,
            "gpu_memory_utilization": 0.95,
            "max_model_len": 32768,
            "max_num_seqs": None,
            "max_num_batched_tokens": None,
            "max_num_partial_prefills": None,
        }
        summary_defaults = {
            "model": "test-model",
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 20,
            "max_tokens": 1000,
            "quantization": None,
            "gpu_memory_utilization": 0.95,
            "max_model_len": 32768,
            "max_num_seqs": None,
            "max_num_batched_tokens": None,
            "max_num_partial_prefills": None,
            "structured_outputs_enabled": True,
            "structured_outputs_backend": "xgrammar",
        }

        if enhance_overrides:
            enhance_defaults.update(enhance_overrides)
        if summary_overrides:
            summary_defaults.update(summary_overrides)

        mock_settings.paths.templates_dir = templates_dir
        mock_settings.llm_enhance_model = enhance_defaults["model"]
        mock_settings.llm_sum_model = summary_defaults["model"]
        mock_settings.llm_enhance = SimpleNamespace(**enhance_defaults)
        mock_settings.llm_sum = SimpleNamespace(**summary_defaults)
        mock_settings.verbose_components = verbose

    def test_end_to_end_enhancement_pipeline(self, tmp_path):
        """Test complete text enhancement pipeline."""
        # Create template directory and files
        templates_dir = tmp_path / "templates" / "prompts"
        templates_dir.mkdir(parents=True)

        # Create enhancement template
        enhancement_template = templates_dir / "text_enhancement_v1.jinja"
        enhancement_template.write_text(
            """
        Please enhance the following transcript by adding proper punctuation and capitalization:
        
        {{ text_input }}
        """
        )

        # Mock vLLM components
        mock_output = Mock()
        mock_output.outputs = [Mock()]
        inner_output = Mock()
        inner_output.text = "Enhanced text with proper punctuation and capitalization."
        inner_output.token_ids = [1, 2, 3]
        inner_output.finish_reason = "stop"
        mock_output.outputs = [inner_output]
        mock_output.prompt_token_ids = [11, 12]

        mock_model = Mock()
        mock_model.get_default_sampling_params.return_value = Mock()
        mock_model.generate.return_value = [mock_output]
        mock_model.chat.return_value = [mock_output]

        with patch("vllm.LLM") as mock_llm_class:
            mock_llm_class.return_value = mock_model

            with patch("src.processors.llm.calculate_checkpoint_hash") as mock_hash:
                mock_hash.return_value = "test-hash"

                with patch("src.processors.llm.get_model_info") as mock_info:
                    mock_info.return_value = {"model_name": "test-model"}

                    with (
                        patch("src.processors.llm.processor.settings") as mock_settings,
                        patch(
                            "src.processors.llm.processor.has_cuda", return_value=True
                        ),
                        patch(
                            "src.processors.llm.processor.apply_overrides_to_sampling"
                        ) as mock_apply,
                    ):
                        mock_apply.return_value = Mock()
                        self._configure_llm_settings(
                            mock_settings,
                            tmp_path / "templates",
                            verbose=True,
                        )

                        # Create processor after mocking settings
                        processor = LLMProcessor()

                        # Test enhancement
                        result = processor.enhance_text("test text without punctuation")

                        assert (
                            result["enhanced_text"]
                            == "Enhanced text with proper punctuation and capitalization."
                        )
                        assert result["enhancement_applied"] is True
                        assert result["edit_distance"] > 0
                        assert "edit_rate" in result
                        assert "model_info" in result

    def test_end_to_end_summarization_pipeline(self, tmp_path):
        """Test complete structured summarization pipeline."""
        # Create template directory and files
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()

        # Create schema file
        schemas_dir = templates_dir / "schemas"
        schemas_dir.mkdir()
        schema_file = schemas_dir / "meeting_notes_v1.json"
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string", "maxLength": 200},
                "abstract": {"type": "string", "maxLength": 1000},
                "main_points": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 500},
                    "minItems": 1,
                    "maxItems": 20,
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 50},
                    "minItems": 1,
                    "maxItems": 10,
                },
            },
            "required": ["title", "abstract", "main_points", "tags"],
            "additionalProperties": False,
        }
        schema_file.write_text(json.dumps(schema, indent=2))

        # Create prompt template
        prompts_dir = templates_dir / "prompts"
        prompts_dir.mkdir()
        prompt_template = prompts_dir / "meeting_notes_v1.jinja"
        prompt_template.write_text(
            """
        Please summarize the following meeting transcript in the specified JSON format:
        
        Transcript:
        {{ transcript }}
        
        Schema:
        {{ schema }}
        
        Generate a structured summary with the following fields:
        - title: Brief, descriptive title
        - abstract: 2-3 sentence summary
        - main_points: Key discussion points (array)
        - tags: Category tags (array, 1-10 items)
        """
        )

        processor = LLMProcessor()

        # Mock vLLM components
        expected_summary = {
            "title": "Weekly Team Meeting",
            "abstract": "Discussion of project progress and upcoming deadlines.",
            "main_points": [
                "Project A is on track for Q1 delivery",
                "Need to address resource constraints for Project B",
                "New team member onboarding scheduled for next week",
            ],
            "tags": ["meeting", "project-update", "team"],
        }

        inner_output = Mock()
        inner_output.text = json.dumps(expected_summary)
        inner_output.token_ids = [1, 2, 3, 4]
        inner_output.finish_reason = "stop"
        mock_output = Mock()
        mock_output.outputs = [inner_output]
        mock_output.prompt_token_ids = [10, 20, 30]

        mock_model = Mock()
        mock_model.get_default_sampling_params.return_value = Mock()
        mock_model.generate.return_value = [mock_output]

        with patch("vllm.LLM") as mock_llm_class:
            mock_llm_class.return_value = mock_model

            with patch("src.processors.llm.calculate_checkpoint_hash") as mock_hash:
                mock_hash.return_value = "test-hash"

                with patch("src.processors.llm.get_model_info") as mock_info:
                    mock_info.return_value = {"model_name": "test-model"}

                    with (
                        patch("src.processors.llm.processor.settings") as mock_settings,
                        patch(
                            "src.processors.llm.processor.has_cuda", return_value=True
                        ),
                        patch(
                            "src.processors.llm.processor.apply_overrides_to_sampling"
                        ) as mock_apply,
                        patch(
                            "src.processors.llm.processor.validate_llm_output"
                        ) as mock_validate,
                    ):
                        mock_apply.return_value = Mock()
                        mock_validate.return_value = (expected_summary, None)
                        self._configure_llm_settings(
                            mock_settings, templates_dir, verbose=False
                        )

                        # Test summarization
                        transcript = "Today we discussed project progress. Project A is on track. We need more resources for Project B. New team member starts next week."
                        with patch.object(
                            processor,
                            "execute",
                            return_value=LLMResult(
                                text=json.dumps(expected_summary),
                                tokens_used=256,
                                model_info={"model_name": "test-model"},
                                metadata={"task": "summary"},
                            ),
                        ) as mock_execute:
                            result = processor.generate_summary(
                                transcript, "meeting_notes_v1"
                            )

                        mock_execute.assert_called_once()

                        assert result["summary"] is not None
                        assert result["summary"]["title"] == "Weekly Team Meeting"
                        assert (
                            result["summary"]["abstract"]
                            == "Discussion of project progress and upcoming deadlines."
                        )
                        assert len(result["summary"]["main_points"]) == 3
                        assert len(result["summary"]["tags"]) == 3
                        assert result["retry_count"] == 0
                        assert "model_info" in result

    def test_sequential_load_execute_unload_cycle(self, tmp_path):
        """Test sequential model loading, execution, and unloading."""
        # Create minimal template
        templates_dir = tmp_path / "templates" / "prompts"
        templates_dir.mkdir(parents=True)
        enhancement_template = templates_dir / "text_enhancement_v1.jinja"
        enhancement_template.write_text("{{ text_input }}")

        processor = LLMProcessor()

        # Mock vLLM components
        inner_output = Mock()
        inner_output.text = "Enhanced text"
        inner_output.token_ids = [1, 2]
        inner_output.finish_reason = "stop"
        mock_output = Mock()
        mock_output.outputs = [inner_output]
        mock_output.prompt_token_ids = [5, 6]

        mock_model = Mock()
        mock_model.get_default_sampling_params.return_value = Mock()
        mock_model.generate.return_value = [mock_output]
        # Add chat() method that returns a list for proper len() support
        mock_model.chat.return_value = [mock_output]

        with patch("vllm.LLM") as mock_llm_class:
            mock_llm_class.return_value = mock_model

            with patch("src.processors.llm.calculate_checkpoint_hash") as mock_hash:
                mock_hash.return_value = "test-hash"

                with patch("src.processors.llm.get_model_info") as mock_info:
                    mock_info.return_value = {"model_name": "test-model"}

                    with (
                        patch("src.processors.llm.processor.settings") as mock_settings,
                        patch(
                            "src.processors.llm.processor.has_cuda", return_value=True
                        ),
                        patch(
                            "src.processors.llm.processor.apply_overrides_to_sampling"
                        ) as mock_apply,
                    ):
                        mock_apply.return_value = Mock()
                        self._configure_llm_settings(
                            mock_settings, tmp_path / "templates", verbose=False
                        )

                        # Test cycle
                        assert processor.model is None
                        assert not processor._model_loaded

                        # Load and execute
                        result = processor.enhance_text("test text")
                        assert processor.model is not None
                        assert processor._model_loaded
                        assert result["enhanced_text"] == "Enhanced text"

                        # Unload
                        processor.unload()
                        assert processor.model is None
                        assert not processor._model_loaded

    @pytest.mark.integration
    def test_multiple_inferences_same_model_instance(self, tmp_path):
        """Test multiple inferences with the same model instance."""
        # Create minimal template
        templates_dir = tmp_path / "templates" / "prompts"
        templates_dir.mkdir(parents=True)
        enhancement_template = templates_dir / "text_enhancement_v1.jinja"
        enhancement_template.write_text("{{ text_input }}")

        # Mock vLLM components
        mock_model = Mock()
        mock_model.get_default_sampling_params.return_value = Mock()

        def make_output(text: str):
            inner = Mock()
            inner.text = text
            inner.token_ids = [1, 2, 3]
            inner.finish_reason = "stop"
            outer = Mock()
            outer.outputs = [inner]
            outer.prompt_token_ids = [7, 8, 9]
            return [outer]

        chat_outputs = [
            make_output("First enhanced text"),
            make_output("Second enhanced text"),
            make_output("Third enhanced text"),
        ]

        mock_model.generate.side_effect = [
            make_output("First enhanced text"),
            make_output("Second enhanced text"),
            make_output("Third enhanced text"),
        ]
        # Add chat() method with same outputs
        mock_model.chat.side_effect = chat_outputs

        with patch("vllm.LLM") as mock_llm_class:
            mock_llm_class.return_value = mock_model

            with patch("src.processors.llm.calculate_checkpoint_hash") as mock_hash:
                mock_hash.return_value = "test-hash"

                with patch("src.processors.llm.get_model_info") as mock_info:
                    mock_info.return_value = {"model_name": "test-model"}

                    with (
                        patch("src.processors.llm.processor.settings") as mock_settings,
                        patch(
                            "src.processors.llm.processor.has_cuda", return_value=True
                        ),
                        patch(
                            "src.processors.llm.processor.apply_overrides_to_sampling"
                        ) as mock_apply,
                    ):
                        mock_apply.return_value = Mock()
                        self._configure_llm_settings(
                            mock_settings, tmp_path / "templates", verbose=False
                        )

                        # Create processor after mocking settings
                        processor = LLMProcessor()

                        # Multiple inferences
                        result1 = processor.enhance_text("first text")
                        result2 = processor.enhance_text("second text")
                        result3 = processor.enhance_text("third text")

                        assert result1["enhanced_text"] == "First enhanced text"
                        assert result2["enhanced_text"] == "Second enhanced text"
                        assert result3["enhanced_text"] == "Third enhanced text"

                        # Model should only be loaded once
                        assert mock_llm_class.call_count == 1
                        assert mock_model.chat.call_count == 3

    @pytest.mark.integration
    def test_error_handling_and_recovery(self, tmp_path):
        """Test error handling and recovery mechanisms."""
        # Create minimal template
        templates_dir = tmp_path / "templates" / "prompts"
        templates_dir.mkdir(parents=True)
        enhancement_template = templates_dir / "text_enhancement_v1.jinja"
        enhancement_template.write_text("{{ text_input }}")

        with (
            patch("src.processors.llm.processor.settings") as mock_settings,
            patch("src.processors.llm.processor.has_cuda", return_value=True),
            patch(
                "src.processors.llm.processor.apply_overrides_to_sampling"
            ) as mock_apply,
        ):
            mock_apply.return_value = Mock()
            self._configure_llm_settings(
                mock_settings, tmp_path / "templates", verbose=False
            )

            # Create processor after mocking settings
            processor = LLMProcessor()

            # Test vLLM not installed
            with patch("vllm.LLM", side_effect=ImportError("vLLM not installed")):
                result = processor.enhance_text("test text")

                assert result["enhanced_text"] == "test text"
                assert result["enhancement_applied"] is False
                assert processor.model is None
                assert processor.checkpoint_hash == "vllm_not_installed"

            # Test model loading error - force reload
            processor._model_loaded = False  # Reset model loaded flag
            processor.model = None
            with patch("vllm.LLM", side_effect=Exception("Load error")):
                with pytest.raises(Exception, match="Load error"):
                    processor.enhance_text("test text")

            # Test generation error - create fresh processor
            mock_model = Mock()
            mock_model.get_default_sampling_params.return_value = Mock()
            mock_model.generate.side_effect = Exception("Generation error")
            # Also mock chat to raise the same error
            mock_model.chat.side_effect = Exception("Generation error")

            with patch("vllm.LLM") as mock_llm_class:
                mock_llm_class.return_value = mock_model

                with patch("src.processors.llm.calculate_checkpoint_hash") as mock_hash:
                    mock_hash.return_value = "test-hash"

                    with patch("src.processors.llm.get_model_info") as mock_info:
                        mock_info.return_value = {"model_name": "test-model"}

                        with (
                            patch(
                                "src.processors.llm.processor.settings"
                            ) as mock_settings,
                            patch(
                                "src.processors.llm.processor.has_cuda",
                                return_value=True,
                            ),
                            patch(
                                "src.processors.llm.processor.apply_overrides_to_sampling"
                            ) as mock_apply,
                        ):
                            mock_apply.return_value = Mock()
                            self._configure_llm_settings(
                                mock_settings, tmp_path / "templates", verbose=False
                            )

                            # Create processor after mocking settings
                            processor2 = LLMProcessor()

                            result = processor2.enhance_text("test text")

                            # Should return original text on generation error
                            assert result["enhanced_text"] == "test text"
                            assert (
                                result["enhancement_applied"] is True
                            )  # Enhancement was attempted

    @pytest.mark.integration
    def test_schema_validation_retry_mechanism(self, tmp_path):
        """Test schema validation retry mechanism."""
        # Create template directory and files
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()

        # Create schema file
        schemas_dir = templates_dir / "schemas"
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

        # Create prompt template
        prompts_dir = templates_dir / "prompts"
        prompts_dir.mkdir()
        prompt_template = prompts_dir / "meeting_notes_v1.jinja"
        prompt_template.write_text("{{ transcript }}")

        # Mock model with different outputs for retries
        mock_model = Mock()
        mock_model.get_default_sampling_params.return_value = Mock()

        # First call returns invalid JSON, second returns valid
        invalid_inner = Mock()
        invalid_inner.text = "invalid json"
        invalid_inner.token_ids = [1]
        invalid_inner.finish_reason = "stop"
        invalid_output = Mock()
        invalid_output.outputs = [invalid_inner]
        invalid_output.prompt_token_ids = [1]

        valid_inner = Mock()
        valid_inner.text = '{"title": "Test Meeting", "tags": ["meeting"]}'
        valid_inner.token_ids = [1, 2, 3]
        valid_inner.finish_reason = "stop"
        valid_output = Mock()
        valid_output.outputs = [valid_inner]
        valid_output.prompt_token_ids = [1, 2]

        outputs = [[invalid_output], [valid_output]]
        mock_model.generate.side_effect = outputs
        mock_model.chat.side_effect = outputs

        with patch("vllm.LLM") as mock_llm_class:
            mock_llm_class.return_value = mock_model

            with patch("src.processors.llm.calculate_checkpoint_hash") as mock_hash:
                mock_hash.return_value = "test-hash"

                with patch("src.processors.llm.get_model_info") as mock_info:
                    mock_info.return_value = {"model_name": "test-model"}

                    with (
                        patch("src.processors.llm.processor.settings") as mock_settings,
                        patch(
                            "src.processors.llm.processor.has_cuda", return_value=True
                        ),
                        patch(
                            "src.processors.llm.processor.apply_overrides_to_sampling"
                        ) as mock_apply,
                        patch(
                            "src.processors.llm.processor.validate_llm_output"
                        ) as mock_validate,
                    ):
                        mock_apply.return_value = Mock()
                        parsed_summary = {
                            "title": "Test Meeting",
                            "tags": ["meeting"],
                        }
                        mock_validate.side_effect = [
                            (None, "JSON decode error"),
                            (parsed_summary, None),
                        ]
                        self._configure_llm_settings(
                            mock_settings, templates_dir, verbose=False
                        )

                        # Create processor after mocking settings
                        processor = LLMProcessor()

                        with patch.object(
                            processor,
                            "execute",
                            side_effect=[
                                LLMResult(
                                    text="invalid json",
                                    tokens_used=100,
                                    model_info={"model_name": "test-model"},
                                    metadata={"task": "summary"},
                                ),
                                LLMResult(
                                    text=json.dumps(parsed_summary),
                                    tokens_used=120,
                                    model_info={"model_name": "test-model"},
                                    metadata={"task": "summary"},
                                ),
                            ],
                        ) as mock_execute:
                            result = processor.generate_summary(
                                "transcript", "meeting_notes_v1"
                            )

                        assert mock_execute.call_count == 2

                        assert result["summary"] == parsed_summary
                        assert result["retry_count"] == 1  # One retry

    @pytest.mark.integration
    def test_version_info_completeness(self, tmp_path):
        """Test that version info is complete and accurate."""
        # Create minimal template
        templates_dir = tmp_path / "templates" / "prompts"
        templates_dir.mkdir(parents=True)
        enhancement_template = templates_dir / "text_enhancement_v1.jinja"
        enhancement_template.write_text("{{ text_input }}")

        # Mock model loading with proper list support for chat()
        inner = Mock()
        inner.text = "test"
        inner.token_ids = [1, 2]
        inner.finish_reason = "stop"
        outer = Mock()
        outer.outputs = [inner]
        outer.prompt_token_ids = [3, 4]

        mock_model = Mock()
        mock_model.get_default_sampling_params.return_value = Mock()
        mock_model.generate.return_value = [outer]
        # Mock chat() to return a list-like object for proper len() support
        mock_model.chat.return_value = [outer]

        with patch("vllm.LLM") as mock_llm_class:
            mock_llm_class.return_value = mock_model

            with patch("src.processors.llm.calculate_checkpoint_hash") as mock_hash:
                mock_hash.return_value = "test-hash-12345"

                with patch("src.processors.llm.get_model_info") as mock_info:
                    mock_info.return_value = {
                        "model_name": "test-model",
                        "checkpoint_hash": "test-hash-12345",
                        "model_type": "qwen",
                    }

                    with (
                        patch("src.processors.llm.processor.settings") as mock_settings,
                        patch(
                            "src.processors.llm.processor.has_cuda", return_value=True
                        ),
                        patch(
                            "src.processors.llm.processor.apply_overrides_to_sampling"
                        ) as mock_apply,
                    ):
                        mock_apply.return_value = Mock()
                        self._configure_llm_settings(
                            mock_settings, tmp_path / "templates", verbose=False
                        )

                        # Create processor after mocking settings
                        processor = LLMProcessor()

                        # Load model
                        processor.enhance_text("test")

                        # Test version info
                        version_info = processor.get_version_info()

                        assert version_info["name"] == "test-model"
                        assert version_info["checkpoint_hash"] == "hf:test-model"
                        assert version_info["quantization"] == "awq-4bit"
                        assert version_info["thinking"] is False
                        assert version_info["reasoning_parser"] is None
                        assert (
                            version_info["structured_output"]["backend"]
                            == "xgrammar"
                        )
                        assert "decoding_params" in version_info
                        assert version_info["decoding_params"]["temperature"] == 0.7
                        assert version_info["decoding_params"]["top_p"] == 0.9
                        assert version_info["decoding_params"]["top_k"] == 20
                        assert version_info["decoding_params"]["max_tokens"] == 1000


class TestLLMCUDAMultiprocessingIntegration:
    """Test LLM loading after CUDA initialization to ensure spawn compatibility."""

    @patch("src.processors.llm.processor.has_cuda")
    @patch("torch.cuda.device_count")
    def test_llm_loads_after_cuda_initialization(
        self, mock_device_count, mock_has_cuda, tmp_path
    ):
        """Test that vLLM loads successfully after CUDA has been initialized."""
        # Setup - simulate CUDA being available and initialized
        mock_has_cuda.return_value = True
        mock_device_count.return_value = 1

        # Simulate prior CUDA usage by calling torch.cuda operations
        try:
            import torch
            from src.utils.device import has_cuda as _has_cuda

            # This simulates the ASR model loading that happens before LLM
            if _has_cuda() and hasattr(torch, "cuda"):
                torch.cuda.empty_cache()
        except ImportError:
            # Skip if torch not available
            pytest.skip("PyTorch not available")

        # Mock vLLM components
        mock_model = Mock()
        mock_model.get_default_sampling_params.return_value = Mock()
        mock_model.generate.return_value = [Mock(outputs=[Mock(text="test output")])]

        templates_dir = tmp_path / "templates"
        (templates_dir / "prompts").mkdir(parents=True)

        with patch("vllm.LLM") as mock_llm_class:
            mock_llm_class.return_value = mock_model

            with patch("src.processors.llm.calculate_checkpoint_hash") as mock_hash:
                mock_hash.return_value = "test-hash"

                with patch("src.processors.llm.get_model_info") as mock_info:
                    mock_info.return_value = {
                        "model_name": "test-model",
                        "checkpoint_hash": "test-hash",
                    }

                    with (
                        patch("src.processors.llm.processor.settings") as mock_settings,
                        patch(
                            "src.processors.llm.processor.apply_overrides_to_sampling"
                        ) as mock_apply,
                    ):
                        mock_apply.return_value = Mock()
                        TestLLMIntegration._configure_llm_settings(
                            mock_settings,
                            templates_dir,
                            enhance_overrides={
                                "gpu_memory_utilization": 0.9,
                                "max_model_len": 16384,
                                "quantization": None,
                            },
                            verbose=False,
                        )

                        # Create processor and load model
                        processor = LLMProcessor()

                        # This should not raise RuntimeError about CUDA initialization
                        processor._load_model()

                        # Verify vLLM was called
                        mock_llm_class.assert_called_once()

                        # Verify model is loaded
                        assert processor._model_loaded is True
                        assert processor.model is not None
