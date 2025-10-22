"""
Integration tests for LLM processor functionality.

Tests cover end-to-end workflows with real vLLM integration where possible,
and comprehensive mocking for CI environments.
"""

import json
from unittest.mock import Mock, patch

import pytest

from src.processors.llm.processor import LLMProcessor


class TestLLMIntegration:
    """Integration tests for LLM processor workflows."""

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
        mock_output.outputs[
            0
        ].text = "Enhanced text with proper punctuation and capitalization."

        mock_model = Mock()
        mock_model.get_default_sampling_params.return_value = Mock()
        mock_model.generate.return_value = [mock_output]

        with patch("vllm.LLM") as mock_llm_class:
            mock_llm_class.return_value = mock_model

            with patch("src.processors.llm.calculate_checkpoint_hash") as mock_hash:
                mock_hash.return_value = "test-hash"

                with patch("src.processors.llm.get_model_info") as mock_info:
                    mock_info.return_value = {"model_name": "test-model"}

                    with patch(
                        "src.processors.llm.processor.settings"
                    ) as mock_settings:
                        mock_settings.paths.templates_dir = tmp_path / "templates"
                        mock_settings.llm_enhance_model = "test-model"
                        mock_settings.llm_enhance_temperature = 0.0
                        mock_settings.llm_enhance_top_p = 0.9
                        mock_settings.llm_enhance_top_k = 20
                        mock_settings.llm_enhance_max_tokens = 1000
                        mock_settings.llm_enhance_gpu_memory_utilization = 0.95
                        mock_settings.llm_enhance_max_model_len = 32768

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
        schema_file = templates_dir / "meeting_notes_v1.json"
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
                    "maxItems": 5,
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
        - tags: Category tags (array, 1-5 items)
        """
        )

        processor = LLMProcessor()

        # Mock vLLM components
        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = json.dumps(
            {
                "title": "Weekly Team Meeting",
                "abstract": "Discussion of project progress and upcoming deadlines.",
                "main_points": [
                    "Project A is on track for Q1 delivery",
                    "Need to address resource constraints for Project B",
                    "New team member onboarding scheduled for next week",
                ],
                "tags": ["meeting", "project-update", "team"],
            }
        )

        mock_model = Mock()
        mock_model.get_default_sampling_params.return_value = Mock()
        mock_model.generate.return_value = [mock_output]

        with patch("vllm.LLM") as mock_llm_class:
            mock_llm_class.return_value = mock_model

            with patch("src.processors.llm.calculate_checkpoint_hash") as mock_hash:
                mock_hash.return_value = "test-hash"

                with patch("src.processors.llm.get_model_info") as mock_info:
                    mock_info.return_value = {"model_name": "test-model"}

                    with patch(
                        "src.processors.llm.processor.settings"
                    ) as mock_settings:
                        mock_settings.paths.templates_dir = templates_dir
                        mock_settings.llm_sum_model = "test-model"
                        mock_settings.llm_sum_temperature = 0.7
                        mock_settings.llm_sum_top_p = 0.9
                        mock_settings.llm_sum_top_k = 20
                        mock_settings.llm_sum_max_tokens = 1000
                        mock_settings.llm_enhance_gpu_memory_utilization = 0.95
                        mock_settings.llm_enhance_max_model_len = 32768

                        # Test summarization
                        transcript = "Today we discussed project progress. Project A is on track. We need more resources for Project B. New team member starts next week."
                        result = processor.generate_summary(
                            transcript, "meeting_notes_v1"
                        )

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
        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = "Enhanced text"

        mock_model = Mock()
        mock_model.get_default_sampling_params.return_value = Mock()
        mock_model.generate.return_value = [mock_output]

        with patch("vllm.LLM") as mock_llm_class:
            mock_llm_class.return_value = mock_model

            with patch("src.processors.llm.calculate_checkpoint_hash") as mock_hash:
                mock_hash.return_value = "test-hash"

                with patch("src.processors.llm.get_model_info") as mock_info:
                    mock_info.return_value = {"model_name": "test-model"}

                    with patch(
                        "src.processors.llm.processor.settings"
                    ) as mock_settings:
                        mock_settings.paths.templates_dir = tmp_path / "templates"
                        mock_settings.llm_enhance_model = "test-model"
                        mock_settings.llm_enhance_temperature = 0.0
                        mock_settings.llm_enhance_top_p = 0.9
                        mock_settings.llm_enhance_top_k = 20
                        mock_settings.llm_enhance_max_tokens = 1000
                        mock_settings.llm_enhance_gpu_memory_utilization = 0.95
                        mock_settings.llm_enhance_max_model_len = 32768

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

        # Different outputs for different calls
        outputs = [
            [Mock(outputs=[Mock(text="First enhanced text")])],
            [Mock(outputs=[Mock(text="Second enhanced text")])],
            [Mock(outputs=[Mock(text="Third enhanced text")])],
        ]
        mock_model.generate.side_effect = outputs

        with patch("vllm.LLM") as mock_llm_class:
            mock_llm_class.return_value = mock_model

            with patch("src.processors.llm.calculate_checkpoint_hash") as mock_hash:
                mock_hash.return_value = "test-hash"

                with patch("src.processors.llm.get_model_info") as mock_info:
                    mock_info.return_value = {"model_name": "test-model"}

                    with patch(
                        "src.processors.llm.processor.settings"
                    ) as mock_settings:
                        mock_settings.paths.templates_dir = tmp_path / "templates"
                        mock_settings.llm_enhance_model = "test-model"
                        mock_settings.llm_enhance_temperature = 0.0
                        mock_settings.llm_enhance_top_p = 0.9
                        mock_settings.llm_enhance_top_k = 20
                        mock_settings.llm_enhance_max_tokens = 1000
                        mock_settings.llm_enhance_gpu_memory_utilization = 0.95
                        mock_settings.llm_enhance_max_model_len = 32768

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
                        assert mock_model.generate.call_count == 3

    @pytest.mark.integration
    def test_error_handling_and_recovery(self, tmp_path):
        """Test error handling and recovery mechanisms."""
        # Create minimal template
        templates_dir = tmp_path / "templates" / "prompts"
        templates_dir.mkdir(parents=True)
        enhancement_template = templates_dir / "text_enhancement_v1.jinja"
        enhancement_template.write_text("{{ text_input }}")

        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.paths.templates_dir = tmp_path / "templates"
            mock_settings.llm_enhance_model = "test-model"
            mock_settings.llm_enhance_temperature = 0.0
            mock_settings.llm_enhance_top_p = 0.9
            mock_settings.llm_enhance_top_k = 20
            mock_settings.llm_enhance_max_tokens = 1000
            mock_settings.llm_enhance_gpu_memory_utilization = 0.95
            mock_settings.llm_enhance_max_model_len = 32768

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

            with patch("vllm.LLM") as mock_llm_class:
                mock_llm_class.return_value = mock_model

                with patch("src.processors.llm.calculate_checkpoint_hash") as mock_hash:
                    mock_hash.return_value = "test-hash"

                    with patch("src.processors.llm.get_model_info") as mock_info:
                        mock_info.return_value = {"model_name": "test-model"}

                        with patch(
                            "src.processors.llm.processor.settings"
                        ) as mock_settings:
                            mock_settings.paths.templates_dir = tmp_path / "templates"
                            mock_settings.llm_enhance_model = "test-model"
                            mock_settings.llm_enhance_temperature = 0.0
                            mock_settings.llm_enhance_top_p = 0.9
                            mock_settings.llm_enhance_top_k = 20
                            mock_settings.llm_enhance_max_tokens = 1000
                            mock_settings.llm_enhance_gpu_memory_utilization = 0.95
                            mock_settings.llm_enhance_max_model_len = 32768

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
        schema_file = templates_dir / "meeting_notes_v1.json"
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
        outputs = [
            [Mock(outputs=[Mock(text="invalid json")])],
            [
                Mock(
                    outputs=[
                        Mock(text='{"title": "Test Meeting", "tags": ["meeting"]}')
                    ]
                )
            ],
        ]
        mock_model.generate.side_effect = outputs

        with patch("vllm.LLM") as mock_llm_class:
            mock_llm_class.return_value = mock_model

            with patch("src.processors.llm.calculate_checkpoint_hash") as mock_hash:
                mock_hash.return_value = "test-hash"

                with patch("src.processors.llm.get_model_info") as mock_info:
                    mock_info.return_value = {"model_name": "test-model"}

                    with patch(
                        "src.processors.llm.processor.settings"
                    ) as mock_settings:
                        mock_settings.paths.templates_dir = templates_dir
                        mock_settings.llm_sum_model = "test-model"
                        mock_settings.llm_sum_temperature = 0.7
                        mock_settings.llm_sum_top_p = 0.9
                        mock_settings.llm_sum_top_k = 20
                        mock_settings.llm_sum_max_tokens = 1000
                        mock_settings.llm_enhance_gpu_memory_utilization = 0.95
                        mock_settings.llm_enhance_max_model_len = 32768

                        # Create processor after mocking settings
                        processor = LLMProcessor()

                        result = processor.generate_summary(
                            "transcript", "meeting_notes_v1"
                        )

                        assert result["summary"] == {
                            "title": "Test Meeting",
                            "tags": ["meeting"],
                        }
                        assert result["retry_count"] == 1  # One retry
                        assert (
                            mock_model.generate.call_count == 2
                        )  # Two generation calls

    @pytest.mark.integration
    def test_version_info_completeness(self, tmp_path):
        """Test that version info is complete and accurate."""
        # Create minimal template
        templates_dir = tmp_path / "templates" / "prompts"
        templates_dir.mkdir(parents=True)
        enhancement_template = templates_dir / "text_enhancement_v1.jinja"
        enhancement_template.write_text("{{ text_input }}")

        # Mock model loading
        mock_model = Mock()
        mock_model.get_default_sampling_params.return_value = Mock()
        mock_model.generate.return_value = [Mock(outputs=[Mock(text="test")])]

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

                    with patch(
                        "src.processors.llm.processor.settings"
                    ) as mock_settings:
                        mock_settings.paths.templates_dir = tmp_path / "templates"
                        mock_settings.llm_enhance_model = "test-model"
                        mock_settings.llm_sum_temperature = 0.7
                        mock_settings.llm_sum_top_p = 0.9
                        mock_settings.llm_sum_top_k = 20
                        mock_settings.llm_sum_max_tokens = 1000
                        mock_settings.llm_enhance_gpu_memory_utilization = 0.95
                        mock_settings.llm_enhance_max_model_len = 32768

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
                            == "json_schema"
                        )
                        assert "decoding_params" in version_info
                        assert version_info["decoding_params"]["temperature"] == 0.7
                        assert version_info["decoding_params"]["top_p"] == 0.9
                        assert version_info["decoding_params"]["top_k"] == 20
                        assert version_info["decoding_params"]["max_tokens"] == 1000


class TestLLMCUDAMultiprocessingIntegration:
    """Test LLM loading after CUDA initialization to ensure spawn compatibility."""

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.device_count")
    def test_llm_loads_after_cuda_initialization(self, mock_device_count, mock_cuda_available):
        """Test that vLLM loads successfully after CUDA has been initialized."""
        # Setup - simulate CUDA being available and initialized
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        
        # Simulate prior CUDA usage by calling torch.cuda operations
        try:
            import torch
            # This simulates the ASR model loading that happens before LLM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            # Skip if torch not available
            pytest.skip("PyTorch not available")
        
        # Mock vLLM components
        mock_model = Mock()
        mock_model.get_default_sampling_params.return_value = Mock()
        mock_model.generate.return_value = [Mock(outputs=[Mock(text="test output")])]
        
        with patch("vllm.LLM") as mock_llm_class:
            mock_llm_class.return_value = mock_model
            
            with patch("src.processors.llm.calculate_checkpoint_hash") as mock_hash:
                mock_hash.return_value = "test-hash"
                
                with patch("src.processors.llm.get_model_info") as mock_info:
                    mock_info.return_value = {
                        "model_name": "test-model",
                        "checkpoint_hash": "test-hash",
                    }
                    
                    with patch("src.processors.llm.processor.settings") as mock_settings:
                        mock_settings.llm_enhance_model = "test-model"
                        mock_settings.llm_enhance_gpu_memory_utilization = 0.9
                        mock_settings.llm_enhance_max_model_len = 16384
                        mock_settings.llm_enhance_quantization = None
                        mock_settings.verbose_components = False
                        
                        # Create processor and load model
                        processor = LLMProcessor()
                        
                        # This should not raise RuntimeError about CUDA initialization
                        processor._load_model()
                        
                        # Verify vLLM was called
                        mock_llm_class.assert_called_once()
                        
                        # Verify model is loaded
                        assert processor._model_loaded is True
                        assert processor.model is not None
