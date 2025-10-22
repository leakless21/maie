"""
Unit tests for dynamic token calculation functionality.

Tests the calculate_dynamic_max_tokens function and LLMProcessor tokenizer integration.
"""

import pytest
from unittest.mock import Mock, patch

from src.processors.llm.config import calculate_dynamic_max_tokens
from src.processors.llm.processor import LLMProcessor


class TestCalculateDynamicMaxTokens:
    """Test the calculate_dynamic_max_tokens function."""

    def test_enhancement_task_ratio(self):
        """Test enhancement task uses 1:1 + 10% buffer ratio."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        
        result = calculate_dynamic_max_tokens(
            input_text="test input",
            tokenizer=mock_tokenizer,
            task="enhancement",
            max_model_len=1000,
            user_override=None
        )
        
        # Enhancement: 5 * 1.1 + 64 = 69.5 -> 69, but min 256
        assert result == 256  # min_tokens for enhancement
        
        # Verify tokenizer was called with add_special_tokens=False
        mock_tokenizer.encode.assert_called_once_with("test input", add_special_tokens=False)

    def test_summarization_task_ratio(self):
        """Test summarization task uses 30% compression ratio."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10 tokens
        
        result = calculate_dynamic_max_tokens(
            input_text="test input",
            tokenizer=mock_tokenizer,
            task="summary",
            max_model_len=1000,
            user_override=None
        )
        
        # Summarization: 10 * 0.3 = 3, but min 128
        assert result == 128  # min_tokens for summarization

    def test_large_input_enhancement(self):
        """Test enhancement with large input that would exceed context."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1] * 1000  # 1000 tokens
        
        result = calculate_dynamic_max_tokens(
            input_text="large input",
            tokenizer=mock_tokenizer,
            task="enhancement",
            max_model_len=2000,  # 2000 - 1000 - 128 = 872 available
            user_override=None
        )
        
        # Should be clamped to available_tokens (872)
        assert result == 872

    def test_large_input_summarization(self):
        """Test summarization with large input."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1] * 1000  # 1000 tokens
        
        result = calculate_dynamic_max_tokens(
            input_text="large input",
            tokenizer=mock_tokenizer,
            task="summary",
            max_model_len=2000,  # 2000 - 1000 - 128 = 872 available
            user_override=None
        )
        
        # Summarization: 1000 * 0.3 = 300, but max is 872 * 0.5 = 436
        # So result should be min(300, 436) = 300
        assert result == 300

    def test_negative_available_tokens(self):
        """Test handling when available_tokens is negative."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1] * 2000  # 2000 tokens
        
        result = calculate_dynamic_max_tokens(
            input_text="very large input",
            tokenizer=mock_tokenizer,
            task="enhancement",
            max_model_len=1000,  # 1000 - 2000 - 128 = negative
            user_override=None
        )
        
        # Should fall back to min_tokens
        assert result == 256

    def test_user_override(self):
        """Test that user_override takes precedence."""
        mock_tokenizer = Mock()
        
        result = calculate_dynamic_max_tokens(
            input_text="test",
            tokenizer=mock_tokenizer,
            task="enhancement",
            max_model_len=1000,
            user_override=500
        )
        
        assert result == 500
        # Tokenizer should not be called when override is provided
        mock_tokenizer.encode.assert_not_called()

    def test_tokenizer_without_add_special_tokens(self):
        """Test fallback when tokenizer doesn't support add_special_tokens parameter."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = [
            TypeError("add_special_tokens not supported"),  # First call fails
            [1, 2, 3, 4, 5]  # Second call succeeds
        ]
        
        result = calculate_dynamic_max_tokens(
            input_text="test input",
            tokenizer=mock_tokenizer,
            task="enhancement",
            max_model_len=1000,
            user_override=None
        )
        
        assert result == 256
        # Should have tried both calls
        assert mock_tokenizer.encode.call_count == 2
        mock_tokenizer.encode.assert_any_call("test input", add_special_tokens=False)
        mock_tokenizer.encode.assert_any_call("test input")

    def test_very_small_input(self):
        """Test with very small input that results in small token counts."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2]  # 2 tokens
        
        result = calculate_dynamic_max_tokens(
            input_text="hi",
            tokenizer=mock_tokenizer,
            task="enhancement",
            max_model_len=1000,
            user_override=None
        )
        
        # Enhancement: 2 * 1.1 + 64 = 66.2 -> 66, but min 256
        assert result == 256


class TestLLMProcessorTokenizerIntegration:
    """Test LLMProcessor tokenizer integration."""

    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_ensure_tokenizer_with_hf_fallback(self, mock_from_pretrained):
        """Test _ensure_tokenizer uses HF tokenizer when vLLM tokenizer unavailable."""
        # Mock HF tokenizer
        mock_hf_tokenizer = Mock()
        mock_from_pretrained.return_value = mock_hf_tokenizer
        
        processor = LLMProcessor()
        processor.model = Mock()
        processor.model.get_tokenizer = None
        processor._ensure_tokenizer("test-model")
        
        assert processor.tokenizer == mock_hf_tokenizer
        mock_from_pretrained.assert_called_once_with(
            "test-model", trust_remote_code=True
        )

    def test_ensure_tokenizer_with_vllm_tokenizer(self):
        """Test _ensure_tokenizer uses vLLM tokenizer when available."""
        # Mock vLLM model with get_tokenizer
        mock_vllm_tokenizer = Mock()
        mock_model = Mock()
        mock_model.get_tokenizer.return_value = mock_vllm_tokenizer
        
        processor = LLMProcessor()
        processor.model = mock_model
        processor._ensure_tokenizer("test-model")
        
        assert processor.tokenizer == mock_vllm_tokenizer

    def test_ensure_tokenizer_already_set(self):
        """Test _ensure_tokenizer does nothing when tokenizer already set."""
        processor = LLMProcessor()
        processor.tokenizer = Mock()
        
        processor._ensure_tokenizer("test-model")
        
        # Should not have called any tokenizer loading
        assert processor.tokenizer is not None

    @patch('src.processors.llm.processor.calculate_dynamic_max_tokens')
    def test_execute_dynamic_tokens_calculation(self, mock_calc_tokens):
        """Test that execute calls dynamic token calculation when tokenizer available."""
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        
        # Mock generation
        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = "generated text"
        mock_model.generate.return_value = [mock_output]
        
        processor = LLMProcessor()
        processor.model = mock_model
        processor.tokenizer = mock_tokenizer
        processor._model_loaded = True
        
        # Mock calculate_dynamic_max_tokens
        mock_calc_tokens.return_value = 500
        
        result = processor.execute("test input", task="enhancement")
        
        # Should have called dynamic token calculation
        mock_calc_tokens.assert_called_once()
        assert result.text == "generated text"

    def test_execute_no_tokenizer_skip_dynamic(self):
        """Test that execute skips dynamic calculation when no tokenizer."""
        # Mock model without tokenizer
        mock_model = Mock()
        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = "generated text"
        mock_model.generate.return_value = [mock_output]
        
        processor = LLMProcessor()
        processor.model = mock_model
        processor.tokenizer = None  # No tokenizer
        processor._model_loaded = True
        
        result = processor.execute("test input", task="enhancement")
        
        # Should still work but without dynamic calculation
        assert result.text == "generated text"

    def test_execute_chat_template_formatting(self):
        """Test that execute applies chat template formatting when appropriate."""
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "formatted chat prompt"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        
        # Mock generation
        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = "generated text"
        mock_model.generate.return_value = [mock_output]
        
        processor = LLMProcessor()
        processor.model = mock_model
        processor.tokenizer = mock_tokenizer
        processor._model_loaded = True
        
        # Test with conversation-like input
        conversation_input = "user: Hello\nassistant: Hi there"
        result = processor.execute(conversation_input, task="enhancement")
        
        # Should have called generate with formatted prompt
        mock_model.generate.assert_called_once()
        call_args = mock_model.generate.call_args
        assert "formatted chat prompt" in str(call_args)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_calculate_dynamic_tokens_with_none_tokenizer(self):
        """Test calculate_dynamic_max_tokens handles None tokenizer gracefully."""
        with pytest.raises(AttributeError):
            calculate_dynamic_max_tokens(
                input_text="test",
                tokenizer=None,
                task="enhancement",
                max_model_len=1000
            )

    def test_calculate_dynamic_tokens_invalid_task(self):
        """Test calculate_dynamic_max_tokens with invalid task defaults to summarization."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        
        result = calculate_dynamic_max_tokens(
            input_text="test",
            tokenizer=mock_tokenizer,
            task="invalid_task",
            max_model_len=1000
        )
        
        # Should default to summarization behavior (30% ratio)
        assert result == 128  # min_tokens for summarization

    def test_very_large_model_len(self):
        """Test with very large model length."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        
        result = calculate_dynamic_max_tokens(
            input_text="test",
            tokenizer=mock_tokenizer,
            task="enhancement",
            max_model_len=1000000  # Very large context
        )
        
        # Should work normally with large context
        assert result >= 256  # At least min_tokens
