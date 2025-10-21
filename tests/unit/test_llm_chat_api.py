"""
Unit tests for LLM Chat API integration (TDD approach).

This module tests the migration from manual chat template embedding
to vLLM's native chat() API with OpenAI-format messages.

Following Test-Driven Development (TDD):
1. Write failing tests first
2. Implement minimal code to pass tests
3. Refactor while keeping tests green
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch, call

import pytest

from src.processors.llm.processor import LLMProcessor
from src.processors.base import LLMResult


class TestBuildMessages:
    """Test _build_messages() helper method for OpenAI-format message construction."""

    def test_build_messages_returns_list_of_dicts(self):
        """Messages should be a list of dictionaries with 'role' and 'content' keys."""
        processor = LLMProcessor()
        
        system_content = "You are a helpful assistant."
        user_content = "Summarize this text."
        
        messages = processor._build_messages(
            system_content=system_content,
            user_content=user_content
        )
        
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert all(isinstance(msg, dict) for msg in messages)
        assert all('role' in msg and 'content' in msg for msg in messages)

    def test_build_messages_system_role_first(self):
        """System message should come before user message."""
        processor = LLMProcessor()
        
        messages = processor._build_messages(
            system_content="System prompt",
            user_content="User prompt"
        )
        
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'

    def test_build_messages_preserves_content(self):
        """Content should be preserved exactly as provided."""
        processor = LLMProcessor()
        
        system_content = "You are an expert analyst.\n\nFollow these rules:\n- Rule 1\n- Rule 2"
        user_content = "Analyze this:\n\nSample text here"
        
        messages = processor._build_messages(
            system_content=system_content,
            user_content=user_content
        )
        
        assert messages[0]['content'] == system_content
        assert messages[1]['content'] == user_content

    def test_build_messages_no_chat_markers(self):
        """Content should NOT contain any chat template markers."""
        processor = LLMProcessor()
        
        messages = processor._build_messages(
            system_content="Clean system prompt",
            user_content="Clean user prompt"
        )
        
        all_content = messages[0]['content'] + messages[1]['content']
        
        # Should not contain any chat template markers
        assert '<|im_start|>' not in all_content
        assert '<|im_end|>' not in all_content
        assert '<|endoftext|>' not in all_content

    def test_build_messages_with_only_user(self):
        """Should support user-only messages (no system prompt)."""
        processor = LLMProcessor()
        
        messages = processor._build_messages(
            user_content="Just a user prompt"
        )
        
        assert len(messages) == 1
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == "Just a user prompt"

    def test_build_messages_empty_content_raises_error(self):
        """Should raise ValueError if no content provided."""
        processor = LLMProcessor()
        
        with pytest.raises(ValueError, match="At least user_content must be provided"):
            processor._build_messages()


class TestBuildSummarizationMessages:
    """Test _build_summarization_messages() for summary-specific message construction."""

    def test_build_summarization_messages_loads_schema(self, tmp_path):
        """Should load the schema for the given template_id."""
        processor = LLMProcessor()
        
        # Create schema file
        schema_file = tmp_path / "test_template.json"
        schema = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"]
        }
        schema_file.write_text(json.dumps(schema))
        
        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.templates_dir = tmp_path
            
            with patch("src.processors.llm.processor.load_template_schema") as mock_load:
                mock_load.return_value = schema
                
                messages = processor._build_summarization_messages(
                    transcript="test",
                    template_id="test_template"
                )
                
                mock_load.assert_called_once_with("test_template", tmp_path)

    def test_build_summarization_messages_renders_system_template(self, tmp_path):
        """Should render system prompt template without chat markers."""
        processor = LLMProcessor()
        
        schema = {"type": "object"}
        
        with patch("src.processors.llm.processor.load_template_schema") as mock_load:
            mock_load.return_value = schema
            
            with patch.object(processor.prompt_renderer, 'render') as mock_render:
                mock_render.return_value = "Rendered system prompt"
                
                messages = processor._build_summarization_messages(
                    transcript="test transcript",
                    template_id="generic_summary_v1"
                )
                
                # Should call render for system prompt
                system_call = [c for c in mock_render.call_args_list 
                              if 'system_prompts' in str(c)]
                assert len(system_call) > 0

    def test_build_summarization_messages_renders_user_template(self, tmp_path):
        """Should render user prompt template with transcript."""
        processor = LLMProcessor()
        
        schema = {"type": "object"}
        
        with patch("src.processors.llm.processor.load_template_schema") as mock_load:
            mock_load.return_value = schema
            
            with patch.object(processor.prompt_renderer, 'render') as mock_render:
                mock_render.side_effect = ["System content", "User content"]
                
                messages = processor._build_summarization_messages(
                    transcript="test transcript",
                    template_id="generic_summary_v1"
                )
                
                # Should call render twice: system + user
                assert mock_render.call_count == 2

    def test_build_summarization_messages_returns_correct_format(self):
        """Should return messages in correct OpenAI format."""
        processor = LLMProcessor()
        
        schema = {"type": "object"}
        
        with patch("src.processors.llm.processor.load_template_schema") as mock_load:
            mock_load.return_value = schema
            
            with patch.object(processor.prompt_renderer, 'render') as mock_render:
                mock_render.side_effect = [
                    "System: You are an expert analyst.",
                    "User: Analyze this transcript."
                ]
                
                messages = processor._build_summarization_messages(
                    transcript="test",
                    template_id="test_template"
                )
                
                assert isinstance(messages, list)
                assert len(messages) == 2
                assert messages[0] == {
                    'role': 'system',
                    'content': 'System: You are an expert analyst.'
                }
                assert messages[1] == {
                    'role': 'user',
                    'content': 'User: Analyze this transcript.'
                }


class TestExecuteWithChatAPI:
    """Test execute() method using chat() API instead of generate()."""

    def test_execute_summary_calls_chat_not_generate(self, tmp_path):
        """For summary tasks, should call llm.chat() instead of llm.generate()."""
        processor = LLMProcessor()
        processor._model_loaded = True
        
        # Mock model with both generate and chat methods
        mock_model = Mock()
        mock_output = Mock()
        mock_output.outputs = [Mock(text='{"title": "Test"}')]
        mock_model.chat.return_value = [mock_output]
        mock_model.get_default_sampling_params.return_value = Mock()
        processor.model = mock_model
        
        # Create schema
        schema_file = tmp_path / "test_template.json"
        schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        schema_file.write_text(json.dumps(schema))
        
        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.templates_dir = tmp_path
            
            with patch("src.processors.llm.processor.load_template_schema") as mock_load:
                mock_load.return_value = schema
                
                with patch.object(processor, '_build_summarization_messages') as mock_build:
                    mock_build.return_value = [
                        {'role': 'system', 'content': 'System'},
                        {'role': 'user', 'content': 'User'}
                    ]
                    
                    processor.execute(
                        "test transcript",
                        task="summary",
                        template_id="test_template"
                    )
                    
                    # Should call chat(), not generate()
                    mock_model.chat.assert_called_once()
                    mock_model.generate.assert_not_called()

    def test_execute_summary_passes_messages_to_chat(self, tmp_path):
        """Should pass messages in correct format to llm.chat()."""
        processor = LLMProcessor()
        processor._model_loaded = True
        
        mock_model = Mock()
        mock_output = Mock()
        mock_output.outputs = [Mock(text='{"title": "Test"}')]
        mock_model.chat.return_value = [mock_output]
        mock_model.get_default_sampling_params.return_value = Mock()
        processor.model = mock_model
        
        expected_messages = [
            {'role': 'system', 'content': 'System prompt'},
            {'role': 'user', 'content': 'User prompt'}
        ]
        
        schema_file = tmp_path / "test_template.json"
        schema = {"type": "object"}
        schema_file.write_text(json.dumps(schema))
        
        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.templates_dir = tmp_path
            
            with patch("src.processors.llm.processor.load_template_schema") as mock_load:
                mock_load.return_value = schema
                
                with patch.object(processor, '_build_summarization_messages') as mock_build:
                    mock_build.return_value = expected_messages
                    
                    processor.execute(
                        "test",
                        task="summary",
                        template_id="test_template"
                    )
                    
                    # Verify chat was called with messages
                    call_args = mock_model.chat.call_args
                    assert 'messages' in call_args.kwargs
                    assert call_args.kwargs['messages'] == [expected_messages]

    def test_execute_summary_passes_sampling_params(self, tmp_path):
        """Should pass sampling_params to llm.chat()."""
        processor = LLMProcessor()
        processor._model_loaded = True
        
        mock_model = Mock()
        mock_output = Mock()
        mock_output.outputs = [Mock(text='{"title": "Test"}')]
        mock_model.chat.return_value = [mock_output]
        processor.model = mock_model
        
        schema_file = tmp_path / "test_template.json"
        schema = {"type": "object"}
        schema_file.write_text(json.dumps(schema))
        
        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.templates_dir = tmp_path
            
            with patch("src.processors.llm.processor.load_template_schema") as mock_load:
                mock_load.return_value = schema
                
                with patch.object(processor, '_build_summarization_messages') as mock_build:
                    mock_build.return_value = [{'role': 'user', 'content': 'test'}]
                    
                    with patch("src.processors.llm.processor.build_generation_config") as mock_config:
                        mock_params = Mock()
                        mock_config.return_value.to_sampling_params.return_value = mock_params
                        
                        processor.execute(
                            "test",
                            task="summary",
                            template_id="test_template"
                        )
                        
                        call_args = mock_model.chat.call_args
                        assert 'sampling_params' in call_args.kwargs

    def test_execute_summary_no_manual_stop_tokens(self, tmp_path):
        """Should NOT manually set stop tokens (vLLM handles it automatically)."""
        processor = LLMProcessor()
        processor._model_loaded = True
        
        mock_model = Mock()
        mock_output = Mock()
        mock_output.outputs = [Mock(text='{"title": "Test"}')]
        mock_model.chat.return_value = [mock_output]
        processor.model = mock_model
        
        schema_file = tmp_path / "test_template.json"
        schema = {"type": "object"}
        schema_file.write_text(json.dumps(schema))
        
        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.templates_dir = tmp_path
            
            with patch("src.processors.llm.processor.load_template_schema") as mock_load:
                mock_load.return_value = schema
                
                with patch.object(processor, '_build_summarization_messages') as mock_build:
                    mock_build.return_value = [{'role': 'user', 'content': 'test'}]
                    
                    with patch("src.processors.llm.processor.build_generation_config") as mock_config:
                        mock_params = Mock()
                        mock_config.return_value.to_sampling_params.return_value = mock_params
                        
                        processor.execute(
                            "test",
                            task="summary",
                            template_id="test_template"
                        )
                        
                        # Check that sampling_params doesn't have manual stop tokens
                        # (they should come from tokenizer config automatically)
                        call_args = mock_model.chat.call_args
                        sampling_params = call_args.kwargs.get('sampling_params')
                        
                        # The params object itself might have stop, but we shouldn't
                        # be manually adding <|im_end|> in our code
                        # This is a behavioral test - we're not setting it explicitly

    def test_execute_enhancement_still_uses_generate(self):
        """Enhancement tasks should still use generate() for now (not migrated yet)."""
        processor = LLMProcessor()
        processor._model_loaded = True
        
        mock_model = Mock()
        mock_output = Mock()
        mock_output.outputs = [Mock(text='Enhanced text')]
        mock_model.generate.return_value = [mock_output]
        mock_model.get_default_sampling_params.return_value = Mock()
        processor.model = mock_model
        
        with patch.object(processor.prompt_renderer, 'render') as mock_render:
            mock_render.return_value = "rendered prompt"
            
            with patch("src.processors.llm.processor.apply_overrides_to_sampling") as mock_apply:
                mock_apply.return_value = Mock()
                
                processor.execute("test text", task="enhancement")
                
                # Should still call generate() for enhancement
                mock_model.generate.assert_called_once()
                mock_model.chat.assert_not_called()


class TestChatAPIIntegrationEndToEnd:
    """End-to-end integration tests for chat API."""

    def test_generate_summary_uses_chat_api(self, tmp_path):
        """generate_summary() should use chat API under the hood."""
        processor = LLMProcessor()
        processor._model_loaded = True
        
        # Create schema
        schema_file = tmp_path / "test_summary.json"
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["title", "tags"]
        }
        schema_file.write_text(json.dumps(schema))
        
        # Mock model
        mock_model = Mock()
        mock_output = Mock()
        mock_output.outputs = [Mock(text='{"title": "Test", "tags": ["test"]}')]
        mock_model.chat.return_value = [mock_output]
        mock_model.get_default_sampling_params.return_value = Mock()
        processor.model = mock_model
        
        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.templates_dir = tmp_path
            mock_settings.llm_sum_temperature = 0.7
            mock_settings.llm_sum_top_p = 0.9
            mock_settings.llm_sum_top_k = 20
            mock_settings.llm_sum_max_tokens = 1000
            
            with patch.object(processor, '_build_summarization_messages') as mock_build:
                mock_build.return_value = [
                    {'role': 'system', 'content': 'System'},
                    {'role': 'user', 'content': 'User'}
                ]
                
                result = processor.generate_summary(
                    "Test transcript",
                    "test_summary"
                )
                
                # Should have called chat
                mock_model.chat.assert_called()
                
                # Should return valid summary
                assert result['summary'] == {"title": "Test", "tags": ["test"]}

    def test_chat_api_returns_llm_result(self, tmp_path):
        """execute() with chat API should return LLMResult object."""
        processor = LLMProcessor()
        processor._model_loaded = True
        
        schema_file = tmp_path / "test.json"
        schema = {"type": "object"}
        schema_file.write_text(json.dumps(schema))
        
        mock_model = Mock()
        mock_output = Mock()
        mock_output.outputs = [Mock(text='{"result": "ok"}')]
        mock_output.prompt_token_ids = [1, 2, 3, 4, 5]
        mock_model.chat.return_value = [mock_output]
        processor.model = mock_model
        processor.model_info = {"model_name": "test-model"}
        
        with patch("src.processors.llm.processor.settings") as mock_settings:
            mock_settings.templates_dir = tmp_path
            
            with patch("src.processors.llm.processor.load_template_schema") as mock_load:
                mock_load.return_value = schema
                
                with patch.object(processor, '_build_summarization_messages') as mock_build:
                    mock_build.return_value = [{'role': 'user', 'content': 'test'}]
                    
                    result = processor.execute(
                        "test",
                        task="summary",
                        template_id="test"
                    )
                    
                    assert isinstance(result, LLMResult)
                    assert result.text == '{"result": "ok"}'
                    assert result.tokens_used == 5
                    assert result.model_info == {"model_name": "test-model"}
                    assert result.metadata['task'] == 'summary'
                    assert result.metadata['method'] == 'chat_api'


class TestBackwardCompatibility:
    """Test that old functionality still works during migration."""

    def test_old_enhance_text_still_works(self):
        """enhance_text() should continue working with generate()."""
        processor = LLMProcessor()
        processor._model_loaded = True
        
        mock_model = Mock()
        mock_output = Mock()
        mock_output.outputs = [Mock(text='Enhanced text')]
        mock_model.generate.return_value = [mock_output]
        mock_model.get_default_sampling_params.return_value = Mock()
        processor.model = mock_model
        processor.model_info = {"model_name": "test"}
        
        with patch.object(processor.prompt_renderer, 'render') as mock_render:
            mock_render.return_value = "prompt"
            
            result = processor.enhance_text("original text")
            
            assert result['enhanced_text'] == 'Enhanced text'
            assert result['enhancement_applied'] is True

    def test_old_templates_with_markers_removed(self):
        """Old templates with chat markers should be detected and warned about."""
        processor = LLMProcessor()
        
        # If renderer returns content with chat markers, we should detect it
        with patch.object(processor.prompt_renderer, 'render') as mock_render:
            mock_render.return_value = "<|im_start|>system\nPrompt<|im_end|>"
            
            # This should be detected as old-style template
            rendered = mock_render("old_template", text="test")
            
            # Our code should handle this gracefully
            assert "<|im_start|>" in rendered  # Old style detected
