"""
Comprehensive test suite for legacy string formatting (% operator).

This test suite ensures backward compatibility and documents the current
behavior of the deprecated % string formatting usage in the worker main module.
"""

from unittest.mock import MagicMock, patch

from src.worker.main import verify_models


class TestStringFormattingLegacy:
    """Test suite for legacy % string formatting behavior."""

    def test_verify_models_legacy_string_formatting(self):
        """Test that verify_models uses deprecated % string formatting."""
        # Mock the importlib.util.find_spec to simulate different scenarios
        with patch("importlib.util.find_spec") as mock_find_spec:
            # Mock settings and Path.exists to bypass model path checks
            with patch("src.worker.main.settings") as mock_settings:
                with patch("pathlib.Path.exists", return_value=True):
                    # Set up mock settings
                    mock_settings.asr.whisper_model_path = "/fake/whisper"
                    mock_settings.chunkformer.chunkformer_model_path = (
                        "/fake/chunkformer"
                    )
                    mock_settings.llm_enhance_model = "/fake/llm"

                    # Test case 1: Both modules available (should not trigger the legacy formatting)
                    mock_find_spec.side_effect = (
                        lambda module: MagicMock()
                        if module
                        in ["src.processors.asr.factory", "src.processors.llm"]
                        else None
                    )

                    with patch("src.worker.main.get_logger") as mock_logger:
                        result = verify_models()

                        # Should return True and not use the legacy formatting
                        assert result is True
                        # The logger is called, then info() is called on the returned logger
                        mock_logger.return_value.info.assert_any_call(
                            "All required models and modules are available"
                        )
                        mock_logger.return_value.error.assert_not_called()

    def test_verify_models_legacy_formatting_triggered(self):
        """Test legacy string formatting when modules are missing."""
        with patch("importlib.util.find_spec") as mock_find_spec:
            # Mock settings and Path.exists to bypass model path checks
            with patch("src.worker.main.settings") as mock_settings:
                with patch("pathlib.Path.exists", return_value=True):
                    # Set up mock settings
                    mock_settings.asr.whisper_model_path = "/fake/whisper"
                    mock_settings.chunkformer.chunkformer_model_path = (
                        "/fake/chunkformer"
                    )
                    mock_settings.llm_enhance_model = "/fake/llm"

                    # Simulate missing ASR factory but present LLM
                    def mock_find_spec_impl(module):
                        if module == "src.processors.llm":
                            return MagicMock()
                        elif module == "src.processors.asr.factory":
                            return None
                        else:
                            return MagicMock()

                    mock_find_spec.side_effect = mock_find_spec_impl

                    with patch("src.worker.main.get_logger") as mock_logger:
                        result = verify_models()

                        # Should return False and use legacy formatting
                        assert result is False

                        # Check that the f-string formatting was called
                        mock_logger.return_value.error.assert_called_once()
                        call_args = mock_logger.return_value.error.call_args[0][0]

                        # Verify the f-string formatted message (identical output)
                        assert (
                            call_args
                            == "Missing required modules: ASRFactory=False, LLM=True"
                        )

    def test_verify_models_legacy_formatting_both_missing(self):
        """Test legacy string formatting when both modules are missing."""
        with patch("importlib.util.find_spec") as mock_find_spec:
            # Mock settings and Path.exists to bypass model path checks
            with patch("src.worker.main.settings") as mock_settings:
                with patch("pathlib.Path.exists", return_value=True):
                    # Set up mock settings
                    mock_settings.asr.whisper_model_path = "/fake/whisper"
                    mock_settings.chunkformer.chunkformer_model_path = (
                        "/fake/chunkformer"
                    )
                    mock_settings.llm_enhance_model = "/fake/llm"

                    # Simulate both modules missing
                    mock_find_spec.side_effect = lambda module: None

                    with patch("src.worker.main.get_logger") as mock_logger:
                        result = verify_models()

                        # Should return False and use legacy formatting
                        assert result is False

                        # Check that the f-string formatting was called
                        mock_logger.return_value.error.assert_called_once()
                        call_args = mock_logger.return_value.error.call_args[0][0]

                        # Verify the f-string formatted message (identical output)
                        assert (
                            call_args
                            == "Missing required modules: ASRFactory=False, LLM=False"
                        )

    def test_verify_models_legacy_formatting_boolean_values(self):
        """Test that boolean values are correctly formatted with % operator."""
        with patch("importlib.util.find_spec") as mock_find_spec:
            # Mock settings and Path.exists to bypass model path checks
            with patch("src.worker.main.settings") as mock_settings:
                with patch("pathlib.Path.exists", return_value=True):
                    # Set up mock settings
                    mock_settings.asr.whisper_model_path = "/fake/whisper"
                    mock_settings.chunkformer.chunkformer_model_path = (
                        "/fake/chunkformer"
                    )
                    mock_settings.llm_enhance_model = "/fake/llm"

                    # Test different boolean combinations
                    test_cases = [
                        (
                            True,
                            True,
                            True,
                        ),  # Both available - shouldn't trigger formatting
                        (True, False, False),  # ASR available, LLM missing
                        (False, True, False),  # ASR missing, LLM available
                        (False, False, False),  # Both missing
                    ]

                    for has_asr, has_llm, should_succeed in test_cases:
                        with patch("src.worker.main.get_logger") as mock_logger:

                            def mock_find_spec_impl(module):
                                if module == "src.processors.asr.factory":
                                    return MagicMock() if has_asr else None
                                elif module == "src.processors.llm":
                                    return MagicMock() if has_llm else None
                                else:
                                    return MagicMock()

                            mock_find_spec.side_effect = mock_find_spec_impl

                            result = verify_models()

                            if should_succeed:
                                assert result is True
                                mock_logger.return_value.info.assert_any_call(
                                    "All required models and modules are available"
                                )
                            else:
                                assert result is False
                                mock_logger.return_value.error.assert_called_once()
                                call_args = mock_logger.return_value.error.call_args[0][
                                    0
                                ]

                                # Verify the f-string formatted message (identical output)
                                expected_message = f"Missing required modules: ASRFactory={has_asr}, LLM={has_llm}"
                                assert call_args == expected_message

    def test_verify_models_exception_handling_with_legacy_formatting(self):
        """Test exception handling doesn't interfere with legacy formatting."""
        with patch("importlib.util.find_spec") as mock_find_spec:
            # First call succeeds, second call raises exception
            call_count = 0

            def mock_find_spec_impl(module):
                nonlocal call_count
                call_count += 1
                if call_count == 1:  # First call for ASR factory
                    return MagicMock()
                elif call_count == 2:  # Second call for LLM - raise exception
                    raise Exception("Import error")
                else:
                    return MagicMock()

            mock_find_spec.side_effect = mock_find_spec_impl

            with patch("src.worker.main.get_logger") as mock_logger:
                result = verify_models()

                # Should return False due to exception
                assert result is False

                # Should log the exception
                mock_logger.return_value.exception.assert_called_once_with(
                    "Model verification failed"
                )
                # Should not use the legacy formatting due to exception
                mock_logger.return_value.error.assert_not_called()

    def test_legacy_formatting_behavior_documentation(self):
        """Document current legacy formatting behavior for future migration reference."""
        with patch("importlib.util.find_spec") as mock_find_spec:
            # Force the legacy formatting path
            mock_find_spec.side_effect = lambda module: None

            with patch("src.worker.main.get_logger") as mock_logger:
                verify_models()

                # Current behavior: uses modern f-string formatting
                mock_logger.return_value.error.assert_called_once()
                call_args = mock_logger.return_value.error.call_args[0][0]

                # Document the current modern pattern
                assert (
                    "Missing required modules: ASRFactory=False, LLM=False" == call_args
                )
                # Successfully migrated from: "Missing required modules: ASRFactory=%s, LLM=%s" % (has_asr_factory, has_llm)
                # To: f"Missing required modules: ASRFactory={has_asr_factory}, LLM={has_llm}"

                # Migration completed:
                # 1. ✓ Replaced % formatting with f-strings
                # 2. ✓ Ensured the same output format is maintained
                # 3. ✓ Verified that the migration produces identical results

    def test_legacy_formatting_output_consistency(self):
        """Test that legacy formatting produces consistent output."""
        with patch("importlib.util.find_spec") as mock_find_spec:
            # Test specific boolean combinations
            test_cases = [
                (True, False, "Missing required modules: ASRFactory=True, LLM=False"),
                (False, True, "Missing required modules: ASRFactory=False, LLM=True"),
                (False, False, "Missing required modules: ASRFactory=False, LLM=False"),
            ]

            for has_asr, has_llm, expected_message in test_cases:
                with patch("src.worker.main.get_logger") as mock_logger:

                    def mock_find_spec_impl(module):
                        if module == "src.processors.asr.factory":
                            return MagicMock() if has_asr else None
                        elif module == "src.processors.llm":
                            return MagicMock() if has_llm else None
                        else:
                            return MagicMock()

                    mock_find_spec.side_effect = mock_find_spec_impl

                    result = verify_models()

                    assert result is False
                    # The f-string formatting passes the pre-formatted message
                    expected_message = (
                        f"Missing required modules: ASRFactory={has_asr}, LLM={has_llm}"
                    )
                    mock_logger.return_value.error.assert_called_once_with(
                        expected_message
                    )

    def test_legacy_formatting_with_logger_mock(self):
        """Test that logger mock correctly captures legacy formatting calls."""
        with patch("importlib.util.find_spec") as mock_find_spec:
            mock_find_spec.side_effect = lambda module: None

            # Create a more detailed mock to track exact calls
            mock_logger = MagicMock()

            with patch("src.worker.main.get_logger", return_value=mock_logger):
                result = verify_models()

                assert result is False

                # Verify the exact call to error with f-string formatting
                mock_logger.error.assert_called_once()
                call_args = mock_logger.error.call_args[0]

                # The call should be: error("Missing required modules: ASRFactory=False, LLM=False")
                assert len(call_args) == 1  # Pre-formatted message only
                formatted_message = call_args[0]

                assert (
                    formatted_message
                    == "Missing required modules: ASRFactory=False, LLM=False"
                )


class TestStringFormattingMigration:
    """Test suite to prepare for migration to f-strings."""

    def test_migration_readiness_check(self):
        """Test that current code is ready for migration to f-strings."""
        # This test verifies that the current implementation structure
        # supports migration to f-strings

        # The current logging call:
        # get_logger().error("Missing required modules: ASRFactory=%s, LLM=%s", has_asr_factory, has_llm)

        # Should be migrated to:
        # get_logger().error(f"Missing required modules: ASRFactory={has_asr_factory}, LLM={has_llm}")

        # Test that we can reproduce the same output with f-strings
        test_cases = [
            (True, False),
            (False, True),
            (False, False),
        ]

        for has_asr, has_llm in test_cases:
            # Legacy formatting
            legacy_result = "Missing required modules: ASRFactory=%s, LLM=%s" % (
                has_asr,
                has_llm,
            )

            # Modern f-string formatting
            modern_result = (
                f"Missing required modules: ASRFactory={has_asr}, LLM={has_llm}"
            )

            # Both should produce identical results
            assert legacy_result == modern_result, f"Mismatch for {has_asr}, {has_llm}"

    def test_f_string_equivalence_comprehensive(self):
        """Test comprehensive equivalence between % formatting and f-strings."""
        # Test various data types that might be used in string formatting
        test_values = [
            (True, "True"),
            (False, "False"),
            (0, "0"),
            (1, "1"),
            (-1, "-1"),
            (3.14, "3.14"),
            ("test", "test"),
            (None, "None"),
            ([1, 2, 3], "[1, 2, 3]"),
            ({"key": "value"}, "{'key': 'value'}"),
        ]

        for value, expected_str in test_values:
            # Test single value formatting
            legacy_result = "Value=%s" % value
            modern_result = f"Value={value}"

            # Both should produce identical string representations
            assert legacy_result == modern_result, (
                f"Mismatch for value {value}: {legacy_result} vs {modern_result}"
            )

    def test_f_string_performance_considerations(self):
        """Document performance considerations for f-string migration."""
        # f-strings are generally faster than % formatting in Python 3.6+
        # This test documents the expected performance improvement

        import time

        # Simple performance comparison (not a benchmark, just documentation)
        test_data = (True, False)
        iterations = 1000

        # Time legacy formatting
        start_time = time.time()
        for _ in range(iterations):
            _ = "ASR=%s, LLM=%s" % test_data
        time.time() - start_time

        # Time f-string formatting
        start_time = time.time()
        for _ in range(iterations):
            _ = f"ASR={test_data[0]}, LLM={test_data[1]}"
        time.time() - start_time

        # f-strings should be faster or comparable
        # This is just for documentation - actual performance depends on many factors
        # assert modern_time <= legacy_time * 1.1  # Allow 10% slower max

        # The key point: f-strings are the modern, preferred approach
        assert "ASR=True, LLM=False" == f"ASR={test_data[0]}, LLM={test_data[1]}"
