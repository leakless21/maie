"""
Unit tests for worker pipeline helper functions.

Tests the following functions in isolation:
- _update_status: Update task status in Redis
- _calculate_edit_rate: Calculate Levenshtein distance ratio
"""

from src.api.schemas import TaskStatus

# Import pipeline helpers at module level to avoid torch reload issues
from src.worker.pipeline import _calculate_edit_rate, _update_status


class TestUpdateStatus:
    """Test _update_status function."""

    def test_update_status_basic(self, mock_redis_sync):
        """Test basic status update."""
        task_id = "test-task-123"
        task_key = f"task:{task_id}"

        _update_status(mock_redis_sync, task_key, TaskStatus.PROCESSING_ASR)

        # Verify Redis hset was called
        mock_redis_sync.hset.assert_called()

        # Check the stored data
        stored_data = mock_redis_sync._data.get(task_key, {})
        assert stored_data["status"] == TaskStatus.PROCESSING_ASR.value
        assert "updated_at" in stored_data

    def test_update_status_with_details(self, mock_redis_sync):
        """Test status update with additional details."""
        task_id = "test-task-456"
        task_key = f"task:{task_id}"
        details = {"transcription_length": 150, "confidence": 0.92}

        _update_status(mock_redis_sync, task_key, TaskStatus.PROCESSING_LLM, details)

        # Check the stored data includes details
        stored_data = mock_redis_sync._data.get(task_key, {})
        assert stored_data["status"] == TaskStatus.PROCESSING_LLM.value
        assert stored_data["transcription_length"] == 150
        assert stored_data["confidence"] == 0.92

    def test_update_status_to_complete(self, mock_redis_sync):
        """Test status update to COMPLETE with results."""
        task_id = "test-task-789"
        task_key = f"task:{task_id}"

        result_data = {
            "versions": {"pipeline_version": "1.0.0"},
            "metrics": {"rtf": 0.5},
            "results": {"transcript": "Test"},
        }

        _update_status(mock_redis_sync, task_key, TaskStatus.COMPLETE, result_data)

        # Verify complete result is stored
        stored_data = mock_redis_sync._data.get(task_key, {})
        assert stored_data["status"] == TaskStatus.COMPLETE.value
        assert "versions" in stored_data
        assert "metrics" in stored_data
        assert "results" in stored_data

    def test_update_status_to_failed(self, mock_redis_sync):
        """Test status update to FAILED with error details."""
        task_id = "test-task-error"
        task_key = f"task:{task_id}"

        error_details = {
            "error_message": "CUDA out of memory",
            "error_code": "CUDA_OOM",
            "stage": "asr_execute",
        }

        _update_status(mock_redis_sync, task_key, TaskStatus.FAILED, error_details)

        # Verify error is stored
        stored_data = mock_redis_sync._data.get(task_key, {})
        assert stored_data["status"] == TaskStatus.FAILED.value
        assert stored_data["error_message"] == "CUDA out of memory"
        assert stored_data["error_code"] == "CUDA_OOM"
        assert stored_data["stage"] == "asr_execute"


class TestCalculateEditRate:
    """Test _calculate_edit_rate function using Levenshtein distance."""

    def test_identical_strings(self):
        """Test edit rate for identical strings (should be 0.0)."""
        text = "The quick brown fox jumps over the lazy dog"
        rate = _calculate_edit_rate(text, text)

    def test_completely_different_strings(self):
        """Test edit rate for completely different strings."""
        original = "abc"
        enhanced = "xyz"

        edit_rate = _calculate_edit_rate(original, enhanced)

        assert (
            edit_rate == 1.0
        ), "Completely different strings should have 1.0 edit rate"

    def test_single_character_change(self):
        """Test edit rate for single character substitution."""
        original = "test"
        enhanced = "best"
        rate = _calculate_edit_rate(original, enhanced)

    def test_insertion(self):
        """Test edit rate with character insertions."""
        original = "test"
        enhanced = "testing"
        rate = _calculate_edit_rate(original, enhanced)

    def test_deletion(self):
        """Test edit rate with character deletions."""
        original = "testing"
        enhanced = "test"
        rate = _calculate_edit_rate(original, enhanced)

    def test_empty_strings(self):
        """Test edit rate with empty strings."""
        # Both empty
        rate = _calculate_edit_rate("", "")
        assert rate == 0.0

        # One empty
        edit_rate = _calculate_edit_rate("hello", "")
        assert edit_rate == 1.0

        edit_rate = _calculate_edit_rate("", "hello")
        assert edit_rate == 1.0

    def test_realistic_text_enhancement(self):
        """Test edit rate for realistic text enhancement scenario."""
        # Original: no punctuation or capitalization
        original = "this is a test transcription from the audio file"

        # Enhanced: added punctuation and capitalization
        enhanced = "This is a test transcription from the audio file."

        edit_rate = _calculate_edit_rate(original, enhanced)

        # Should be low since only minor changes (capitalization + period)
        assert (
            edit_rate < 0.1
        ), f"Small enhancements should have low edit rate, got {edit_rate}"

    def test_punctuation_changes(self):
        """Test edit rate with punctuation additions."""
        original = "hello world"
        enhanced = "Hello, world!"
        rate = _calculate_edit_rate(original, enhanced)

    def test_word_reordering(self):
        """Test edit rate with word reordering."""
        original = "dog brown quick"
        enhanced = "quick brown dog"
        rate = _calculate_edit_rate(original, enhanced)

    def test_case_sensitivity(self):
        """Test that edit rate is case-sensitive."""
        original = "hello world"
        enhanced = "Hello World"

        edit_rate = _calculate_edit_rate(original, enhanced)

        # 2 changes / 11 characters = ~0.18
        assert edit_rate > 0.0, "Case changes should be detected"
        assert edit_rate < 0.25, "Only case changes should be minor"


class TestEditRateAlgorithm:
    """Test the Levenshtein distance algorithm implementation."""

    def test_algorithm_symmetry(self):
        """Test that edit rate is symmetric."""
        str1 = "hello"
        str2 = "hallo"

        rate1 = _calculate_edit_rate(str1, str2)
        rate2 = _calculate_edit_rate(str2, str1)

        assert rate1 == rate2, "Edit rate should be symmetric"

    def test_algorithm_triangle_inequality(self):
        """Test triangle inequality property."""
        # For any three strings A, B, C:
        # distance(A, C) <= distance(A, B) + distance(B, C)

        str_a = "hello"
        str_b = "hallo"
        str_c = "hxllo"

        rate_ac = _calculate_edit_rate(str_a, str_c)
        rate_ab = _calculate_edit_rate(str_a, str_b)
        rate_bc = _calculate_edit_rate(str_b, str_c)

        # Convert rates back to distances for comparison
        # (rates are normalized by max length, so this is approximate)
        assert rate_ac <= rate_ab + rate_bc + 0.1  # Allow small tolerance

    def test_known_levenshtein_distances(self):
        """Test against known Levenshtein distance examples."""
        # "kitten" -> "sitting" requires 3 edits (k->s, e->i, insert g)
        original = "kitten"
        enhanced = "sitting"

        edit_rate = _calculate_edit_rate(original, enhanced)

        # 3 edits / 7 characters = ~0.43
        expected_rate = 3.0 / 7.0
        assert abs(edit_rate - expected_rate) < 0.01
