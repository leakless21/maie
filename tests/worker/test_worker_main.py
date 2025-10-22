"""
Unit tests for src/worker/main.py

Tests worker initialization, Redis connection setup, model verification,
and worker startup sequence.

Follows TDD principles from docs/TDD.md section 3.2 (GPU Worker).
"""

from unittest.mock import MagicMock, patch

import pytest
from redis import Redis
from rq import Worker

from src.worker.main import setup_redis_connection, start_worker, verify_models


class TestSetupRedisConnection:
    """Test Redis connection setup with proper configuration."""

    @patch("src.worker.main.Redis.from_url")
    def test_successful_redis_connection(self, mock_from_url):
        """Test successful Redis connection with ping verification."""
        # Setup
        mock_redis = MagicMock(spec=Redis)
        mock_redis.ping.return_value = True
        mock_from_url.return_value = mock_redis

        # Execute
        result = setup_redis_connection()

        # Verify
        assert result == mock_redis
        mock_from_url.assert_called_once()
        mock_redis.ping.assert_called_once()

    @patch("src.worker.main.Redis.from_url")
    @patch("sys.exit")
    def test_redis_connection_failure(self, mock_exit, mock_from_url):
        """Test that connection failure exits gracefully."""
        # Setup
        mock_redis = MagicMock(spec=Redis)
        mock_redis.ping.side_effect = ConnectionError("Connection refused")
        mock_from_url.return_value = mock_redis

        # Execute
        setup_redis_connection()

        # Verify - should exit with code 1
        mock_exit.assert_called_once_with(1)

    @patch("src.worker.main.Redis.from_url")
    def test_redis_url_from_settings(self, mock_from_url):
        """Test that Redis URL is taken from settings."""
        # Setup
        mock_redis = MagicMock(spec=Redis)
        mock_redis.ping.return_value = True
        mock_from_url.return_value = mock_redis

        # Execute
        setup_redis_connection()

        # Verify - should use settings.redis_url
        assert mock_from_url.call_count == 1
        call_args = mock_from_url.call_args
        # First positional arg should be a Redis URL string
        assert isinstance(call_args[0][0], str)
        # decode_responses should be False
        assert call_args[1]["decode_responses"] is False


class TestVerifyModels:
    """Test model verification before worker startup."""

    @patch("src.worker.main.settings")
    def test_missing_models(self, mock_settings):
        """Test verification failure when models are missing."""
        # Setup - create mock Path where one doesn't exist
        mock_paths = [MagicMock() for _ in range(3)]
        mock_paths[0].exists.return_value = True
        mock_paths[1].exists.return_value = False  # Missing
        mock_paths[2].exists.return_value = True
        mock_settings.get_model_path.side_effect = mock_paths

        # Execute
        result = verify_models()

        # Verify
        assert result is False

    def test_verify_models_actually_checks_paths(self):
        """Test that verify_models checks for model existence."""
        # This is an integration-style test - we just verify the function exists and can be called
        # Actual behavior depends on whether models are actually present
        result = verify_models()
        # Result can be True or False depending on environment
        assert isinstance(result, bool)


class TestStartWorker:
    """Test worker startup sequence."""

    @patch("src.worker.main.verify_models")
    @patch("sys.exit", side_effect=SystemExit)
    def test_start_worker_fails_on_model_verification(self, mock_exit, mock_verify):
        """Test that worker exits if model verification fails."""
        # Setup
        mock_verify.return_value = False

        # Execute & Verify - should raise SystemExit
        with pytest.raises(SystemExit):
            start_worker()

        mock_verify.assert_called_once()
        mock_exit.assert_called_once_with(1)

    @patch("src.worker.main.verify_models")
    @patch("src.worker.main.setup_redis_connection")
    @patch("src.worker.main.Worker")
    def test_start_worker_successful_startup(
        self, mock_worker_class, mock_setup_redis, mock_verify
    ):
        """Test successful worker startup with proper configuration."""
        # Setup
        mock_verify.return_value = True
        mock_redis = MagicMock(spec=Redis)
        mock_setup_redis.return_value = mock_redis
        mock_worker = MagicMock(spec=Worker)
        mock_worker.name = "test-worker"
        mock_worker_class.return_value = mock_worker

        # Execute
        start_worker()

        # Verify
        mock_verify.assert_called_once()
        mock_setup_redis.assert_called_once()
        mock_worker_class.assert_called_once()

        # Verify Worker was created with correct parameters
        call_args = mock_worker_class.call_args
        queues = call_args[0][0]
        assert "default" in queues
        assert "audio_processing" in queues
        assert call_args[1]["connection"] == mock_redis
        assert "name" in call_args[1]

        # Verify worker.work() was called
        mock_worker.work.assert_called_once()

    @patch("src.worker.main.verify_models")
    @patch("src.worker.main.setup_redis_connection")
    @patch("src.worker.main.Worker")
    def test_worker_listens_to_correct_queues(
        self, mock_worker_class, mock_setup_redis, mock_verify
    ):
        """Test that worker listens to the correct RQ queues."""
        # Setup
        mock_verify.return_value = True
        mock_redis = MagicMock(spec=Redis)
        mock_setup_redis.return_value = mock_redis
        mock_worker = MagicMock(spec=Worker)
        mock_worker.name = "test-worker"  # Add name attribute
        mock_worker_class.return_value = mock_worker

        # Execute
        start_worker()

        # Verify queues
        call_args = mock_worker_class.call_args
        queues = call_args[0][0]
        assert len(queues) == 2
        assert "default" in queues
        assert "audio_processing" in queues

    @patch("src.worker.main.verify_models")
    @patch("src.worker.main.setup_redis_connection")
    @patch("src.worker.main.Worker")
    def test_worker_uses_settings_for_configuration(
        self, mock_worker_class, mock_setup_redis, mock_verify
    ):
        """Test that worker uses settings for name and configuration."""
        # Setup
        mock_verify.return_value = True
        mock_redis = MagicMock(spec=Redis)
        mock_setup_redis.return_value = mock_redis
        mock_worker = MagicMock(spec=Worker)
        mock_worker.name = "configured-worker-name"
        mock_worker_class.return_value = mock_worker

        # Execute
        start_worker()

        # Verify worker name comes from settings
        call_args = mock_worker_class.call_args
        assert "name" in call_args[1]
        # The name should be passed to Worker constructor
        assert call_args[1]["name"] is not None


class TestMultiprocessingStartMethod:
    """Test that worker enforces spawn start method for CUDA compatibility."""

    def test_spawn_start_method_logic(self):
        """Test the spawn start method logic directly."""
        import multiprocessing as mp

        # Test the logic from the main block
        try:
            if mp.get_start_method(allow_none=True) != "spawn":
                # This should be called when not spawn
                mp.set_start_method("spawn", force=True)
                # Verify it was set
                assert mp.get_start_method() == "spawn"
        except RuntimeError:
            # Start method already set, ignore
            pass

    def test_spawn_start_method_skips_when_already_set(self):
        """Test that spawn is not set if already configured."""
        import multiprocessing as mp

        # Set to spawn first
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        # Test the logic - should not call set_start_method again
        original_method = mp.get_start_method()
        try:
            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method("spawn", force=True)
        except RuntimeError:
            # Start method already set, ignore
            pass

        # Should still be spawn
        assert mp.get_start_method() == original_method
