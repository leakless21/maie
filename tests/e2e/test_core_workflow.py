import pytest
import json
from pathlib import Path

class TestCoreWorkflow:
    """Core E2E workflow tests"""

    def test_happy_path_whisper(self, api_client, test_assets_dir):
        """Test complete workflow with Whisper backend"""
        audio_file = test_assets_dir / "sample_30s.wav"
        if not audio_file.exists():
            pytest.skip(f"Test audio file not found: {audio_file}")

        # Submit job
        response = api_client.submit_audio(
            audio_file,
            features=["clean_transcript", "summary"],
            template_id="meeting_notes_v1"
        )
        assert response.status_code == 202
        task_id = response.json()['task_id']

        # Wait for completion
        result = api_client.wait_for_completion(task_id, timeout=180)

        # Validate result structure
        self._validate_complete_result(result)

    def test_happy_path_chunkformer(self, api_client, test_assets_dir):
        """Test complete workflow with ChunkFormer backend"""
        audio_file = test_assets_dir / "sample_30s.wav"
        if not audio_file.exists():
            pytest.skip(f"Test audio file not found: {audio_file}")

        response = api_client.submit_audio(
            audio_file,
            features=["clean_transcript", "summary"],
            template_id="meeting_notes_v1",
            asr_backend="chunkformer"
        )
        assert response.status_code == 202
        task_id = response.json()['task_id']

        result = api_client.wait_for_completion(task_id, timeout=120)
        self._validate_complete_result(result)

    def test_feature_combinations(self, api_client, test_assets_dir):
        """Test different feature combinations"""
        audio_file = test_assets_dir / "sample_30s.wav"
        if not audio_file.exists():
            pytest.skip(f"Test audio file not found: {audio_file}")

        test_cases = [
            ["raw_transcript"],
            ["clean_transcript"],
            ["summary"],
            ["raw_transcript", "clean_transcript", "summary"]
        ]

        for features in test_cases:
            response = api_client.submit_audio(
                audio_file,
                features=features,
                template_id="meeting_notes_v1" if "summary" in features else None
            )
            assert response.status_code == 202
            task_id = response.json()['task_id']

            result = api_client.wait_for_completion(task_id)
            self._validate_features_present(result, features)

    def _validate_complete_result(self, result):
        """Validate complete processing result"""
        assert result['status'] == 'COMPLETE'
        assert 'task_id' in result
        assert 'versions' in result
        assert 'metrics' in result
        assert 'results' in result

        # Validate versions
        versions = result['versions']
        assert 'pipeline_version' in versions
        assert 'asr_backend' in versions
        assert 'summarization_llm' in versions

        # Validate metrics
        metrics = result['metrics']
        required_metrics = ['input_duration_seconds', 'processing_time_seconds',
                          'rtf', 'vad_coverage', 'asr_confidence_avg', 'edit_rate_cleaning']
        for metric in required_metrics:
            assert metric in metrics

        # Validate results
        results = result['results']
        assert 'raw_transcript' in results
        assert 'clean_transcript' in results
        assert 'summary' in results

        # Validate summary structure
        summary = results['summary']
        assert 'title' in summary
        assert 'abstract' in summary
        assert 'main_points' in summary
        assert 'tags' in summary
        assert isinstance(summary['tags'], list)

    def _validate_features_present(self, result, requested_features):
        """Validate that requested features are present in results"""
        results = result['results']
        feature_mapping = {
            'raw_transcript': 'raw_transcript',
            'clean_transcript': 'clean_transcript',
            'summary': 'summary'
        }

        for feature in requested_features:
            if feature in feature_mapping:
                assert feature_mapping[feature] in results