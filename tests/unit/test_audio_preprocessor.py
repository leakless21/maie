import json
import os
from pathlib import Path
import subprocess
import wave
import struct
import math
import pytest
from unittest.mock import MagicMock

# Import module under test
from src.processors.audio import preprocessor as preprocessor_module

@pytest.fixture(autouse=True)
def deterministic_settings(monkeypatch):
    """Set predictable settings for audio preprocessor during tests"""
    monkeypatch.setattr(preprocessor_module, 'MIN_DURATION_SEC', 1.0, raising=False)
    yield

def make_sine_wav(path: Path, duration_s: float = 1.0, sample_rate: int = 16000):
    """Helper to create a mono 16kHz PCM S16LE WAV file with a sine wave"""
    n_channels = 1
    sampwidth = 2
    n_frames = int(sample_rate * duration_s)
    frequency = 440.0
    amplitude = 32767
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        for i in range(n_frames):
            value = int(amplitude * math.sin(2.0 * math.pi * frequency * i / sample_rate))
            wf.writeframes(struct.pack('<h', value))
    return path

def fake_subprocess_result(returncode=0, stdout='', stderr=''):
    mock = MagicMock()
    mock.returncode = returncode
    mock.stdout = stdout
    mock.stderr = stderr
    return mock

class TestAudioPreprocessor:
    """Streamlined tests for AudioPreprocessor matching docs"""

    def test_probe_parses_ffprobe_output(self, tmp_path, monkeypatch):
        ap_cls = getattr(preprocessor_module, 'AudioPreprocessor', None)
        assert ap_cls is not None
        p = ap_cls()
        f = tmp_path / "audio.ogg"
        f.write_bytes(b"OggSFAKE")
        probe_out = {'format': {'duration': '2.34', 'format_name': 'ogg'}, 'streams': [{'codec_type': 'audio', 'sample_rate': '48000', 'channels': 2, 'codec_name': 'opus', 'bits_per_sample': '16'}]}
        monkeypatch.setattr(subprocess, 'run', lambda *a, **k: fake_subprocess_result(0, stdout=json.dumps(probe_out)))
        meta = p._probe_audio(f)
        assert meta['format'] == 'ogg'
        assert abs(meta['duration'] - 2.34) < 1e-6
        assert meta['sample_rate'] == 48000
        assert meta['channels'] == 2

    def test_preprocess_normalizes_when_needed(self, tmp_path, monkeypatch):
        ap_cls = getattr(preprocessor_module, 'AudioPreprocessor', None)
        assert ap_cls is not None
        p = ap_cls()
        src = tmp_path / "source.mp3"
        src.write_bytes(b"FAKE")
        # make probe show non-16k mono so normalization happens
        probe = {'format': {'duration': '1.5', 'format_name': 'mp3'}, 'streams': [{'codec_type': 'audio', 'sample_rate': '44100', 'channels': 2}]}
        monkeypatch.setattr(subprocess, 'run', lambda *a, **k: fake_subprocess_result(0, stdout=json.dumps(probe)))
        # stub normalize to return expected path
        out = tmp_path / "preprocessed.wav"
        monkeypatch.setattr(preprocessor_module.AudioPreprocessor, '_normalize_audio', lambda self, inp, meta: out)
        meta = p.preprocess(src)
        assert str(meta['normalized_path']) == str(out)

    def test_preprocess_rejects_too_short(self, tmp_path, monkeypatch):
        ap_cls = getattr(preprocessor_module, 'AudioPreprocessor', None)
        assert ap_cls is not None
        p = ap_cls()
        short = tmp_path / "short.wav"
        make_sine_wav(short, duration_s=0.5)
        probe_out = {'format': {'duration': '0.5', 'format_name': 'wav'}, 'streams': [{'codec_type': 'audio', 'sample_rate': '16000', 'channels': 1}]}
        monkeypatch.setattr(subprocess, 'run', lambda *a, **k: fake_subprocess_result(0, stdout=json.dumps(probe_out)))
        with pytest.raises(ValueError):
            p.preprocess(short)

    def test_normalize_failure_raises(self, tmp_path, monkeypatch):
        ap_cls = getattr(preprocessor_module, 'AudioPreprocessor', None)
        assert ap_cls is not None
        p = ap_cls()
        src = tmp_path / "bad.flac"
        src.write_bytes(b"CORRUPTED")
        # Simulate ffmpeg failure during normalization
        monkeypatch.setattr(subprocess, 'run', lambda *a, **k: fake_subprocess_result(1, stderr="ffmpeg error"))
        with pytest.raises(ValueError):
            p._normalize_audio(src, {'format': 'flac', 'sample_rate': 44100, 'channels': 2, 'duration': 1.2})

    def test_integration_calls_factory_with_preprocessed_path(self, tmp_path, monkeypatch):
        ap_cls = getattr(preprocessor_module, 'AudioPreprocessor', None)
        assert ap_cls is not None
        p = ap_cls()
        task_id = "task-123"
        raw_dir = tmp_path / "data" / "audio" / task_id
        raw_dir.mkdir(parents=True)
        raw = raw_dir / "raw.wav"
        make_sine_wav(raw, duration_s=1.5)
        preprocessed = raw_dir / "preprocessed.wav"
        # stub preprocess to return normalized path
        monkeypatch.setattr(preprocessor_module.AudioPreprocessor, 'preprocess', lambda self, path: {'format': 'wav', 'duration': 1.5, 'sample_rate': 16000, 'channels': 1, 'normalized_path': preprocessed})
        # mock factory.create to capture call
        from src.processors.asr import factory as asr_factory_module
        called = {}
        def fake_create(backend_type_arg, **kwargs):
            called['backend'] = backend_type_arg
            called['kwargs'] = kwargs
            called['config'] = kwargs.get('config', {})
            return "ASR_INSTANCE"
        monkeypatch.setattr(asr_factory_module.ASRProcessorFactory, 'create', staticmethod(fake_create))
        # call helper (process_task_audio)
        res = p.process_task_audio(task_id, str(raw))
        assert called['backend'] == 'whisper'
        assert called['config'].get('task_id') == task_id
        assert called['config'].get('audio_path') == str(preprocessed)