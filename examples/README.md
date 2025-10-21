# MAIE Examples

This directory contains example scripts for testing and using different components of the MAIE (Multimodal AI Engine) system. Each script demonstrates how to use specific processors and backends for various AI tasks.

## Overview

The MAIE system consists of several key components that can be tested independently:

- **Audio Preprocessing**: Validate and normalize audio files
- **ASR (Automatic Speech Recognition)**: Convert speech to text using different backends
- **LLM Processing**: Text enhancement and summary using vLLM

## Prerequisites

Before running these examples, ensure you have:

1. **Pixi package manager** installed:

   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. **Dependencies installed**:

   ```bash
   pixi install
   ```

3. **Required models and data**:
   - Audio files for testing (place in `data/audio/`)
   - ASR models (Whisper, ChunkFormer)
   - LLM models for text processing

## Example Scripts

### 1. Audio Preprocessing (`infer_preprocess.py`)

Validates and normalizes audio files for ASR input.

**Usage:**

```bash
# Basic preprocessing
pixi run python examples/infer_preprocess.py --input-audio data/audio/sample.wav

# With custom output path
pixi run python examples/infer_preprocess.py --input-audio input.mp3 --output-audio normalized.wav
```

**Output:**

- JSON format with audio metadata
- Keys: `format`, `duration`, `sample_rate`, `channels`, `normalized`, `normalized_path`

**Example output:**

```json
{
  "format": "wav",
  "duration": 5.2,
  "sample_rate": 16000,
  "channels": 1,
  "normalized": true,
  "normalized_path": "/path/to/normalized.wav"
}
```

### 2. ASR with Whisper (`infer_asr_whisper.py`)

Speech-to-text conversion using OpenAI's Whisper model.

**Usage:**

```bash
# Basic transcription
pixi run python examples/infer_asr_whisper.py --audio data/audio/sample.wav

# With custom model and parameters
pixi run python examples/infer_asr_whisper.py --audio input.wav --model-path data/models/whisper --beam-size 5 --vad-filter

# JSON output with metrics
pixi run python examples/infer_asr_whisper.py --audio input.wav --json
```

**Parameters:**

- `--audio`: Path to input audio file (required)
- `--model-path`: Custom model path (optional)
- `--beam-size`: Beam search size (optional)
- `--vad-filter`: Enable voice activity detection filtering
- `--json`: Output results in JSON format

**Example output:**

```
Hello, this is a test transcription.
```

**JSON output:**

```json
{
  "transcript": "Hello, this is a test transcription.",
  "rtf": 0.15,
  "confidence": 0.95
}
```

### 3. ASR with ChunkFormer (`infer_asr_chunkformer.py`)

Speech-to-text conversion using ChunkFormer model for streaming/real-time scenarios.

**Usage:**

```bash
# Basic transcription
pixi run python examples/infer_asr_chunkformer.py --audio data/audio/sample.wav

# With context parameters
pixi run python examples/infer_asr_chunkformer.py --audio input.wav --left-context 5 --right-context 5

# JSON output
pixi run python examples/infer_asr_chunkformer.py --audio input.wav --json
```

**Parameters:**

- `--audio`: Path to input audio file (required)
- `--model-path`: Custom model path (optional)
- `--left-context`: Left context window size (optional)
- `--right-context`: Right context window size (optional)
- `--json`: Output results in JSON format

### 4. LLM Processing (`infer_vllm.py`)

Text enhancement and summary using vLLM backend.

**Usage:**

```bash
# Text enhancement
pixi run python examples/infer_vllm.py --task enhancement --text "hello world"

# Meeting summary
pixi run python examples/infer_vllm.py --task summary --text "...transcript..." --template-id meeting

# With custom model and parameters
pixi run python examples/infer_vllm.py --task enhancement --text "input text" --model-path data/models/llm --temperature 0.7 --max-tokens 200
```

**Parameters:**

- `--task`: Task type (`enhancement` or `summary`)
- `--text`: Input text to process (required)
- `--template-id`: Template ID for summary tasks
- `--model-path`: Custom model path (optional)
- `--temperature`: Sampling temperature (0.0-2.0)
- `--top-p`: Top-p sampling parameter
- `--top-k`: Top-k sampling parameter
- `--max-tokens`: Maximum tokens to generate

**Example outputs:**

- **Enhancement**: Enhanced/cleaned text
- **Summary**: Structured JSON summary or formatted text

### 5. Vietnamese LLM Testing (`run_vietnamese_tests.py`)

Comprehensive testing suite for Vietnamese language processing.

**Usage:**

```bash
# Run all Vietnamese tests
pixi run python examples/run_vietnamese_tests.py

# Show help and configuration options
pixi run python examples/run_vietnamese_tests.py --help
```

**Environment Variables:**

- `LLM_TEST_MODEL_PATH`: Path to local LLM model
- `LLM_TEST_API_KEY`: API key for cloud LLM service
- `LLM_TEST_TEMPERATURE`: Generation temperature
- `LLM_TEST_MAX_TOKENS`: Maximum tokens
- `LLM_TEST_TIMEOUT`: Request timeout

### 6. vLLM Testing (`test_vllm.py`)

Basic vLLM functionality test with sample prompts.

**Usage:**

```bash
pixi run python examples/test_vllm.py
```

## Complete Workflow Example

Here's how to test the complete MAIE pipeline:

```bash
# 1. Preprocess audio
pixi run python examples/infer_preprocess.py --input-audio data/audio/meeting.wav --output-audio normalized.wav

# 2. Transcribe audio (choose one backend)
pixi run python examples/infer_asr_whisper.py --audio normalized.wav --json > transcript.json

# 3. Enhance transcript
pixi run python examples/infer_vllm.py --task enhancement --text "$(cat transcript.json | jq -r '.transcript')"

# 4. Summarize meeting
pixi run python examples/infer_vllm.py --task summary --text "$(cat transcript.json | jq -r '.transcript')" --template-id meeting
```

## Configuration

### Model Paths

Configure model paths in `src/config/settings.py` or use environment variables:

```bash
export LLM_TEST_MODEL_PATH="/path/to/your/llm/model"
export ASR_MODEL_PATH="/path/to/your/asr/model"
```

### Audio Requirements

- **Format**: WAV, MP3, FLAC, M4A
- **Sample Rate**: 16kHz (will be normalized if different)
- **Channels**: Mono or stereo (will be converted to mono)
- **Duration**: Any length (longer files will be chunked)

### Error Handling

All scripts provide comprehensive error handling:

- **File not found**: Clear error messages with suggested solutions
- **Model loading errors**: Detailed error reporting
- **Processing failures**: Graceful error handling with cleanup
- **JSON output**: Structured error responses for programmatic use

## Troubleshooting

### Common Issues

1. **"Audio not found" error**:

   - Check file path and permissions
   - Ensure audio file exists and is readable

2. **"Model loading failed" error**:

   - Verify model path is correct
   - Check model file integrity
   - Ensure sufficient disk space and memory

3. **"Dependencies not found" error**:

   - Run `pixi install` to install dependencies
   - Check Python environment setup

4. **"Permission denied" error**:
   - Check file permissions
   - Ensure write access to output directories

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
export LOG_LEVEL=DEBUG
pixi run python examples/infer_preprocess.py --input-audio data/audio/sample.wav
```

## Performance Notes

- **ASR Processing**: Whisper is generally faster for short audio, ChunkFormer for streaming
- **LLM Processing**: Local models require significant GPU memory, cloud APIs have rate limits
- **Audio Preprocessing**: Usually very fast, normalization may take longer for large files

## Support

For issues or questions:

1. Check the main project documentation
2. Review error messages and logs
3. Verify configuration settings
4. Test with sample data first

## License

This examples directory is part of the MAIE project. See the main project license for details.
