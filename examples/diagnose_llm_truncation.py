#!/usr/bin/env python3
"""
Diagnose LLM truncation issues by analyzing token counts and limits.

Usage:
  python examples/diagnose_llm_truncation.py --audio data/audio/sample.mp3
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import configure_logging, get_logger, settings
from src.config.logging import get_module_logger
from src.processors.audio import AudioPreprocessor
from src.processors.asr.factory import ASRFactory
from src.processors.llm import LLMProcessor
from src.processors.llm.schema_validator import load_template_schema


def main() -> int:
    logger = configure_logging() or get_logger()
    logger = get_module_logger(__name__)

    parser = argparse.ArgumentParser(description="Diagnose LLM truncation issues")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument(
        "--template", default="interview_transcript_v1", help="Template ID"
    )
    parser.add_argument(
        "--backend", default="chunkformer", help="ASR backend"
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return 2

    print("=" * 80)
    print("DIAGNOSTIC REPORT: LLM Truncation Analysis")
    print("=" * 80)

    # Step 1: Get ASR transcript
    print("\n1. Extracting ASR transcript...")
    asr = None
    try:
        preprocessor = AudioPreprocessor()
        metadata = preprocessor.preprocess(audio_path)
        processing_audio_path = metadata.get("normalized_path") or audio_path
        audio_duration = float(metadata.get("duration", 0.0))

        asr = ASRFactory.create(args.backend)
        import time
        start_time = time.time()
        asr_result = asr.execute(audio_data=open(processing_audio_path, "rb").read())
        processing_time = time.time() - start_time
        rtf = processing_time / audio_duration if audio_duration > 0 else 0.0
        transcript = asr_result.text
        
        char_count = len(transcript)
        word_count = len(transcript.split())
        
        print(f"   ✓ ASR complete: {char_count:,} chars, {word_count:,} words")
    except Exception as e:
        logger.exception(f"ASR failed: {e}")
        return 3
    finally:
        if asr:
            asr.unload()

    # Step 2: Load LLM and tokenizer
    print("\n2. Loading LLM and analyzing token counts...")
    llm = None
    try:
        llm = LLMProcessor()
        llm._load_model()
        llm._ensure_tokenizer(llm.model_path)
        
        # Count transcript tokens
        transcript_tokens = len(llm.tokenizer.encode(transcript, add_special_tokens=False))
        print(f"   ✓ Transcript tokens: {transcript_tokens:,}")
        
        # Load template schema
        schema = load_template_schema(args.template)
        
        # Build prompt (simplified version of what generate_summary does)
        prompt_template = schema.get("prompt_template", "")
        prompt = prompt_template.replace("{{transcript}}", transcript)
        prompt_tokens = len(llm.tokenizer.encode(prompt, add_special_tokens=False))
        
        print(f"   ✓ Full prompt tokens: {prompt_tokens:,}")
        
        # Analyze configuration
        max_model_len = settings.llm_sum_max_model_len
        max_tokens_setting = settings.llm_sum_max_tokens
        
        print(f"\n3. Configuration Analysis:")
        print(f"   max_model_len: {max_model_len:,} tokens (total context window)")
        print(f"   max_tokens setting: {max_tokens_setting or 'None (dynamic)'}")
        
        # Calculate available output space
        available_output = max_model_len - prompt_tokens - 128  # 128 token safety margin
        
        print(f"\n4. Token Budget Breakdown:")
        print(f"   Total context window:  {max_model_len:,} tokens")
        print(f"   Used by prompt:        {prompt_tokens:,} tokens ({(prompt_tokens/max_model_len)*100:.1f}%)")
        print(f"   Safety margin:         128 tokens")
        print(f"   Available for output:  {available_output:,} tokens")
        
        # Dynamic calculation for summary task (30% compression)
        recommended_output = int(transcript_tokens * 0.3)
        min_output = 128
        max_safe_output = min(available_output, max(recommended_output, min_output))
        
        print(f"\n5. Recommendations:")
        print(f"   Recommended output tokens: {recommended_output:,} (30% of input)")
        print(f"   Safe maximum: {max_safe_output:,}")
        
        if available_output < recommended_output:
            print(f"\n   ⚠️  WARNING: Not enough space for recommended output!")
            print(f"   - Prompt uses {prompt_tokens:,} tokens, leaving only {available_output:,} for output")
            print(f"   - Need at least {recommended_output:,} tokens for quality summary")
            print(f"\n   SOLUTIONS:")
            print(f"   1. Increase max_model_len to at least {prompt_tokens + recommended_output + 128:,}")
            print(f"   2. Use a shorter transcript or split into chunks")
            print(f"   3. Simplify the prompt template")
        else:
            print(f"\n   ✓ Configuration looks good!")
            print(f"   - {available_output:,} tokens available")
            print(f"   - {recommended_output:,} tokens recommended")
            print(f"   - {available_output - recommended_output:,} tokens buffer")
        
        # Test actual generation
        print(f"\n6. Testing actual generation...")
        try:
            result = llm.generate_summary(
                transcript=transcript,
                template_id=args.template,
            )
            
            if result.get("summary"):
                print(f"   ✓ Generation successful!")
                print(f"   Retry count: {result.get('retry_count', 0)}")
                summary_str = json.dumps(result["summary"], ensure_ascii=False)
                print(f"   Output length: {len(summary_str):,} chars")
            else:
                error = result.get("error", "Unknown error")
                print(f"   ✗ Generation failed: {error}")
                return 1
                
        except Exception as e:
            logger.exception(f"Generation test failed: {e}")
            return 1
        
        print("\n" + "=" * 80)
        print("DIAGNOSIS COMPLETE")
        print("=" * 80 + "\n")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Diagnostic failed: {e}")
        return 1
    finally:
        if llm:
            llm.unload()


if __name__ == "__main__":
    raise SystemExit(main())
