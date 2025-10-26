#!/usr/bin/env python3
"""
Test script to see what faster-whisper returns when word_timestamps=True
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.logging import configure_logging
from src.processors.asr.factory import ASRFactory

def main():
    configure_logging()
    
    # Use the same audio file from the logs
    audio_path = "data/audio/9cccbecf-305d-49db-b899-fb794ae7fecd/preprocessed.wav"
    
    if not Path(audio_path).exists():
        print(f"Audio file not found: {audio_path}")
        return

    print("Testing Whisper word timestamps...")

    # Load Whisper with word timestamps enabled
    asr_config = {
        "model_path": "data/models/era-x-wow-turbo-v1.1-ct2",
        "device": "cuda",
        "compute_type": "float16",
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500, "speech_pad_ms": 400},
        "word_timestamps": True,  # Enable word timestamps
    }
    
    asr_backend = ASRFactory.create("whisper", **asr_config)
    
    print("Running ASR with word timestamps...")
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    
    asr_result = asr_backend.execute(audio_data=audio_data)
    
    print(f"ASR completed: {len(asr_result.segments)} segments")
    
    # Check what's actually in the segments
    print("\nFirst segment details:")
    if asr_result.segments:
        first_seg = asr_result.segments[0]
        print(f"Segment type: {type(first_seg)}")
        print(f"Segment keys: {first_seg.keys() if isinstance(first_seg, dict) else dir(first_seg)}")
        print(f"Segment content: {first_seg}")
        
        # Check if there are words
        if hasattr(first_seg, 'words'):
            print(f"Words attribute: {first_seg.words}")
        elif isinstance(first_seg, dict) and 'words' in first_seg:
            print(f"Words in dict: {first_seg['words']}")
        else:
            print("No words found in segment")
    
    # Let's also check the raw faster-whisper output
    print("\nChecking raw faster-whisper output...")
    
    # Import faster-whisper directly to see what it returns
    try:
        from faster_whisper import WhisperModel
        
        model = WhisperModel(
            "data/models/era-x-wow-turbo-v1.1-ct2",
            device="cuda",
            compute_type="float16"
        )
        
        segments, info = model.transcribe(audio_path, word_timestamps=True)
        
        print("Raw faster-whisper segments:")
        for i, segment in enumerate(segments):
            if i >= 2:  # Only show first 2 segments
                break
            print(f"Segment {i}:")
            print(f"  Type: {type(segment)}")
            print(f"  Start: {segment.start}")
            print(f"  End: {segment.end}")
            print(f"  Text: {segment.text}")
            
            # Check for words attribute
            if hasattr(segment, 'words'):
                print(f"  Words: {segment.words}")
                if segment.words:
                    print(f"  First word: {segment.words[0]}")
                    print(f"  First word type: {type(segment.words[0])}")
                    if hasattr(segment.words[0], '__dict__'):
                        print(f"  First word attributes: {segment.words[0].__dict__}")
            else:
                print("  No words attribute found")
            print()
            
    except Exception as e:
        print(f"Error testing raw faster-whisper: {e}")

if __name__ == "__main__":
    main()
