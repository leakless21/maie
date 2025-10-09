"""Audio preprocessor module for MAIE.

This module provides audio preprocessing capabilities including format normalization,
quality validation, and metadata extraction using ffmpeg/ffprobe.
"""

import subprocess
import tempfile
import os
from typing import Dict, Any, Optional


class AudioPreprocessor:
    """Audio preprocessor for normalizing and validating audio files."""
    
    def __init__(self):
        """Initialize the audio preprocessor."""
        pass
    
    def normalize_format(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Normalize audio to WAV 16kHz mono format.
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output file (optional, creates temp file if not provided)
            
        Returns:
            Path to normalized audio file
        """
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        # Use ffmpeg to convert to WAV 16kHz mono
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ar', '16000',  # sample rate 16kHz
            '-ac', '1',      # mono
            '-sample_fmt', 's16',  # 16-bit
            '-y',            # overwrite output file
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"FFmpeg failed to normalize audio: {result.stderr}")
        
        return output_path
    
    def validate_quality(self, audio_path: str) -> Dict[str, Any]:
        """Validate audio quality and return quality metrics.
        
        Args:
            audio_path: Path to audio file to validate
            
        Returns:
            Dictionary containing quality metrics
        """
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Get audio info
            info = self.get_audio_info(audio_path)
            
            # Validate audio properties
            quality_metrics = {
                'is_valid': True,
                'duration': info.get('duration', 0),
                'sample_rate': info.get('sample_rate', 0),
                'channels': info.get('channels', 0),
                'bit_depth': info.get('bit_depth', 0),
                'format': info.get('format', ''),
                'size_bytes': os.path.getsize(audio_path)
            }
            
            # Additional quality checks can be added here
            if info.get('sample_rate', 0) < 8000:
                quality_metrics['is_valid'] = False
                quality_metrics['issues'] = quality_metrics.get('issues', []) + ['low_sample_rate']
            
            return quality_metrics
            
        except Exception as e:
            return {
                'is_valid': False,
                'error': str(e),
                'issues': ['validation_error']
            }
    
    def get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """Get detailed audio information using ffprobe.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing audio metadata
        """
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-show_format',
            '-show_streams',
            '-print_format', 'json',
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"FFprobe failed to get audio info: {result.stderr}")
        
        import json
        probe_data = json.loads(result.stdout)
        
        # Extract audio stream info
        audio_stream = None
        for stream in probe_data.get('streams', []):
            if stream.get('codec_type') == 'audio':
                audio_stream = stream
                break
        
        if audio_stream is None:
            raise ValueError("No audio stream found in file")
        
        # Extract format info
        format_info = probe_data.get('format', {})
        
        return {
            'duration': float(format_info.get('duration', 0)),
            'sample_rate': int(audio_stream.get('sample_rate', 0)),
            'channels': int(audio_stream.get('channels', 0)),
            'bit_rate': format_info.get('bit_rate', 0),
            'format': format_info.get('format_name', ''),
            'codec': audio_stream.get('codec_name', ''),
            'bit_depth': int(audio_stream.get('bits_per_sample', 0))
        }