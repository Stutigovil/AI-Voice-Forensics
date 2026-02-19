"""
Audio Utilities
===============
Common audio processing functions for loading, preprocessing, and converting audio.

Features:
- Multi-format audio loading
- Resampling and normalization
- Audio segmentation
- Format conversion
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Union, List

import numpy as np
import librosa
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Audio processing utilities for loading and preprocessing audio files.
    
    Handles various audio formats and provides consistent output
    for downstream feature extraction and analysis.
    """
    
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus', '.webm']
    
    def __init__(
        self,
        target_sr: int = 16000,
        mono: bool = True,
        normalize: bool = True
    ):
        """
        Initialize the audio processor.
        
        Args:
            target_sr: Target sample rate for all audio
            mono: Convert to mono if True
            normalize: Normalize audio amplitude if True
        """
        self.target_sr = target_sr
        self.mono = mono
        self.normalize = normalize
    
    def load(
        self,
        audio_path: Union[str, Path],
        duration: Optional[float] = None,
        offset: float = 0.0
    ) -> Tuple[np.ndarray, int]:
        """
        Load an audio file with optional duration and offset.
        
        Args:
            audio_path: Path to the audio file
            duration: Duration in seconds to load (None for full file)
            offset: Start time in seconds
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if audio_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            logger.warning(f"Unsupported format {audio_path.suffix}, attempting to load anyway")
        
        # Load with librosa (handles most formats)
        y, sr = librosa.load(
            str(audio_path),
            sr=self.target_sr,
            mono=self.mono,
            duration=duration,
            offset=offset
        )
        
        # Normalize if requested
        if self.normalize:
            y = self._normalize_audio(y)
        
        return y, self.target_sr
    
    def _normalize_audio(self, y: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        max_val = np.abs(y).max()
        if max_val > 0:
            y = y / max_val
        return y
    
    def preprocess(
        self,
        audio_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        trim_silence: bool = True,
        top_db: int = 20
    ) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio for analysis.
        
        Args:
            audio_path: Path to input audio
            output_path: Optional path to save preprocessed audio
            trim_silence: Whether to trim silence from start/end
            top_db: Threshold in dB for silence trimming
            
        Returns:
            Tuple of (preprocessed_audio, sample_rate)
        """
        # Load audio
        y, sr = self.load(audio_path)
        
        # Trim silence if requested
        if trim_silence:
            y, _ = librosa.effects.trim(y, top_db=top_db)
        
        # Save if output path provided
        if output_path:
            self.save(y, sr, output_path)
        
        return y, sr
    
    def save(
        self,
        y: np.ndarray,
        sr: int,
        output_path: Union[str, Path],
        format: str = "wav"
    ) -> None:
        """
        Save audio to file.
        
        Args:
            y: Audio array
            sr: Sample rate
            output_path: Path to save audio
            format: Output format
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sf.write(str(output_path), y, sr, format=format)
        logger.info(f"Audio saved to: {output_path}")
    
    def get_duration(self, audio_path: Union[str, Path]) -> float:
        """Get duration of audio file in seconds."""
        return librosa.get_duration(path=str(audio_path))
    
    def resample(
        self,
        y: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return y
        return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    
    def segment_audio(
        self,
        y: np.ndarray,
        sr: int,
        segment_length: float = 5.0,
        overlap: float = 0.5
    ) -> List[np.ndarray]:
        """
        Split audio into overlapping segments.
        
        Args:
            y: Audio array
            sr: Sample rate
            segment_length: Length of each segment in seconds
            overlap: Overlap between segments (0-1)
            
        Returns:
            List of audio segments
        """
        segment_samples = int(segment_length * sr)
        hop_samples = int(segment_samples * (1 - overlap))
        
        segments = []
        for start in range(0, len(y) - segment_samples + 1, hop_samples):
            segment = y[start:start + segment_samples]
            segments.append(segment)
        
        # Handle last segment if it doesn't fit perfectly
        if len(y) > start + segment_samples:
            last_segment = y[-segment_samples:]
            segments.append(last_segment)
        
        return segments
    
    def convert_format(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_format: str = "wav"
    ) -> Path:
        """
        Convert audio file to different format.
        
        Args:
            input_path: Path to input audio
            output_path: Path for output audio
            target_format: Target format (wav, mp3, flac)
            
        Returns:
            Path to converted file
        """
        y, sr = self.load(input_path)
        output_path = Path(output_path)
        
        self.save(y, sr, output_path, format=target_format)
        return output_path


def load_audio(
    audio_path: str,
    sr: int = 16000,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Convenience function to load audio file.
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate
        mono: Convert to mono
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    processor = AudioProcessor(target_sr=sr, mono=mono)
    return processor.load(audio_path)


def preprocess_audio(
    audio_path: str,
    output_path: Optional[str] = None,
    sr: int = 16000,
    trim_silence: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Convenience function to preprocess audio.
    
    Args:
        audio_path: Path to audio file
        output_path: Optional path to save preprocessed audio
        sr: Target sample rate
        trim_silence: Whether to trim silence
        
    Returns:
        Tuple of (preprocessed_audio, sample_rate)
    """
    processor = AudioProcessor(target_sr=sr)
    return processor.preprocess(audio_path, output_path, trim_silence)


def get_audio_stats(audio_path: str) -> dict:
    """
    Get basic statistics about an audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with audio statistics
    """
    processor = AudioProcessor()
    y, sr = processor.load(audio_path)
    
    return {
        "duration": len(y) / sr,
        "sample_rate": sr,
        "num_samples": len(y),
        "max_amplitude": float(np.abs(y).max()),
        "mean_amplitude": float(np.abs(y).mean()),
        "rms": float(np.sqrt(np.mean(y**2)))
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python audio_utils.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    stats = get_audio_stats(audio_file)
    
    print("\nAudio Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
