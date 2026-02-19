"""
Whisper Speech-to-Text Transcription Module
============================================
Converts audio files to text using OpenAI's Whisper model.

Features:
- Multi-format audio support
- Word-level timestamps
- Configurable model sizes
- GPU/CPU support
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict

import whisper
import torch
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """Represents a segment of transcribed text with timing information."""
    id: int
    start: float
    end: float
    text: str
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TranscriptionResult:
    """Complete transcription result with metadata."""
    text: str
    segments: List[TranscriptSegment]
    language: str
    duration: float
    word_count: int
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "segments": [seg.to_dict() for seg in self.segments],
            "language": self.language,
            "duration": self.duration,
            "word_count": self.word_count
        }
    
    def to_json(self, filepath: str) -> None:
        """Save transcription to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class WhisperTranscriber:
    """
    Audio transcription using OpenAI Whisper.
    
    Supports multiple model sizes for accuracy/speed tradeoffs:
    - tiny: Fastest, least accurate (~1GB VRAM)
    - base: Good balance (~1GB VRAM)
    - small: Better accuracy (~2GB VRAM)
    - medium: High accuracy (~5GB VRAM)
    - large: Best accuracy (~10GB VRAM)
    """
    
    SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    
    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        language: str = "en"
    ):
        """
        Initialize the Whisper transcriber.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on ('cpu' or 'cuda'). Auto-detected if None.
            language: Target language code (e.g., 'en' for English)
        """
        if model_size not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model size must be one of: {self.SUPPORTED_MODELS}")
        
        self.model_size = model_size
        self.language = language
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing Whisper model '{model_size}' on device '{self.device}'")
        
        # Load Whisper model
        self.model = whisper.load_model(model_size, device=self.device)
        logger.info("Whisper model loaded successfully")
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        word_timestamps: bool = True,
        verbose: bool = False
    ) -> TranscriptionResult:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path: Path to the audio file
            word_timestamps: Whether to include word-level timestamps
            verbose: Whether to print progress
            
        Returns:
            TranscriptionResult with full transcription and segments
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing: {audio_path}")
        
        # Transcribe with Whisper
        result = self.model.transcribe(
            str(audio_path),
            language=self.language,
            word_timestamps=word_timestamps,
            verbose=verbose,
            fp16=(self.device == "cuda")  # Use FP16 on GPU for speed
        )
        
        # Extract segments with timestamps
        segments = []
        for i, seg in enumerate(result.get("segments", [])):
            segment = TranscriptSegment(
                id=i,
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                confidence=seg.get("avg_logprob", 0.0)
            )
            segments.append(segment)
        
        # Calculate duration from segments
        duration = segments[-1].end if segments else 0.0
        
        # Create result object
        transcription = TranscriptionResult(
            text=result["text"].strip(),
            segments=segments,
            language=result.get("language", self.language),
            duration=duration,
            word_count=len(result["text"].split())
        )
        
        logger.info(f"Transcription complete: {transcription.word_count} words, {duration:.2f}s")
        
        return transcription
    
    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[TranscriptionResult]:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            output_dir: Optional directory to save transcript JSONs
            
        Returns:
            List of TranscriptionResult objects
        """
        results = []
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        for audio_path in audio_paths:
            try:
                result = self.transcribe(audio_path)
                results.append(result)
                
                # Save to JSON if output directory specified
                if output_dir:
                    stem = Path(audio_path).stem
                    result.to_json(output_dir / f"{stem}_transcript.json")
                    
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_path}: {e}")
                continue
        
        return results
    
    def get_words_with_timestamps(
        self,
        audio_path: Union[str, Path]
    ) -> List[Dict]:
        """
        Get word-level transcription with precise timestamps.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of dicts with 'word', 'start', 'end' keys
        """
        result = self.model.transcribe(
            str(audio_path),
            language=self.language,
            word_timestamps=True
        )
        
        words = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                words.append({
                    "word": word_info["word"].strip(),
                    "start": word_info["start"],
                    "end": word_info["end"],
                    "probability": word_info.get("probability", 0.0)
                })
        
        return words


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    output_path: Optional[str] = None
) -> Dict:
    """
    Convenience function to transcribe a single audio file.
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size
        output_path: Optional path to save transcript JSON
        
    Returns:
        Dictionary with transcription results
    """
    transcriber = WhisperTranscriber(model_size=model_size)
    result = transcriber.transcribe(audio_path)
    
    if output_path:
        result.to_json(output_path)
    
    return result.to_dict()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    transcriber = WhisperTranscriber(model_size="base")
    result = transcriber.transcribe(audio_file)
    
    print(f"\nTranscript:\n{result.text}")
    print(f"\nDuration: {result.duration:.2f}s")
    print(f"Word count: {result.word_count}")
