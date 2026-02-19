"""
AI Voice Detection
==================

A multimodal AI voice authenticity detection system combining:
- Speech-to-Text (Whisper)
- Audio feature analysis (MFCC, pitch, energy)
- Text/linguistic analysis (perplexity, disfluency)
- LightGBM classification with SHAP explainability
- RAG-powered forensic explanations

Author: AI Voice Detection Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Voice Detection Team"

from .model.predict import Predictor, predict_audio
from .model.train_lgbm import AIVoiceClassifier, train_model
from .stt.transcribe import WhisperTranscriber
from .feature_extraction.audio_features import AudioFeatureExtractor
from .feature_extraction.text_features import TextFeatureExtractor
from .rag.llm_explainer import LLMExplainer

__all__ = [
    "Predictor",
    "predict_audio",
    "AIVoiceClassifier",
    "train_model",
    "WhisperTranscriber",
    "AudioFeatureExtractor",
    "TextFeatureExtractor",
    "LLMExplainer"
]
