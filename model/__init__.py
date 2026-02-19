"""
Model Module
============
LightGBM-based classification for AI voice detection.
"""

from .train_lgbm import AIVoiceClassifier, train_model
from .predict import Predictor, predict_audio

__all__ = [
    "AIVoiceClassifier",
    "train_model",
    "Predictor",
    "predict_audio"
]
