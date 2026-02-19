"""
Prediction Module
=================
End-to-end prediction pipeline for AI voice detection.

This module combines:
1. Audio preprocessing
2. Speech-to-text transcription
3. Audio feature extraction
4. Text feature extraction
5. Model prediction
6. Explanation generation
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict

import numpy as np

from ..stt.transcribe import WhisperTranscriber
from ..feature_extraction.audio_features import AudioFeatureExtractor
from ..feature_extraction.text_features import TextFeatureExtractor
from .train_lgbm import AIVoiceClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results."""
    prediction: int  # 0 = Human, 1 = AI
    prediction_label: str
    confidence: float
    probability_ai: float
    
    # Transcription
    transcript: str
    transcript_word_count: int
    audio_duration: float
    
    # Feature analysis
    top_audio_features: Dict[str, float]
    top_text_features: Dict[str, float]
    anomalous_features: List[str]
    
    # Explainability
    explanation_summary: str
    shap_values: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class Predictor:
    """
    End-to-end predictor for AI voice detection.
    
    Pipeline:
    1. Load and preprocess audio
    2. Transcribe with Whisper
    3. Extract audio features (MFCC, pitch, energy, etc.)
    4. Extract text features (perplexity, disfluency, etc.)
    5. Combine features and run classifier
    6. Generate human-readable explanation
    """
    
    # Feature thresholds for anomaly detection
    ANOMALY_THRESHOLDS = {
        'disfluency_rate': {'low': 0.5, 'high': 10.0, 'direction': 'low_suspicious'},
        'perplexity': {'low': 20, 'high': 200, 'direction': 'low_suspicious'},
        'pitch_var': {'low': 10, 'high': 500, 'direction': 'low_suspicious'},
        'sentence_length_var': {'low': 2, 'high': 50, 'direction': 'low_suspicious'},
        'filler_count': {'low': 0, 'high': 20, 'direction': 'low_suspicious'}
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        whisper_model: str = "base",
        use_gpu: bool = False
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to saved LightGBM model
            whisper_model: Whisper model size
            use_gpu: Whether to use GPU for inference
        """
        self.device = "cuda" if use_gpu else "cpu"
        
        # Initialize components
        logger.info("Initializing prediction pipeline...")
        
        self.transcriber = WhisperTranscriber(
            model_size=whisper_model,
            device=self.device
        )
        
        self.audio_extractor = AudioFeatureExtractor()
        self.text_extractor = TextFeatureExtractor(use_transformers=True)
        
        # Load classifier
        self.classifier = AIVoiceClassifier()
        if model_path and Path(model_path).exists():
            self.classifier.load(model_path)
            self.model_loaded = True
        else:
            logger.warning("No model loaded. Call load_model() before prediction.")
            self.model_loaded = False
        
        logger.info("Prediction pipeline initialized")
    
    def load_model(self, model_path: str) -> None:
        """Load a trained classification model."""
        self.classifier.load(model_path)
        self.model_loaded = True
        logger.info(f"Loaded model from {model_path}")
    
    def extract_features(
        self,
        audio_path: str,
        transcript: Optional[str] = None
    ) -> tuple:
        """
        Extract all features from audio.
        
        Args:
            audio_path: Path to audio file
            transcript: Optional pre-computed transcript
            
        Returns:
            Tuple of (features_array, audio_features, text_features, transcript)
        """
        # Get transcript if not provided
        if transcript is None:
            logger.info("Transcribing audio...")
            transcription = self.transcriber.transcribe(audio_path)
            transcript = transcription.text
        
        # Extract audio features
        logger.info("Extracting audio features...")
        audio_features = self.audio_extractor.extract(audio_path)
        
        # Extract text features
        logger.info("Extracting text features...")
        text_features = self.text_extractor.extract(transcript)
        
        # Combine features
        combined_features = np.concatenate([
            audio_features.to_flat_array(),
            text_features.to_flat_array()
        ])
        
        return combined_features, audio_features, text_features, transcript
    
    def detect_anomalies(
        self,
        audio_features,
        text_features
    ) -> List[str]:
        """
        Detect anomalous features that may indicate AI generation.
        
        Args:
            audio_features: Extracted audio features
            text_features: Extracted text features
            
        Returns:
            List of anomalous feature descriptions
        """
        anomalies = []
        
        # Check text features
        text_dict = text_features.to_dict()
        
        # Low disfluency rate (AI typically has very few "uh", "um")
        if text_dict['disfluency_rate'] < self.ANOMALY_THRESHOLDS['disfluency_rate']['low']:
            anomalies.append("Very low disfluency rate (no fillers like 'uh', 'um')")
        
        # Low perplexity (text too predictable, likely LLM-generated)
        if text_dict['perplexity'] < self.ANOMALY_THRESHOLDS['perplexity']['low']:
            anomalies.append("Unusually low text perplexity (highly predictable text)")
        
        # Low sentence length variance (too uniform)
        if text_dict['sentence_length_var'] < self.ANOMALY_THRESHOLDS['sentence_length_var']['low']:
            anomalies.append("Very uniform sentence lengths (unnatural pattern)")
        
        # No filler words
        if text_dict['filler_count'] == 0:
            anomalies.append("Complete absence of filler words")
        
        # Check audio features
        audio_dict = audio_features.to_dict()
        
        # Low pitch variance (monotone voice)
        if audio_dict['pitch_var'] < self.ANOMALY_THRESHOLDS['pitch_var']['low']:
            anomalies.append("Very low pitch variation (monotone delivery)")
        
        # Very low jitter (too smooth)
        if audio_dict['pitch_jitter'] < 0.01:
            anomalies.append("Abnormally low pitch jitter (unnaturally smooth)")
        
        # High spectral flatness (vocoder artifacts)
        if audio_dict['spectral_flatness_mean'] > 0.5:
            anomalies.append("High spectral flatness (possible vocoder artifacts)")
        
        return anomalies
    
    def generate_explanation(
        self,
        prediction: int,
        confidence: float,
        anomalies: List[str],
        audio_features,
        text_features
    ) -> str:
        """
        Generate a human-readable explanation for the prediction.
        
        Args:
            prediction: Model prediction (0/1)
            confidence: Prediction confidence
            anomalies: List of detected anomalies
            audio_features: Audio feature object
            text_features: Text feature object
            
        Returns:
            Explanation string
        """
        label = "AI-Generated" if prediction == 1 else "Human"
        
        explanation_parts = [
            f"Classification: {label} (Confidence: {confidence:.1%})",
            ""
        ]
        
        if prediction == 1:  # AI-generated
            explanation_parts.append("Evidence suggesting AI generation:")
            
            if anomalies:
                for anomaly in anomalies:
                    explanation_parts.append(f"  • {anomaly}")
            else:
                explanation_parts.append("  • Overall feature pattern matches AI-generated samples")
            
            # Add specific feature evidence
            text_dict = text_features.to_dict()
            audio_dict = audio_features.to_dict()
            
            explanation_parts.append("")
            explanation_parts.append("Key indicators:")
            explanation_parts.append(f"  • Disfluency rate: {text_dict['disfluency_rate']:.2f}% (human average: 3-7%)")
            explanation_parts.append(f"  • Text perplexity: {text_dict['perplexity']:.1f} (lower = more AI-like)")
            explanation_parts.append(f"  • Pitch jitter: {audio_dict['pitch_jitter']:.4f} (human range: 0.02-0.05)")
            
        else:  # Human
            explanation_parts.append("Evidence suggesting human speech:")
            explanation_parts.append("  • Natural speech patterns detected")
            
            text_dict = text_features.to_dict()
            audio_dict = audio_features.to_dict()
            
            if text_dict['disfluency_rate'] > 1:
                explanation_parts.append(f"  • Natural disfluency rate: {text_dict['disfluency_rate']:.2f}%")
            if text_dict['filler_count'] > 0:
                explanation_parts.append(f"  • Filler words present: {text_dict['filler_count']}")
            if audio_dict['pitch_jitter'] > 0.02:
                explanation_parts.append(f"  • Natural pitch variation (jitter: {audio_dict['pitch_jitter']:.4f})")
        
        return "\n".join(explanation_parts)
    
    def predict(
        self,
        audio_path: str,
        return_features: bool = False
    ) -> Union[PredictionResult, tuple]:
        """
        Run full prediction pipeline on an audio file.
        
        Args:
            audio_path: Path to audio file
            return_features: If True, also return raw features
            
        Returns:
            PredictionResult (and optionally raw features)
        """
        if not self.model_loaded:
            raise RuntimeError("No model loaded. Call load_model() first or provide model_path in constructor.")
        
        logger.info(f"Processing: {audio_path}")
        
        # Extract features
        features, audio_features, text_features, transcript = self.extract_features(audio_path)
        
        # Get prediction
        features_2d = features.reshape(1, -1)
        prediction = self.classifier.predict(features_2d)[0]
        probability = self.classifier.predict_proba(features_2d)[0]
        
        confidence = probability if prediction == 1 else (1 - probability)
        
        # Get SHAP explanation
        explanation_dict = self.classifier.explain_prediction(features_2d, use_shap=True)
        
        # Detect anomalies
        anomalies = self.detect_anomalies(audio_features, text_features)
        
        # Generate explanation
        explanation_text = self.generate_explanation(
            prediction, confidence, anomalies, audio_features, text_features
        )
        
        # Get top contributing features
        audio_dict = audio_features.to_dict()
        text_dict = text_features.to_dict()
        
        top_audio = {k: v for k, v in sorted(
            {k: v for k, v in audio_dict.items() if isinstance(v, (int, float))}.items(),
            key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
            reverse=True
        )[:5]}
        
        top_text = {k: v for k, v in sorted(
            {k: v for k, v in text_dict.items() if isinstance(v, (int, float))}.items(),
            key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
            reverse=True
        )[:5]}
        
        # Create result
        result = PredictionResult(
            prediction=int(prediction),
            prediction_label="AI-Generated" if prediction == 1 else "Human",
            confidence=float(confidence),
            probability_ai=float(probability),
            transcript=transcript,
            transcript_word_count=len(transcript.split()),
            audio_duration=audio_features.duration,
            top_audio_features=top_audio,
            top_text_features=top_text,
            anomalous_features=anomalies,
            explanation_summary=explanation_text,
            shap_values=explanation_dict.get('shap_values')
        )
        
        if return_features:
            return result, features, audio_features, text_features
        
        return result
    
    def predict_batch(
        self,
        audio_paths: List[str]
    ) -> List[PredictionResult]:
        """
        Run predictions on multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            List of PredictionResult objects
        """
        results = []
        
        for path in audio_paths:
            try:
                result = self.predict(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                continue
        
        return results


def predict_audio(
    audio_path: str,
    model_path: str,
    whisper_model: str = "base"
) -> Dict:
    """
    Convenience function for single audio prediction.
    
    Args:
        audio_path: Path to audio file
        model_path: Path to saved model
        whisper_model: Whisper model size
        
    Returns:
        Dictionary with prediction results
    """
    predictor = Predictor(
        model_path=model_path,
        whisper_model=whisper_model
    )
    
    result = predictor.predict(audio_path)
    return result.to_dict()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python predict.py <audio_file> <model_path>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    model_path = sys.argv[2]
    
    predictor = Predictor(model_path=model_path)
    result = predictor.predict(audio_file)
    
    print("\n" + "=" * 60)
    print("AI VOICE DETECTION RESULT")
    print("=" * 60)
    print(f"\nPrediction: {result.prediction_label}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"\nTranscript preview: {result.transcript[:200]}...")
    print(f"\n{result.explanation_summary}")
