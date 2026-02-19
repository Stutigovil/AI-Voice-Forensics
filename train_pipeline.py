"""
Training Pipeline
=================
Complete training pipeline for AI voice detection model.

Usage:
    python train_pipeline.py --data-dir data/raw --output-dir model/saved
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import json

import numpy as np
from tqdm import tqdm

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from stt.transcribe import WhisperTranscriber
from feature_extraction.audio_features import AudioFeatureExtractor
from feature_extraction.text_features import TextFeatureExtractor
from model.train_lgbm import AIVoiceClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_audio_files(data_dir: Path) -> Tuple[List[Path], List[int]]:
    """
    Collect audio files from directory structure.
    
    Expected structure:
    data_dir/
    ├── human/      # Label 0
    │   └── *.wav
    └── ai/         # Label 1
        └── *.wav
    """
    audio_files = []
    labels = []
    
    human_dir = data_dir / "human"
    ai_dir = data_dir / "ai"
    
    # Collect human samples
    if human_dir.exists():
        for audio_file in human_dir.glob("**/*.wav"):
            audio_files.append(audio_file)
            labels.append(0)
        for audio_file in human_dir.glob("**/*.mp3"):
            audio_files.append(audio_file)
            labels.append(0)
    
    # Collect AI samples
    if ai_dir.exists():
        for audio_file in ai_dir.glob("**/*.wav"):
            audio_files.append(audio_file)
            labels.append(1)
        for audio_file in ai_dir.glob("**/*.mp3"):
            audio_files.append(audio_file)
            labels.append(1)
    
    logger.info(f"Found {sum(1 for l in labels if l == 0)} human samples")
    logger.info(f"Found {sum(1 for l in labels if l == 1)} AI samples")
    
    return audio_files, labels


def extract_features_batch(
    audio_files: List[Path],
    transcriber: WhisperTranscriber,
    audio_extractor: AudioFeatureExtractor,
    text_extractor: TextFeatureExtractor
) -> np.ndarray:
    """Extract features from all audio files."""
    
    all_features = []
    failed = []
    
    for audio_path in tqdm(audio_files, desc="Extracting features"):
        try:
            # Transcribe
            transcript = transcriber.transcribe(str(audio_path))
            
            # Extract audio features
            audio_features = audio_extractor.extract(str(audio_path))
            
            # Extract text features
            text_features = text_extractor.extract(transcript.text)
            
            # Combine
            combined = np.concatenate([
                audio_features.to_flat_array(),
                text_features.to_flat_array()
            ])
            
            all_features.append(combined)
            
        except Exception as e:
            logger.warning(f"Failed to process {audio_path}: {e}")
            failed.append(audio_path)
    
    if failed:
        logger.warning(f"Failed to process {len(failed)} files")
    
    return np.vstack(all_features) if all_features else np.array([])


def main():
    parser = argparse.ArgumentParser(description="Train AI voice detection model")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Directory containing training data")
    parser.add_argument("--output-dir", type=str, default="model/saved",
                        help="Directory to save trained model")
    parser.add_argument("--whisper-model", type=str, default="base",
                        help="Whisper model size")
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="Number of LightGBM boosting rounds")
    parser.add_argument("--use-demo-data", action="store_true",
                        help="Use synthetic demo data for testing")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AI Voice Detection - Training Pipeline")
    print("=" * 60)
    
    if args.use_demo_data:
        # Generate synthetic demo data
        print("\nUsing synthetic demo data for testing...")
        
        np.random.seed(42)
        n_samples = 500
        n_audio_features = 80
        n_text_features = 24
        n_features = n_audio_features + n_text_features
        
        # Human samples: higher disfluency, more variation
        X_human = np.random.randn(n_samples // 2, n_features)
        X_human[:, n_audio_features] += 2  # Higher disfluency rate
        X_human[:, n_audio_features + 1] += 1  # More filler words
        
        # AI samples: lower disfluency, more uniform
        X_ai = np.random.randn(n_samples // 2, n_features)
        X_ai[:, n_audio_features] -= 1  # Lower disfluency
        X_ai[:, n_audio_features + 2] += 1.5  # Lower perplexity signal
        
        X = np.vstack([X_human, X_ai])
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
        
        # Shuffle
        indices = np.random.permutation(n_samples)
        X, y = X[indices], y[indices]
        
        # Feature names
        audio_feature_names = [f"audio_{i}" for i in range(n_audio_features)]
        text_feature_names = [f"text_{i}" for i in range(n_text_features)]
        feature_names = audio_feature_names + text_feature_names
        
    else:
        # Real data processing
        data_dir = Path(args.data_dir)
        
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            logger.info("Please create data/raw/human/ and data/raw/ai/ directories with audio files")
            logger.info("Or use --use-demo-data to test with synthetic data")
            sys.exit(1)
        
        # Collect files
        audio_files, labels = collect_audio_files(data_dir)
        
        if len(audio_files) == 0:
            logger.error("No audio files found!")
            sys.exit(1)
        
        # Initialize components
        print("\nInitializing components...")
        transcriber = WhisperTranscriber(model_size=args.whisper_model)
        audio_extractor = AudioFeatureExtractor()
        text_extractor = TextFeatureExtractor()
        
        # Extract features
        print("\nExtracting features...")
        X = extract_features_batch(
            audio_files, transcriber, audio_extractor, text_extractor
        )
        y = np.array(labels[:len(X)])  # Match length with successfully processed files
        
        if len(X) == 0:
            logger.error("No features extracted!")
            sys.exit(1)
        
        # Get feature names
        from feature_extraction.audio_features import AudioFeatures
        from feature_extraction.text_features import TextFeatures
        feature_names = AudioFeatures.get_feature_names() + TextFeatures.get_feature_names()
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: Human={sum(y==0)}, AI={sum(y==1)}")
    
    # Train model
    print("\nTraining LightGBM classifier...")
    classifier = AIVoiceClassifier(
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        num_leaves=31
    )
    
    metrics = classifier.fit(X, y, feature_names=feature_names)
    
    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Cross-validation
    print("\nRunning cross-validation...")
    cv_scores = classifier.cross_validate(X, y, n_folds=5)
    
    print("\nCross-Validation Results:")
    for metric, value in cv_scores.items():
        print(f"  {metric}: {value:.4f}")
    
    # Feature importance
    print("\nTop 15 Important Features:")
    importance = classifier.get_feature_importance(top_k=15)
    for rank, (feature, score) in enumerate(importance.items(), 1):
        print(f"  {rank:2d}. {feature}: {score:.4f}")
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "lgbm_model.pkl"
    classifier.save(str(model_path))
    
    print("\n" + "=" * 60)
    print(f"Model saved to: {model_path}")
    print("=" * 60)
    
    # Save training config
    config = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                   for k, v in metrics.items()},
        "cv_scores": {k: float(v) for k, v in cv_scores.items()},
        "class_distribution": {
            "human": int(sum(y == 0)),
            "ai": int(sum(y == 1))
        }
    }
    
    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
