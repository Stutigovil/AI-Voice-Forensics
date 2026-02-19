"""
Main Inference Pipeline
=======================
Run AI voice detection on audio files.

Usage:
    python run_detection.py audio.wav
    python run_detection.py audio.wav --model model/saved/lgbm_model.pkl
    python run_detection.py --batch audio_dir/
"""

import sys
import argparse
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="AI Voice Detection - Detect AI-generated audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_detection.py sample.wav
  python run_detection.py sample.wav --model model/saved/lgbm_model.pkl
  python run_detection.py --batch audio_folder/
  python run_detection.py sample.wav --json output.json
        """
    )
    
    parser.add_argument("audio", nargs="?", type=str,
                        help="Path to audio file to analyze")
    parser.add_argument("--batch", type=str,
                        help="Directory of audio files for batch processing")
    parser.add_argument("--model", type=str, default="model/saved/lgbm_model.pkl",
                        help="Path to trained model")
    parser.add_argument("--whisper-model", type=str, default="base",
                        help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--json", type=str,
                        help="Output results to JSON file")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed output")
    
    args = parser.parse_args()
    
    if not args.audio and not args.batch:
        parser.print_help()
        print("\nError: Please provide an audio file or use --batch for directory processing")
        sys.exit(1)
    
    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}")
        print("Training a demo model...")
        print("Run: python train_pipeline.py --use-demo-data")
        print()
    
    from model.predict import Predictor
    
    print("=" * 60)
    print("AI Voice Detection System")
    print("=" * 60)
    
    # Initialize predictor
    print("\nInitializing components...")
    try:
        predictor = Predictor(
            model_path=str(model_path) if model_path.exists() else None,
            whisper_model=args.whisper_model
        )
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        sys.exit(1)
    
    results = []
    
    if args.batch:
        # Batch processing
        batch_dir = Path(args.batch)
        if not batch_dir.exists():
            print(f"Error: Directory not found: {batch_dir}")
            sys.exit(1)
        
        audio_files = list(batch_dir.glob("*.wav")) + list(batch_dir.glob("*.mp3"))
        print(f"\nProcessing {len(audio_files)} files...")
        
        for audio_path in audio_files:
            print(f"\nAnalyzing: {audio_path.name}")
            try:
                result = predictor.predict(str(audio_path))
                results.append({
                    "file": str(audio_path),
                    "prediction": result.prediction_label,
                    "confidence": result.confidence,
                    "transcript_preview": result.transcript[:100] + "..."
                })
                print(f"  Result: {result.prediction_label} ({result.confidence:.1%})")
            except Exception as e:
                print(f"  Error: {e}")
                results.append({"file": str(audio_path), "error": str(e)})
    
    else:
        # Single file processing
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)
        
        print(f"\nAnalyzing: {audio_path}")
        
        try:
            result = predictor.predict(str(audio_path))
        except Exception as e:
            print(f"\nError: {e}")
            sys.exit(1)
        
        # Display results
        print("\n" + "=" * 60)
        print("DETECTION RESULT")
        print("=" * 60)
        
        print(f"\n📊 Classification: {result.prediction_label}")
        print(f"📈 Confidence: {result.confidence:.1%}")
        print(f"🎤 AI Probability: {result.probability_ai:.1%}")
        print(f"⏱️  Duration: {result.audio_duration:.1f}s")
        print(f"📝 Word Count: {result.transcript_word_count}")
        
        print("\n" + "-" * 60)
        print("TRANSCRIPT")
        print("-" * 60)
        # Truncate long transcripts
        transcript = result.transcript
        if len(transcript) > 500:
            transcript = transcript[:500] + "..."
        print(transcript)
        
        if result.anomalous_features:
            print("\n" + "-" * 60)
            print("DETECTED ANOMALIES")
            print("-" * 60)
            for anomaly in result.anomalous_features:
                print(f"  ⚠️  {anomaly}")
        
        print("\n" + "-" * 60)
        print("FORENSIC ANALYSIS")
        print("-" * 60)
        print(result.explanation_summary)
        
        if args.verbose and result.top_audio_features:
            print("\n" + "-" * 60)
            print("TOP AUDIO FEATURES")
            print("-" * 60)
            for feature, value in list(result.top_audio_features.items())[:5]:
                if isinstance(value, float):
                    print(f"  {feature}: {value:.4f}")
                else:
                    print(f"  {feature}: {value}")
        
        if args.verbose and result.top_text_features:
            print("\n" + "-" * 60)
            print("TOP TEXT FEATURES")
            print("-" * 60)
            for feature, value in list(result.top_text_features.items())[:5]:
                if isinstance(value, float):
                    print(f"  {feature}: {value:.4f}")
                else:
                    print(f"  {feature}: {value}")
        
        results.append(result.to_dict())
    
    # Save to JSON if requested
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✅ Results saved to: {args.json}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
