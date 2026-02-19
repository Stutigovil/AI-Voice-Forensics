# Model Directory

Saved models are stored here after training.

## Expected Files

```
saved/
├── lgbm_model.pkl        # Trained LightGBM classifier
├── lgbm_model_v2.pkl     # Version history
└── training_config.json  # Training configuration
```

## Model Info

The default model (lgbm_model.pkl) contains:
- Trained LightGBM model
- Feature names
- Training metrics
- Model parameters

## Loading a Model

```python
from ai_voice_detection.model import AIVoiceClassifier

classifier = AIVoiceClassifier()
classifier.load("model/saved/lgbm_model.pkl")

print(f"Training accuracy: {classifier.training_metrics['accuracy']:.2%}")
```
