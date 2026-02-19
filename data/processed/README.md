# Processed Data Directory

This directory stores extracted features and processed data.

## Contents

After running feature extraction, you'll find:

```
processed/
├── features/
│   ├── audio_features.npy     # Extracted audio features
│   ├── text_features.npy      # Extracted text features
│   └── combined_features.npy  # Combined feature matrix
├── labels.npy                 # Classification labels
├── transcripts/               # Saved transcripts
│   └── *.json
└── metadata.json              # Dataset metadata
```

## Feature Format

- Audio features: 80+ dimensions (MFCC, pitch, spectral, energy)
- Text features: 24 dimensions (perplexity, disfluency, POS, etc.)
- Combined: 100+ dimensional feature vectors
