# 🎙️ AI Voice Forensics

A multimodal AI voice authenticity detection system that determines whether audio contains **human speech** or **AI-generated voice** (from TTS systems like Tacotron, VITS, Bark, ElevenLabs).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🎯 Key Features

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Speech-to-Text** | OpenAI Whisper | Transcribe audio with timestamps |
| **Audio Analysis** | librosa | Extract MFCC, pitch, spectral features |
| **Text Analysis** | GPT-2, spaCy | Perplexity, disfluency detection |
| **Classification** | LightGBM + SHAP | Binary classification with explainability |
| **Explanations** | RAG + LLM | Forensic analysis grounded in research |
| **API** | FastAPI | Production-ready REST endpoint |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      INPUT: Audio File                       │
└─────────────────────────────┬────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                 WHISPER TRANSCRIPTION                        │
│              Audio → Text + Timestamps                       │
└─────────────────────────────┬────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
┌─────────────────────┐           ┌─────────────────────┐
│   AUDIO FEATURES    │           │   TEXT FEATURES     │
│                     │           │                     │
│ • MFCC (mean, var)  │           │ • Perplexity        │
│ • Pitch jitter      │           │ • Disfluency rate   │
│ • Spectral features │           │ • POS entropy       │
│ • Energy variance   │           │ • Repetition score  │
└──────────┬──────────┘           └──────────┬──────────┘
           │                                  │
           └────────────────┬─────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              LIGHTGBM CLASSIFIER + SHAP                      │
│                 Human (0) vs AI (1)                          │
└─────────────────────────────┬────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                       RAG MODULE                             │
│       Vector DB (FAISS) + LLM Forensic Analysis              │
└─────────────────────────────┬────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  OUTPUT: Prediction + Confidence + Transcript + Explanation  │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
ai_voice_detection/
├── stt/
│   └── transcribe.py          # Whisper STT
├── feature_extraction/
│   ├── audio_features.py      # MFCC, pitch, spectral
│   └── text_features.py       # Perplexity, disfluency
├── model/
│   ├── train_lgbm.py          # Training + SHAP
│   └── predict.py             # Inference pipeline
├── rag/
│   ├── build_index.py         # FAISS vector index
│   ├── retriever.py           # Document retrieval
│   └── llm_explainer.py       # LLM explanations
├── api/
│   └── app.py                 # FastAPI server
├── utils/
│   ├── audio_utils.py         # Audio processing
│   └── text_utils.py          # Text processing
├── config.yaml                # Configuration
├── requirements.txt           # Dependencies
├── train_pipeline.py          # Training script
└── run_detection.py           # CLI inference
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Stutigovil/AI-Voice-Forensics.git
cd AI-Voice-Forensics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLP models
python -m spacy download en_core_web_sm
```

### 2. Train Model

```bash
# Quick test with demo data
python train_pipeline.py --use-demo-data

# Train with your own data (place in data/raw/human/ and data/raw/ai/)
python train_pipeline.py --data-dir data/raw
```

### 3. Run Detection

```bash
# Analyze single file
python run_detection.py audio.wav

# Batch processing
python run_detection.py --batch audio_folder/

# Export results
python run_detection.py audio.wav --json results.json
```

### 4. Start API Server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🔌 API Usage

### Endpoint: `POST /analyze-audio`

```bash
curl -X POST "http://localhost:8000/analyze-audio" \
  -F "audio_file=@sample.wav"
```

### Response

```json
{
  "prediction": "AI-Generated",
  "confidence": 0.87,
  "probability_ai": 0.87,
  "transcript": "Hello, this is a sample audio...",
  "duration_seconds": 5.2,
  "anomalies": [
    "Very low disfluency rate",
    "Absence of filler words"
  ],
  "explanation": "Classification: AI-Generated (87% confidence)...",
  "model_attribution": "Likely VITS-family or ElevenLabs",
  "recommendations": ["Consider manual review"]
}
```

---

## 🔬 Detection Signals

### AI-Generated Indicators
| Signal | Description |
|--------|-------------|
| Low disfluency rate | < 1% (humans: 3-7%) |
| Low perplexity | < 30 (predictable LLM text) |
| Low pitch jitter | < 0.015 (unnaturally smooth) |
| No filler words | Missing "uh", "um", "like" |
| Uniform sentences | Low variance in length |

### Human Speech Indicators
| Signal | Description |
|--------|-------------|
| Natural disfluencies | Fillers, false starts |
| Pitch micro-variations | Jitter 0.02-0.05 |
| Breath sounds | Natural pauses |
| Variable prosody | Emotional expression |

---

## 📊 Datasets

### Recommended for Training

| Dataset | Description |
|---------|-------------|
| [ASVspoof 2019/2021](https://www.asvspoof.org/) | Official deepfake challenge |
| [LibriSpeech](https://www.openslr.org/12/) | Human speech baseline |
| Custom TTS samples | Generate with Bark, Coqui, ElevenLabs |

---

## ⚙️ Configuration

Edit `config.yaml`:

```yaml
whisper:
  model_size: "base"  # tiny, base, small, medium, large
  device: "cpu"       # or "cuda"

model:
  lgbm:
    n_estimators: 200
    learning_rate: 0.05

llm:
  provider: "local"   # openai, gemini, or local
```

---

## 🧠 What Makes This Project Unique

1. **Multimodal Analysis** - Combines audio AND text features (not just audio)
2. **Disfluency Detection** - Catches AI's lack of natural speech fillers
3. **Feature Mismatch** - Identifies human text + AI prosody patterns
4. **Forensic Explanations** - RAG-grounded analysis, not just predictions
5. **Model Attribution** - Suggests which TTS system was likely used

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.

---



**Skills demonstrated:** ML, NLP, Speech Processing, LLMs, RAG, API Development
