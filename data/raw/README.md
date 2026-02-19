# Raw Data Directory

Place your raw audio files here for training and testing.

## Recommended Structure

```
raw/
├── human/           # Human speech samples
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
└── ai/              # AI-generated speech samples
    ├── tacotron/
    ├── vits/
    ├── bark/
    └── ...
```

## Dataset Sources

1. **ASVspoof 2019/2021** - https://www.asvspoof.org/
2. **LibriSpeech** - https://www.openslr.org/12/
3. **Kaggle** - Search "AI Generated Audio Detection"

## Custom AI Samples

Generate AI audio using:
- Bark: https://github.com/suno-ai/bark
- Coqui TTS: https://github.com/coqui-ai/TTS
- ElevenLabs API
