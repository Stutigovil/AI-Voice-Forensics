"""
Audio Feature Extraction Module
===============================
Extracts acoustic features from audio for AI voice detection.

Features extracted:
- MFCC (Mel-Frequency Cepstral Coefficients)
- Spectral features (centroid, bandwidth, rolloff)
- Zero crossing rate
- Pitch/F0 characteristics
- Energy/amplitude variance
- Formant-related features

These features help distinguish between human speech and 
AI-generated audio (TTS systems like Tacotron, VITS, Bark).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import librosa
import scipy.stats as stats
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """Container for extracted audio features."""
    # MFCC features
    mfcc_mean: List[float]
    mfcc_var: List[float]
    mfcc_delta_mean: List[float]
    mfcc_delta_var: List[float]
    
    # Spectral features
    spectral_centroid_mean: float
    spectral_centroid_var: float
    spectral_bandwidth_mean: float
    spectral_bandwidth_var: float
    spectral_rolloff_mean: float
    spectral_rolloff_var: float
    spectral_flatness_mean: float
    spectral_contrast_mean: List[float]
    
    # Rhythm/temporal features
    zero_crossing_rate_mean: float
    zero_crossing_rate_var: float
    tempo: float
    
    # Pitch features
    pitch_mean: float
    pitch_var: float
    pitch_range: float
    pitch_jitter: float
    
    # Energy features
    rms_mean: float
    rms_var: float
    energy_entropy: float
    
    # Statistical features
    skewness: float
    kurtosis: float
    
    # Duration
    duration: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_flat_array(self) -> np.ndarray:
        """Convert features to flat numpy array for model input."""
        flat = []
        for key, value in self.to_dict().items():
            if isinstance(value, list):
                flat.extend(value)
            else:
                flat.append(value)
        return np.array(flat, dtype=np.float32)
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get names of all features for model interpretation."""
        names = []
        # MFCC names (13 coefficients × 4 stats)
        for stat in ['mean', 'var', 'delta_mean', 'delta_var']:
            for i in range(13):
                names.append(f'mfcc_{i}_{stat}')
        
        # Spectral feature names
        names.extend([
            'spectral_centroid_mean', 'spectral_centroid_var',
            'spectral_bandwidth_mean', 'spectral_bandwidth_var',
            'spectral_rolloff_mean', 'spectral_rolloff_var',
            'spectral_flatness_mean'
        ])
        
        # Spectral contrast (7 bands)
        for i in range(7):
            names.append(f'spectral_contrast_{i}')
        
        # Other features
        names.extend([
            'zero_crossing_rate_mean', 'zero_crossing_rate_var',
            'tempo', 'pitch_mean', 'pitch_var', 'pitch_range', 'pitch_jitter',
            'rms_mean', 'rms_var', 'energy_entropy',
            'skewness', 'kurtosis', 'duration'
        ])
        
        return names


class AudioFeatureExtractor:
    """
    Extracts acoustic features from audio files for AI detection.
    
    This extractor focuses on features that distinguish human speech
    from AI-generated audio, including:
    - Prosodic naturalness (pitch variation, rhythm)
    - Spectral artifacts (from vocoder processing)
    - Energy patterns (breath sounds, natural variation)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        frame_length: int = 2048
    ):
        """
        Initialize the feature extractor.
        
        Args:
            sample_rate: Target sample rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
            frame_length: Frame length for analysis
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frame_length = frame_length
    
    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio file with consistent preprocessing."""
        y, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        
        # Normalize
        y = y / (np.abs(y).max() + 1e-8)
        
        return y, sr
    
    def extract_mfcc_features(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract MFCC features and deltas.
        
        MFCCs capture the spectral envelope of speech, which can reveal
        artifacts from neural vocoders used in TTS systems.
        """
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Delta coefficients (temporal dynamics)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        return {
            'mfcc_mean': np.mean(mfcc, axis=1),
            'mfcc_var': np.var(mfcc, axis=1),
            'mfcc_delta_mean': np.mean(mfcc_delta, axis=1),
            'mfcc_delta_var': np.var(mfcc_delta, axis=1),
            'mfcc_delta2_mean': np.mean(mfcc_delta2, axis=1),
            'mfcc_delta2_var': np.var(mfcc_delta2, axis=1)
        }
    
    def extract_spectral_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract spectral features.
        
        AI-generated audio often has distinctive spectral characteristics
        due to mel-spectrogram reconstruction artifacts.
        """
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Spectral flatness (tonality)
        spectral_flatness = librosa.feature.spectral_flatness(
            y=y, hop_length=self.hop_length
        )[0]
        
        # Spectral contrast (energy distribution across bands)
        spectral_contrast = librosa.feature.spectral_contrast(
            y=y, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),
            'spectral_centroid_var': float(np.var(spectral_centroid)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_bandwidth_var': float(np.var(spectral_bandwidth)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_rolloff_var': float(np.var(spectral_rolloff)),
            'spectral_flatness_mean': float(np.mean(spectral_flatness)),
            'spectral_contrast_mean': np.mean(spectral_contrast, axis=1).tolist()
        }
    
    def extract_pitch_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract pitch/F0 features.
        
        Human speech has natural pitch variations and micro-fluctuations
        (jitter) that TTS systems often struggle to replicate.
        """
        # Extract pitch using pYIN algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Filter out unvoiced frames
        f0_voiced = f0[~np.isnan(f0)]
        
        if len(f0_voiced) == 0:
            return {
                'pitch_mean': 0.0,
                'pitch_var': 0.0,
                'pitch_range': 0.0,
                'pitch_jitter': 0.0
            }
        
        # Calculate jitter (pitch perturbation)
        # Jitter is the cycle-to-cycle variation in pitch period
        pitch_diff = np.abs(np.diff(f0_voiced))
        jitter = np.mean(pitch_diff) / np.mean(f0_voiced) if np.mean(f0_voiced) > 0 else 0
        
        return {
            'pitch_mean': float(np.mean(f0_voiced)),
            'pitch_var': float(np.var(f0_voiced)),
            'pitch_range': float(np.ptp(f0_voiced)),  # Peak-to-peak
            'pitch_jitter': float(jitter)
        }
    
    def extract_energy_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract energy/amplitude features.
        
        Natural speech has characteristic energy patterns including
        breath pauses and dynamic range that can be altered in TTS.
        """
        # RMS energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        # Energy entropy (measure of energy distribution randomness)
        # Higher entropy = more random/natural energy distribution
        energy_hist, _ = np.histogram(rms, bins=50, density=True)
        energy_hist = energy_hist + 1e-10  # Avoid log(0)
        energy_entropy = -np.sum(energy_hist * np.log2(energy_hist))
        
        return {
            'rms_mean': float(np.mean(rms)),
            'rms_var': float(np.var(rms)),
            'energy_entropy': float(energy_entropy)
        }
    
    def extract_temporal_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal/rhythm features.
        
        Zero-crossing rate and tempo can reveal unnatural timing
        patterns in synthesized speech.
        """
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=self.frame_length, hop_length=self.hop_length
        )[0]
        
        # Tempo estimation
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sample_rate)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sample_rate)[0]
        
        return {
            'zero_crossing_rate_mean': float(np.mean(zcr)),
            'zero_crossing_rate_var': float(np.var(zcr)),
            'tempo': float(tempo)
        }
    
    def extract_statistical_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features from the raw waveform.
        
        Higher-order statistics can capture non-Gaussian characteristics
        that differ between human and synthetic speech.
        """
        return {
            'skewness': float(stats.skew(y)),
            'kurtosis': float(stats.kurtosis(y))
        }
    
    def extract(self, audio_path: Union[str, Path]) -> AudioFeatures:
        """
        Extract all features from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            AudioFeatures object with all extracted features
        """
        logger.info(f"Extracting features from: {audio_path}")
        
        # Load audio
        y, sr = self.load_audio(audio_path)
        duration = len(y) / sr
        
        # Extract all feature groups
        mfcc_features = self.extract_mfcc_features(y)
        spectral_features = self.extract_spectral_features(y)
        pitch_features = self.extract_pitch_features(y)
        energy_features = self.extract_energy_features(y)
        temporal_features = self.extract_temporal_features(y)
        statistical_features = self.extract_statistical_features(y)
        
        # Combine into AudioFeatures object
        features = AudioFeatures(
            mfcc_mean=mfcc_features['mfcc_mean'].tolist(),
            mfcc_var=mfcc_features['mfcc_var'].tolist(),
            mfcc_delta_mean=mfcc_features['mfcc_delta_mean'].tolist(),
            mfcc_delta_var=mfcc_features['mfcc_delta_var'].tolist(),
            spectral_centroid_mean=spectral_features['spectral_centroid_mean'],
            spectral_centroid_var=spectral_features['spectral_centroid_var'],
            spectral_bandwidth_mean=spectral_features['spectral_bandwidth_mean'],
            spectral_bandwidth_var=spectral_features['spectral_bandwidth_var'],
            spectral_rolloff_mean=spectral_features['spectral_rolloff_mean'],
            spectral_rolloff_var=spectral_features['spectral_rolloff_var'],
            spectral_flatness_mean=spectral_features['spectral_flatness_mean'],
            spectral_contrast_mean=spectral_features['spectral_contrast_mean'],
            zero_crossing_rate_mean=temporal_features['zero_crossing_rate_mean'],
            zero_crossing_rate_var=temporal_features['zero_crossing_rate_var'],
            tempo=temporal_features['tempo'],
            pitch_mean=pitch_features['pitch_mean'],
            pitch_var=pitch_features['pitch_var'],
            pitch_range=pitch_features['pitch_range'],
            pitch_jitter=pitch_features['pitch_jitter'],
            rms_mean=energy_features['rms_mean'],
            rms_var=energy_features['rms_var'],
            energy_entropy=energy_features['energy_entropy'],
            skewness=statistical_features['skewness'],
            kurtosis=statistical_features['kurtosis'],
            duration=duration
        )
        
        logger.info(f"Extracted {len(features.to_flat_array())} features")
        return features
    
    def extract_batch(
        self,
        audio_paths: List[Union[str, Path]],
        return_array: bool = True
    ) -> Union[List[AudioFeatures], np.ndarray]:
        """
        Extract features from multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            return_array: If True, return as numpy array
            
        Returns:
            List of AudioFeatures or numpy array
        """
        features_list = []
        
        for path in audio_paths:
            try:
                features = self.extract(path)
                features_list.append(features)
            except Exception as e:
                logger.error(f"Failed to extract features from {path}: {e}")
                continue
        
        if return_array:
            return np.vstack([f.to_flat_array() for f in features_list])
        
        return features_list


def extract_audio_features(
    audio_path: str,
    sample_rate: int = 16000
) -> Dict:
    """
    Convenience function to extract audio features.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Dictionary of extracted features
    """
    extractor = AudioFeatureExtractor(sample_rate=sample_rate)
    features = extractor.extract(audio_path)
    return features.to_dict()


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python audio_features.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    extractor = AudioFeatureExtractor()
    features = extractor.extract(audio_file)
    
    print("\nExtracted Audio Features:")
    print("=" * 50)
    print(json.dumps(features.to_dict(), indent=2))
