"""
FastAPI Application Module
==========================
REST API for AI voice detection service.

Endpoints:
- POST /analyze-audio: Analyze audio file for AI detection
- GET /health: Health check
- GET /model-info: Get model information
"""

import os
import sys
import uuid
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

import yaml
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stt.transcribe import WhisperTranscriber
from feature_extraction.audio_features import AudioFeatureExtractor
from feature_extraction.text_features import TextFeatureExtractor
from model.train_lgbm import AIVoiceClassifier
from rag.llm_explainer import LLMExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Response Models
# ============================================================================

class AnalysisResult(BaseModel):
    """Response model for audio analysis."""
    request_id: str = Field(..., description="Unique request identifier")
    prediction: str = Field(..., description="Classification result (Human/AI-Generated)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probability_ai: float = Field(..., ge=0, le=1, description="Probability of AI-generated")
    transcript: str = Field(..., description="Transcribed text from audio")
    duration_seconds: float = Field(..., description="Audio duration in seconds")
    word_count: int = Field(..., description="Number of words in transcript")
    anomalies: List[str] = Field(default=[], description="Detected anomalies")
    explanation: str = Field(..., description="Forensic explanation")
    model_attribution: str = Field(default="", description="Potential TTS model attribution")
    recommendations: List[str] = Field(default=[], description="Recommendations")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    whisper_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_type: str
    n_features: int
    training_accuracy: Optional[float]
    feature_importance: Dict[str, float]


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str
    detail: str
    request_id: Optional[str] = None


# ============================================================================
# Application State
# ============================================================================

class AppState:
    """Global application state."""
    
    def __init__(self):
        self.transcriber: Optional[WhisperTranscriber] = None
        self.audio_extractor: Optional[AudioFeatureExtractor] = None
        self.text_extractor: Optional[TextFeatureExtractor] = None
        self.classifier: Optional[AIVoiceClassifier] = None
        self.explainer: Optional[LLMExplainer] = None
        self.config: Dict = {}
        self.is_initialized: bool = False
    
    def load_config(self, config_path: str = "config.yaml") -> Dict:
        """Load configuration from YAML file."""
        config_file = Path(__file__).parent.parent / config_path
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'whisper': {'model_size': 'base', 'device': 'cpu'},
                'model': {'model_path': 'model/saved/lgbm_model.pkl'},
                'api': {'max_file_size': 52428800}
            }
        
        return self.config
    
    def initialize(self):
        """Initialize all components."""
        if self.is_initialized:
            return
        
        logger.info("Initializing application components...")
        
        # Load config
        self.load_config()
        
        # Initialize Whisper transcriber
        whisper_config = self.config.get('whisper', {})
        try:
            self.transcriber = WhisperTranscriber(
                model_size=whisper_config.get('model_size', 'base'),
                device=whisper_config.get('device', 'cpu')
            )
            logger.info("Whisper transcriber initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {e}")
        
        # Initialize feature extractors
        self.audio_extractor = AudioFeatureExtractor()
        self.text_extractor = TextFeatureExtractor(use_transformers=True)
        logger.info("Feature extractors initialized")
        
        # Initialize classifier
        model_path = Path(__file__).parent.parent / self.config.get('model', {}).get('model_path', 'model/saved/lgbm_model.pkl')
        self.classifier = AIVoiceClassifier()
        
        if model_path.exists():
            try:
                self.classifier.load(str(model_path))
                logger.info(f"Loaded classifier from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Model not available.")
        else:
            logger.warning(f"Model not found at {model_path}. Train a model first.")
        
        # Initialize LLM explainer
        llm_config = self.config.get('llm', {})
        self.explainer = LLMExplainer(
            provider=llm_config.get('provider', 'local'),
            model=llm_config.get('model', 'gpt-3.5-turbo')
        )
        logger.info("LLM explainer initialized")
        
        self.is_initialized = True
        logger.info("Application initialization complete")


# Global state
app_state = AppState()


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting AI Voice Detection API...")
    app_state.initialize()
    yield
    # Shutdown
    logger.info("Shutting down AI Voice Detection API...")


# ============================================================================
# FastAPI Application
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    application = FastAPI(
        title="AI Voice Detection API",
        description="""
        Multimodal AI voice authenticity detection service.
        
        Combines:
        - Speech-to-Text (Whisper)
        - Audio feature analysis (MFCC, pitch, energy)
        - Text feature analysis (perplexity, disfluency)
        - LightGBM classification
        - RAG-powered forensic explanations
        """,
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return application


app = create_app()


# ============================================================================
# Helper Functions
# ============================================================================

def cleanup_temp_file(filepath: str):
    """Clean up temporary file."""
    try:
        Path(filepath).unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file: {e}")


async def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file to temporary location."""
    suffix = Path(upload_file.filename).suffix or ".wav"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await upload_file.read()
        tmp.write(content)
        return tmp.name


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect."""
    return {"message": "AI Voice Detection API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and component availability.
    """
    return HealthResponse(
        status="healthy" if app_state.is_initialized else "initializing",
        model_loaded=app_state.classifier is not None and app_state.classifier.is_fitted,
        whisper_loaded=app_state.transcriber is not None,
        version="1.0.0"
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def get_model_info():
    """
    Get information about the loaded model.
    
    Returns model type, features, and performance metrics.
    """
    if not app_state.classifier or not app_state.classifier.is_fitted:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_type="LightGBM Classifier",
        n_features=app_state.classifier.training_metrics.get('n_features', 0),
        training_accuracy=app_state.classifier.training_metrics.get('accuracy'),
        feature_importance=app_state.classifier.get_feature_importance(top_k=10)
    )


@app.post(
    "/analyze-audio",
    response_model=AnalysisResult,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Processing error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"}
    },
    tags=["Analysis"]
)
async def analyze_audio(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file to analyze")
):
    """
    Analyze an audio file for AI generation detection.
    
    This endpoint performs:
    1. Audio transcription using Whisper
    2. Audio feature extraction (MFCC, pitch, energy, etc.)
    3. Text feature extraction (perplexity, disfluency, etc.)
    4. Binary classification (Human vs AI-Generated)
    5. RAG-powered forensic explanation
    
    **Supported formats**: WAV, MP3, FLAC, M4A, OGG
    
    **Max file size**: 50MB
    """
    request_id = str(uuid.uuid4())[:8]
    
    # Validate initialization
    if not app_state.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service is initializing. Please try again shortly."
        )
    
    # Validate model
    if not app_state.classifier or not app_state.classifier.is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Classification model not loaded. Please contact administrator."
        )
    
    # Validate file
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    allowed_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus', '.webm']
    file_ext = Path(audio_file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save to temp file
    try:
        temp_path = await save_upload_file(audio_file)
        background_tasks.add_task(cleanup_temp_file, temp_path)
    except Exception as e:
        logger.error(f"[{request_id}] Failed to save upload: {e}")
        raise HTTPException(status_code=500, detail="Failed to process uploaded file")
    
    try:
        logger.info(f"[{request_id}] Processing audio: {audio_file.filename}")
        
        # Step 1: Transcribe
        logger.info(f"[{request_id}] Transcribing audio...")
        transcription = app_state.transcriber.transcribe(temp_path)
        transcript = transcription.text
        duration = transcription.duration
        
        if not transcript.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from audio. File may be silent or corrupted."
            )
        
        # Step 2: Extract audio features
        logger.info(f"[{request_id}] Extracting audio features...")
        audio_features = app_state.audio_extractor.extract(temp_path)
        
        # Step 3: Extract text features
        logger.info(f"[{request_id}] Extracting text features...")
        text_features = app_state.text_extractor.extract(transcript)
        
        # Step 4: Combine features and predict
        combined_features = np.concatenate([
            audio_features.to_flat_array(),
            text_features.to_flat_array()
        ]).reshape(1, -1)
        
        prediction = app_state.classifier.predict(combined_features)[0]
        probability = app_state.classifier.predict_proba(combined_features)[0]
        confidence = probability if prediction == 1 else (1 - probability)
        
        # Step 5: Detect anomalies
        anomalies = detect_anomalies(audio_features.to_dict(), text_features.to_dict())
        
        # Step 6: Generate explanation
        logger.info(f"[{request_id}] Generating explanation...")
        explanation = app_state.explainer.generate_explanation(
            prediction=prediction,
            confidence=confidence,
            transcript=transcript[:500],
            anomalous_features=anomalies,
            audio_features=audio_features.to_dict(),
            text_features=text_features.to_dict()
        )
        
        logger.info(f"[{request_id}] Analysis complete: {'AI' if prediction == 1 else 'Human'} ({confidence:.1%})")
        
        return AnalysisResult(
            request_id=request_id,
            prediction="AI-Generated" if prediction == 1 else "Human",
            confidence=float(confidence),
            probability_ai=float(probability),
            transcript=transcript,
            duration_seconds=float(duration),
            word_count=len(transcript.split()),
            anomalies=anomalies,
            explanation=explanation.summary + "\n\n" + explanation.prediction_analysis,
            model_attribution=explanation.potential_tts_model,
            recommendations=explanation.recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process audio: {str(e)}"
        )


def detect_anomalies(audio_features: Dict, text_features: Dict) -> List[str]:
    """Detect anomalies in features that may indicate AI generation."""
    anomalies = []
    
    # Text-based anomalies
    if text_features.get('disfluency_rate', 10) < 0.5:
        anomalies.append("Very low disfluency rate (no natural speech fillers)")
    
    if text_features.get('filler_count', 10) == 0:
        anomalies.append("Complete absence of filler words (uh, um)")
    
    if text_features.get('perplexity', 100) < 30:
        anomalies.append("Unusually low text perplexity (highly predictable text)")
    
    if text_features.get('sentence_length_var', 10) < 2:
        anomalies.append("Very uniform sentence lengths")
    
    # Audio-based anomalies
    if audio_features.get('pitch_jitter', 1) < 0.015:
        anomalies.append("Abnormally low pitch jitter (unnaturally smooth)")
    
    if audio_features.get('pitch_var', 1000) < 30:
        anomalies.append("Very low pitch variation (monotone delivery)")
    
    if audio_features.get('spectral_flatness_mean', 0) > 0.5:
        anomalies.append("High spectral flatness (possible vocoder artifacts)")
    
    return anomalies


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Get config
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        api_config = config.get('api', {})
    else:
        api_config = {}
    
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    
    print(f"Starting AI Voice Detection API at http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
