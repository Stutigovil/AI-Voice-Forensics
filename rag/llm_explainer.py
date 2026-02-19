"""
LLM Explainer Module
====================
Generates forensic explanations using LLM with RAG context.

Combines retrieved knowledge base documents with prediction details
to generate human-readable forensic analysis of AI-generated voice.
"""

import os
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from .retriever import DocumentRetriever, Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ForensicExplanation:
    """Container for forensic explanation results."""
    summary: str
    prediction_analysis: str
    evidence_points: List[str]
    potential_tts_model: str
    confidence_assessment: str
    recommendations: List[str]
    retrieved_context: str
    
    def to_dict(self) -> Dict:
        return {
            'summary': self.summary,
            'prediction_analysis': self.prediction_analysis,
            'evidence_points': self.evidence_points,
            'potential_tts_model': self.potential_tts_model,
            'confidence_assessment': self.confidence_assessment,
            'recommendations': self.recommendations,
            'retrieved_context': self.retrieved_context
        }
    
    def to_markdown(self) -> str:
        """Format as markdown for display."""
        md_parts = [
            "# AI Voice Detection - Forensic Analysis",
            "",
            "## Summary",
            self.summary,
            "",
            "## Analysis",
            self.prediction_analysis,
            "",
            "## Evidence Points",
        ]
        
        for point in self.evidence_points:
            md_parts.append(f"- {point}")
        
        md_parts.extend([
            "",
            "## Model Attribution",
            self.potential_tts_model,
            "",
            "## Confidence Assessment",
            self.confidence_assessment,
            "",
            "## Recommendations",
        ])
        
        for rec in self.recommendations:
            md_parts.append(f"- {rec}")
        
        return "\n".join(md_parts)


class LLMExplainer:
    """
    Generates forensic explanations using LLM APIs.
    
    Supports:
    - OpenAI (GPT-3.5, GPT-4)
    - Google Gemini
    - Local/fallback rule-based explanations
    """
    
    SYSTEM_PROMPT = """You are an expert forensic analyst specializing in AI-generated audio detection. 
Your task is to provide clear, evidence-based explanations for voice authenticity classifications.

When analyzing potential AI-generated audio, consider:
1. Acoustic features (pitch variation, spectral characteristics, prosody)
2. Linguistic patterns (disfluencies, perplexity, sentence structure)
3. Known TTS system artifacts (Tacotron, VITS, Bark, etc.)
4. Comparison with established research findings

Provide explanations that are:
- Factual and evidence-based
- Accessible to non-technical users
- Clear about confidence levels
- Actionable with recommendations
"""
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        index_path: str = "rag/index"
    ):
        """
        Initialize the LLM explainer.
        
        Args:
            provider: LLM provider ('openai', 'gemini', or 'local')
            model: Model name
            api_key: API key (or set via environment variable)
            temperature: Generation temperature
            index_path: Path to RAG index
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        
        # Initialize retriever
        self.retriever = DocumentRetriever(index_path=index_path)
        
        # Initialize LLM client
        self.client = None
        self.api_available = False
        
        if provider == "openai":
            self._init_openai(api_key)
        elif provider == "gemini":
            self._init_gemini(api_key)
        else:
            logger.info("Using local rule-based explanations (no LLM API)")
    
    def _init_openai(self, api_key: Optional[str]):
        """Initialize OpenAI client."""
        try:
            import openai
            
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if key:
                self.client = openai.OpenAI(api_key=key)
                self.api_available = True
                logger.info("OpenAI client initialized")
            else:
                logger.warning("No OpenAI API key found. Using local explanations.")
                
        except ImportError:
            logger.warning("OpenAI package not installed. Using local explanations.")
    
    def _init_gemini(self, api_key: Optional[str]):
        """Initialize Google Gemini client."""
        try:
            import google.generativeai as genai
            
            key = api_key or os.environ.get("GOOGLE_API_KEY")
            if key:
                genai.configure(api_key=key)
                self.client = genai.GenerativeModel(self.model)
                self.api_available = True
                logger.info("Gemini client initialized")
            else:
                logger.warning("No Google API key found. Using local explanations.")
                
        except ImportError:
            logger.warning("Google GenAI package not installed. Using local explanations.")
    
    def generate_explanation(
        self,
        prediction: int,
        confidence: float,
        transcript: str,
        anomalous_features: List[str],
        audio_features: Optional[Dict] = None,
        text_features: Optional[Dict] = None,
        shap_values: Optional[Dict] = None
    ) -> ForensicExplanation:
        """
        Generate a forensic explanation for the prediction.
        
        Args:
            prediction: Model prediction (0=Human, 1=AI)
            confidence: Prediction confidence
            transcript: Audio transcript
            anomalous_features: List of detected anomalies
            audio_features: Audio feature dictionary
            text_features: Text feature dictionary
            shap_values: SHAP explanation values
            
        Returns:
            ForensicExplanation object
        """
        # Retrieve relevant context
        retrieved_docs = self.retriever.retrieve(
            prediction=prediction,
            anomalous_features=anomalous_features,
            audio_features=audio_features,
            text_features=text_features
        )
        
        context = self.retriever.format_context(retrieved_docs)
        
        if self.api_available:
            return self._generate_llm_explanation(
                prediction, confidence, transcript, anomalous_features,
                audio_features, text_features, shap_values, context
            )
        else:
            return self._generate_local_explanation(
                prediction, confidence, transcript, anomalous_features,
                audio_features, text_features, context
            )
    
    def _generate_llm_explanation(
        self,
        prediction: int,
        confidence: float,
        transcript: str,
        anomalous_features: List[str],
        audio_features: Optional[Dict],
        text_features: Optional[Dict],
        shap_values: Optional[Dict],
        context: str
    ) -> ForensicExplanation:
        """Generate explanation using LLM API."""
        
        # Build prompt
        prediction_label = "AI-Generated" if prediction == 1 else "Human"
        
        user_prompt = f"""Analyze the following voice authenticity detection result:

**Classification**: {prediction_label}
**Confidence**: {confidence:.1%}

**Transcript Preview** (first 300 chars):
{transcript[:300]}...

**Detected Anomalies**:
{chr(10).join(f'- {a}' for a in anomalous_features) if anomalous_features else '- None detected'}

**Key Audio Features**:
- Pitch jitter: {audio_features.get('pitch_jitter', 'N/A') if audio_features else 'N/A'}
- Pitch variance: {audio_features.get('pitch_var', 'N/A') if audio_features else 'N/A'}
- Spectral flatness: {audio_features.get('spectral_flatness_mean', 'N/A') if audio_features else 'N/A'}

**Key Text Features**:
- Disfluency rate: {text_features.get('disfluency_rate', 'N/A') if text_features else 'N/A'}
- Perplexity: {text_features.get('perplexity', 'N/A') if text_features else 'N/A'}
- Filler count: {text_features.get('filler_count', 'N/A') if text_features else 'N/A'}

**Reference Knowledge Base**:
{context}

Based on this information, provide:
1. A brief summary (2-3 sentences)
2. Detailed analysis of the prediction
3. List of specific evidence points
4. Potential TTS model attribution (if AI-generated)
5. Confidence assessment
6. Recommendations for further verification

Format your response as JSON with the following keys:
- summary
- analysis
- evidence_points (list)
- potential_model
- confidence_assessment
- recommendations (list)
"""
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=1000
                )
                content = response.choices[0].message.content
                
            elif self.provider == "gemini":
                response = self.client.generate_content(
                    f"{self.SYSTEM_PROMPT}\n\n{user_prompt}"
                )
                content = response.text
            
            # Parse response
            return self._parse_llm_response(content, context)
            
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return self._generate_local_explanation(
                prediction, confidence, transcript, anomalous_features,
                audio_features, text_features, context
            )
    
    def _parse_llm_response(self, content: str, context: str) -> ForensicExplanation:
        """Parse LLM response into ForensicExplanation."""
        import json
        import re
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                
                return ForensicExplanation(
                    summary=data.get('summary', 'Analysis complete.'),
                    prediction_analysis=data.get('analysis', content),
                    evidence_points=data.get('evidence_points', []),
                    potential_tts_model=data.get('potential_model', 'Unknown'),
                    confidence_assessment=data.get('confidence_assessment', 'Moderate confidence'),
                    recommendations=data.get('recommendations', []),
                    retrieved_context=context[:500]
                )
        except json.JSONDecodeError:
            pass
        
        # Fallback: use raw content
        return ForensicExplanation(
            summary="See detailed analysis below.",
            prediction_analysis=content,
            evidence_points=[],
            potential_tts_model="See analysis",
            confidence_assessment="See analysis",
            recommendations=[],
            retrieved_context=context[:500]
        )
    
    def _generate_local_explanation(
        self,
        prediction: int,
        confidence: float,
        transcript: str,
        anomalous_features: List[str],
        audio_features: Optional[Dict],
        text_features: Optional[Dict],
        context: str
    ) -> ForensicExplanation:
        """Generate explanation using rule-based logic (no LLM)."""
        
        prediction_label = "AI-Generated" if prediction == 1 else "Human"
        
        # Build summary
        if prediction == 1:
            summary = f"The audio sample has been classified as AI-generated with {confidence:.1%} confidence. "
            summary += "Multiple indicators suggest synthetic speech generation."
        else:
            summary = f"The audio sample has been classified as human speech with {confidence:.1%} confidence. "
            summary += "Natural speech patterns were detected."
        
        # Build analysis
        analysis_parts = [f"Classification: {prediction_label} ({confidence:.1%} confidence)"]
        
        if prediction == 1:
            analysis_parts.append("\nThe analysis identified several indicators consistent with AI-generated audio:")
            
            if audio_features:
                if audio_features.get('pitch_jitter', 1) < 0.02:
                    analysis_parts.append("- Abnormally low pitch jitter suggests synthesized speech")
                if audio_features.get('pitch_var', 1000) < 50:
                    analysis_parts.append("- Limited pitch variation indicates monotonic TTS output")
            
            if text_features:
                if text_features.get('disfluency_rate', 10) < 1:
                    analysis_parts.append("- Near-absence of speech disfluencies (fillers)")
                if text_features.get('perplexity', 100) < 30:
                    analysis_parts.append("- Low text perplexity suggests LLM-generated script")
        else:
            analysis_parts.append("\nNatural speech characteristics detected:")
            
            if text_features:
                if text_features.get('filler_count', 0) > 0:
                    analysis_parts.append(f"- Natural filler words present ({text_features.get('filler_count')} detected)")
                if text_features.get('disfluency_rate', 0) > 2:
                    analysis_parts.append(f"- Normal disfluency rate ({text_features.get('disfluency_rate'):.1f}%)")
            
            if audio_features:
                if audio_features.get('pitch_jitter', 0) > 0.02:
                    analysis_parts.append("- Natural pitch micro-variations detected")
        
        analysis = "\n".join(analysis_parts)
        
        # Evidence points
        evidence = anomalous_features.copy() if anomalous_features else []
        
        if prediction == 1 and not evidence:
            evidence = [
                "Feature pattern matches AI-generated training samples",
                "Prosodic characteristics consistent with neural TTS"
            ]
        elif prediction == 0 and not evidence:
            evidence = [
                "Natural speech rhythm and variation detected",
                "Authentic prosodic patterns observed"
            ]
        
        # Model attribution
        if prediction == 1:
            if audio_features and audio_features.get('pitch_jitter', 1) < 0.015:
                potential_model = "Likely VITS-family or ElevenLabs (very smooth pitch)"
            elif text_features and text_features.get('disfluency_rate', 10) < 0.5:
                potential_model = "Potentially Tacotron2 or similar seq2seq model"
            else:
                potential_model = "Neural TTS system (specific model undetermined)"
        else:
            potential_model = "N/A - Classified as human speech"
        
        # Confidence assessment
        if confidence > 0.9:
            conf_assessment = "High confidence - Strong indicators support the classification"
        elif confidence > 0.7:
            conf_assessment = "Moderate-high confidence - Multiple indicators align"
        elif confidence > 0.5:
            conf_assessment = "Moderate confidence - Some ambiguity in features"
        else:
            conf_assessment = "Low confidence - Borderline case, manual review recommended"
        
        # Recommendations
        if prediction == 1:
            recommendations = [
                "Consider manual review of the audio sample",
                "Cross-reference with known TTS model outputs",
                "Analyze longer samples if available for consistency"
            ]
        else:
            recommendations = [
                "Classification appears reliable",
                "No additional verification recommended",
                "Archive for reference if needed"
            ]
        
        if confidence < 0.7:
            recommendations.insert(0, "LOW CONFIDENCE: Manual verification strongly recommended")
        
        return ForensicExplanation(
            summary=summary,
            prediction_analysis=analysis,
            evidence_points=evidence,
            potential_tts_model=potential_model,
            confidence_assessment=conf_assessment,
            recommendations=recommendations,
            retrieved_context=context[:500] if context else "No context retrieved"
        )


def generate_forensic_explanation(
    prediction: int,
    confidence: float,
    transcript: str,
    anomalous_features: List[str],
    audio_features: Optional[Dict] = None,
    text_features: Optional[Dict] = None,
    llm_provider: str = "local"
) -> Dict:
    """
    Convenience function to generate forensic explanation.
    
    Args:
        prediction: Model prediction (0=Human, 1=AI)
        confidence: Prediction confidence
        transcript: Audio transcript
        anomalous_features: Detected anomalies
        audio_features: Audio features dict
        text_features: Text features dict
        llm_provider: LLM provider to use
        
    Returns:
        Dictionary with explanation
    """
    explainer = LLMExplainer(provider=llm_provider)
    explanation = explainer.generate_explanation(
        prediction, confidence, transcript, anomalous_features,
        audio_features, text_features
    )
    return explanation.to_dict()


if __name__ == "__main__":
    # Test the explainer
    print("LLM Explainer Test")
    print("=" * 50)
    
    explainer = LLMExplainer(provider="local")
    
    # Test AI-generated explanation
    print("\nTest: AI-generated prediction")
    explanation = explainer.generate_explanation(
        prediction=1,
        confidence=0.87,
        transcript="This is a sample transcript of the audio that was analyzed.",
        anomalous_features=[
            "Very low disfluency rate",
            "Absence of filler words",
            "Low pitch jitter"
        ],
        audio_features={'pitch_jitter': 0.01, 'pitch_var': 45},
        text_features={'disfluency_rate': 0.3, 'perplexity': 25, 'filler_count': 0}
    )
    
    print("\n" + explanation.to_markdown())
