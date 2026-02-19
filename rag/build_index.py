"""
Vector Index Builder Module
===========================
Builds and manages the vector database for RAG-based explanations.

Knowledge Base Contents:
1. Research paper summaries on AI speech synthesis
2. Known TTS model artifacts (Tacotron, VITS, Bark)
3. Linguistic patterns of LLM-generated text
4. Detection methodology documentation
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document chunk for indexing."""
    id: str
    text: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'text': self.text,
            'metadata': self.metadata
        }


# Knowledge base content for AI voice detection
KNOWLEDGE_BASE_CONTENT = [
    # TTS Model Artifacts
    {
        "id": "tts_tacotron",
        "category": "tts_artifacts",
        "title": "Tacotron/Tacotron2 Artifacts",
        "text": """
Tacotron and Tacotron2 are sequence-to-sequence text-to-speech models that generate mel spectrograms from text.
Common artifacts include:
- Attention alignment issues causing word skipping or repetition
- Mel spectrogram reconstruction errors leading to metallic or buzzy sounds
- Griffin-Lim vocoder introduces phase artifacts
- WaveGlow/HiFi-GAN vocoders may produce subtle harmonic distortions
- Prosody tends to be more monotonic than natural speech
- Breath sounds are often synthesized unnaturally or absent
- Sentence-final intonation patterns may be incorrect
Detection indicators: Low pitch jitter, uniform energy distribution, absence of micro-prosodic variations.
"""
    },
    {
        "id": "tts_vits",
        "category": "tts_artifacts",
        "title": "VITS Model Artifacts",
        "text": """
VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) is a parallel end-to-end TTS model.
Characteristics and artifacts:
- End-to-end generation may produce smoother but less natural prosody
- Normalizing flows can cause subtle spectral artifacts
- Duration prediction may lead to unnatural timing
- Speaker embeddings can cause inconsistent voice quality
- Less prone to attention failures than autoregressive models
- May produce unnaturally consistent formant transitions
- Emotional expression tends to be limited
Detection indicators: Very smooth F0 contours, consistent spectral energy, lack of hesitation markers.
"""
    },
    {
        "id": "tts_bark",
        "category": "tts_artifacts",
        "title": "Bark Model Artifacts",
        "text": """
Bark by Suno is a transformer-based text-to-audio model capable of generating speech with non-verbal sounds.
Characteristics:
- Can generate laughter, sighs, and other non-verbal sounds
- Uses semantic tokens and acoustic tokens pipeline
- May produce more natural prosody than earlier models
- Can exhibit "hallucinations" generating incorrect words
- Background noise generation may be unrealistic
- Long-form generation may have consistency issues
- Speaker identity may drift over long utterances
Detection indicators: Non-verbal sound placement may seem random, potential word substitutions, drift in voice quality.
"""
    },
    {
        "id": "tts_elevenlabs",
        "category": "tts_artifacts",
        "title": "ElevenLabs Artifacts",
        "text": """
ElevenLabs uses advanced neural TTS with voice cloning capabilities.
Characteristics:
- High-quality output often indistinguishable from human speech
- Voice cloning may produce subtle inconsistencies
- Emotional range can be limited despite high quality
- May exhibit unnaturally perfect pronunciation
- Breath patterns may be too regular
- Lacks natural speech disfluencies (um, uh)
- Sentence transitions may be too smooth
Detection indicators: Perfect pronunciation, absence of speech errors, overly consistent pacing.
"""
    },
    
    # Linguistic Patterns of AI-Generated Text
    {
        "id": "llm_text_patterns",
        "category": "linguistic_patterns",
        "title": "LLM-Generated Text Characteristics",
        "text": """
Large Language Model generated text exhibits distinct linguistic patterns:
- Lower perplexity scores (more predictable word sequences)
- More uniform sentence lengths
- Absence of natural speech disfluencies
- Perfect grammar with no false starts or repairs
- Repetitive n-gram patterns
- Formulaic phrase structures
- Overuse of certain transition words
- Lack of personal anecdotes or specific details
- Tendency toward generic, hedging language
When converted to speech via TTS, these patterns persist and can be detected through text analysis.
"""
    },
    {
        "id": "human_speech_patterns",
        "category": "linguistic_patterns",
        "title": "Natural Human Speech Patterns",
        "text": """
Natural human speech has distinctive characteristics:
- Contains filler words (uh, um, er, like, you know)
- Includes false starts and self-corrections
- Variable sentence lengths with natural rhythm
- Personal pronouns and specific references
- Irregular pauses and breath sounds
- Micro-prosodic variations (shimmer, jitter)
- Natural pitch contours with emotion
- Contextual disfluencies increase under cognitive load
- Speech rate varies with content complexity
These features are difficult for AI systems to replicate authentically.
"""
    },
    
    # Research Findings
    {
        "id": "research_deepfake_detection",
        "category": "research",
        "title": "Audio Deepfake Detection Research Summary",
        "text": """
Key findings from audio deepfake detection research:
1. ASVspoof Challenge findings show neural vocoders leave detectable artifacts
2. MFCC coefficients capture vocoder fingerprints
3. Spectral analysis reveals phase discontinuities
4. Bispectral analysis can detect non-linear distortions
5. End-to-end neural networks achieve high detection rates
6. Transfer learning from speech recognition improves detection
7. Multimodal approaches combining audio and text features outperform single-modality
8. Zero-shot generalization remains challenging
9. Adversarial attacks can evade detection systems
Best practices: Combine acoustic and linguistic features, use ensemble methods, regular model updates.
"""
    },
    {
        "id": "research_vocoder_fingerprints",
        "category": "research",
        "title": "Vocoder Fingerprint Analysis",
        "text": """
Different vocoders leave distinct fingerprints in generated audio:
Griffin-Lim: Phase reconstruction artifacts, metallic quality
WaveNet: High quality but computationally expensive, subtle artifacts in onsets
WaveGlow: Flow-based artifacts, potential for high-frequency distortions
HiFi-GAN: Fast inference, may have aliasing artifacts
MelGAN: Band-limiting effects, potential harmonics issues
UnivNet: Generally high quality, fewer distinguishable artifacts
Multi-band vocoders may show band-boundary artifacts
Neural vocoders often smooth micro-timing variations
Detection approach: Analyze spectral characteristics, especially in higher frequencies.
"""
    },
    
    # Detection Methodology
    {
        "id": "methodology_audio_features",
        "category": "methodology",
        "title": "Audio Feature Analysis Methodology",
        "text": """
Audio feature analysis for AI voice detection:
MFCC Analysis:
- Extract 13-40 MFCCs and their deltas
- Compare mean and variance patterns
- AI audio often shows more uniform MFCC distributions

Pitch Analysis:
- Extract F0 using PYIN or CREPE
- Calculate jitter (cycle-to-cycle variation)
- Human speech has jitter ~0.02-0.05
- AI speech often has lower jitter values

Energy Analysis:
- RMS energy distribution
- Energy entropy (natural speech is more random)
- Breath patterns and pauses

Spectral Analysis:
- Spectral centroid and bandwidth
- Spectral flatness (tonal vs noise)
- Formant transitions
"""
    },
    {
        "id": "methodology_text_features",
        "category": "methodology",
        "title": "Text Feature Analysis Methodology",
        "text": """
Text feature analysis for detecting AI-generated speech:
Perplexity Analysis:
- Calculate using GPT-2 or similar LM
- Lower perplexity suggests AI generation
- Human perplexity typically 50-200
- AI-generated text often below 50

Disfluency Analysis:
- Count filler words (um, uh, er)
- Detect word repetitions
- Find false starts and repairs
- Human: 3-7 disfluencies per 100 words
- AI: Often 0-1 disfluencies per 100 words

POS Pattern Analysis:
- Calculate POS tag entropy
- AI text may have unusual distributions
- Function word ratio analysis

Coherence Metrics:
- Sentence-to-sentence similarity
- Topic consistency
- Reference resolution
"""
    },
    
    # Model Attribution
    {
        "id": "model_attribution",
        "category": "attribution",
        "title": "TTS Model Attribution Guidelines",
        "text": """
Guidelines for attributing detected AI audio to specific TTS systems:
VITS-family indicators:
- Very smooth pitch contours
- Consistent speaker timbre
- Natural-sounding but emotionally flat

Tacotron-family indicators:
- Potential attention alignment issues
- Metallic quality in some phonemes
- More variable quality across utterances

Bark indicators:
- May include non-verbal sounds
- Potential word substitutions
- Variable audio quality

ElevenLabs indicators:
- High production quality
- Perfect pronunciation
- Lack of disfluencies

Coqui TTS indicators:
- Similar to VITS characteristics
- May show specific vocoder artifacts

Note: Attribution confidence varies; use multiple indicators.
"""
    }
]


class KnowledgeBaseBuilder:
    """
    Builds and manages the vector index for RAG.
    
    Supports both FAISS and ChromaDB as vector stores.
    """
    
    def __init__(
        self,
        vector_db: str = "faiss",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str = "rag/index"
    ):
        """
        Initialize the knowledge base builder.
        
        Args:
            vector_db: Vector database type ('faiss' or 'chromadb')
            embedding_model: Sentence transformer model for embeddings
            index_path: Path to store the index
        """
        self.vector_db = vector_db
        self.embedding_model_name = embedding_model
        self.index_path = Path(index_path)
        
        # Initialize embedding model
        self.embedding_model = None
        self._init_embedding_model()
        
        # Vector store
        self.index = None
        self.documents = []
        self.embeddings = None
    
    def _init_embedding_model(self):
        """Initialize the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def add_documents(self, documents: List[Dict]) -> None:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of document dictionaries with 'id', 'text', and optionally 'metadata'
        """
        for doc in documents:
            document = Document(
                id=doc.get('id', str(len(self.documents))),
                text=doc['text'],
                metadata={
                    'category': doc.get('category', 'general'),
                    'title': doc.get('title', ''),
                    **doc.get('metadata', {})
                }
            )
            self.documents.append(document)
        
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def load_default_knowledge_base(self) -> None:
        """Load the default AI voice detection knowledge base."""
        self.add_documents(KNOWLEDGE_BASE_CONTENT)
        logger.info("Loaded default knowledge base")
    
    def build_index(self) -> None:
        """Build the vector index from loaded documents."""
        if not self.documents:
            raise ValueError("No documents loaded. Call add_documents() or load_default_knowledge_base() first.")
        
        logger.info(f"Building {self.vector_db} index for {len(self.documents)} documents...")
        
        # Generate embeddings
        texts = [doc.text for doc in self.documents]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        self.embeddings = np.array(embeddings).astype('float32')
        
        # Store embeddings in documents
        for doc, emb in zip(self.documents, self.embeddings):
            doc.embedding = emb
        
        if self.vector_db == "faiss":
            self._build_faiss_index()
        elif self.vector_db == "chromadb":
            self._build_chroma_index()
        else:
            raise ValueError(f"Unsupported vector database: {self.vector_db}")
        
        logger.info("Index built successfully")
    
    def _build_faiss_index(self):
        """Build FAISS index."""
        import faiss
        
        # Create index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine with normalized vectors)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Add to index
        self.index.add(self.embeddings)
        
        logger.info(f"FAISS index created with {self.index.ntotal} vectors")
    
    def _build_chroma_index(self):
        """Build ChromaDB collection."""
        import chromadb
        from chromadb.config import Settings
        
        # Create ChromaDB client
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(self.index_path / "chromadb"),
            anonymized_telemetry=False
        ))
        
        # Create collection
        self.index = client.create_collection(
            name="ai_voice_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Add documents
        self.index.add(
            embeddings=self.embeddings.tolist(),
            documents=[doc.text for doc in self.documents],
            metadatas=[doc.metadata for doc in self.documents],
            ids=[doc.id for doc in self.documents]
        )
        
        logger.info(f"ChromaDB collection created with {self.index.count()} documents")
    
    def save(self) -> None:
        """Save the index and documents to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Save documents metadata
        doc_data = [doc.to_dict() for doc in self.documents]
        with open(self.index_path / "documents.json", 'w') as f:
            json.dump(doc_data, f, indent=2)
        
        # Save embeddings
        np.save(self.index_path / "embeddings.npy", self.embeddings)
        
        if self.vector_db == "faiss":
            import faiss
            faiss.write_index(self.index, str(self.index_path / "faiss.index"))
        
        logger.info(f"Index saved to: {self.index_path}")
    
    def load(self) -> None:
        """Load a previously saved index."""
        # Load documents
        with open(self.index_path / "documents.json", 'r') as f:
            doc_data = json.load(f)
        
        self.documents = [
            Document(id=d['id'], text=d['text'], metadata=d['metadata'])
            for d in doc_data
        ]
        
        # Load embeddings
        self.embeddings = np.load(self.index_path / "embeddings.npy")
        
        if self.vector_db == "faiss":
            import faiss
            self.index = faiss.read_index(str(self.index_path / "faiss.index"))
        
        logger.info(f"Index loaded from: {self.index_path}")
    
    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant Documents
        """
        # Get query embedding
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        
        if self.vector_db == "faiss":
            import faiss
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    doc.metadata['score'] = float(score)
                    results.append(doc)
            
            return results
        
        elif self.vector_db == "chromadb":
            results = self.index.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            
            return [
                Document(
                    id=id_,
                    text=text,
                    metadata={**meta, 'score': 1.0}  # ChromaDB returns sorted results
                )
                for id_, text, meta in zip(
                    results['ids'][0],
                    results['documents'][0],
                    results['metadatas'][0]
                )
            ]
        
        return []


def build_vector_index(
    output_path: str = "rag/index",
    vector_db: str = "faiss"
) -> KnowledgeBaseBuilder:
    """
    Convenience function to build the knowledge base.
    
    Args:
        output_path: Path to save the index
        vector_db: Vector database type
        
    Returns:
        KnowledgeBaseBuilder instance
    """
    builder = KnowledgeBaseBuilder(
        vector_db=vector_db,
        index_path=output_path
    )
    
    builder.load_default_knowledge_base()
    builder.build_index()
    builder.save()
    
    return builder


if __name__ == "__main__":
    # Build the default knowledge base
    print("Building AI Voice Detection Knowledge Base")
    print("=" * 50)
    
    builder = build_vector_index()
    
    # Test search
    print("\nTesting search...")
    results = builder.search("vocoder artifacts in TTS systems", top_k=3)
    
    print(f"\nFound {len(results)} relevant documents:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.metadata.get('title', 'Untitled')}")
        print(f"   Category: {doc.metadata.get('category', 'N/A')}")
        print(f"   Score: {doc.metadata.get('score', 0):.4f}")
        print(f"   Preview: {doc.text[:200]}...")
