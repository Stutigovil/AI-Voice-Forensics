"""
Document Retriever Module
=========================
Retrieves relevant documents from the knowledge base for RAG explanations.

Takes model predictions and anomalous features to find relevant
documentation for generating forensic explanations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .build_index import KnowledgeBaseBuilder, Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentRetriever:
    """
    Retrieves relevant documents for RAG-based explanations.
    
    Combines semantic search with rule-based filtering to find
    the most relevant documentation for explaining predictions.
    """
    
    # Feature to category mapping
    FEATURE_CATEGORIES = {
        'pitch': ['tts_artifacts', 'methodology'],
        'mfcc': ['methodology', 'research'],
        'spectral': ['tts_artifacts', 'research'],
        'disfluency': ['linguistic_patterns', 'methodology'],
        'perplexity': ['linguistic_patterns', 'methodology'],
        'vocoder': ['tts_artifacts', 'research'],
        'jitter': ['methodology', 'tts_artifacts'],
        'prosody': ['tts_artifacts', 'linguistic_patterns']
    }
    
    # TTS model keywords
    TTS_KEYWORDS = {
        'tacotron': 'tts_tacotron',
        'vits': 'tts_vits',
        'bark': 'tts_bark',
        'elevenlabs': 'tts_elevenlabs',
        'wavenet': 'research_vocoder_fingerprints'
    }
    
    def __init__(
        self,
        index_path: str = "rag/index",
        vector_db: str = "faiss",
        top_k: int = 5
    ):
        """
        Initialize the retriever.
        
        Args:
            index_path: Path to the vector index
            vector_db: Vector database type
            top_k: Default number of results to retrieve
        """
        self.index_path = Path(index_path)
        self.top_k = top_k
        
        # Initialize knowledge base
        self.kb = KnowledgeBaseBuilder(
            vector_db=vector_db,
            index_path=str(index_path)
        )
        
        # Try to load existing index
        if self.index_path.exists() and (self.index_path / "documents.json").exists():
            self.kb.load()
            self.is_loaded = True
            logger.info("Loaded existing knowledge base index")
        else:
            # Build new index
            logger.info("Building new knowledge base index...")
            self.kb.load_default_knowledge_base()
            self.kb.build_index()
            self.kb.save()
            self.is_loaded = True
    
    def retrieve(
        self,
        prediction: int,
        anomalous_features: List[str],
        audio_features: Optional[Dict] = None,
        text_features: Optional[Dict] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents based on prediction and features.
        
        Args:
            prediction: Model prediction (0=Human, 1=AI)
            anomalous_features: List of anomalous feature descriptions
            audio_features: Dictionary of audio features
            text_features: Dictionary of text features
            
        Returns:
            List of relevant Documents
        """
        if not self.is_loaded:
            logger.error("Knowledge base not loaded")
            return []
        
        # Build search query from context
        query = self._build_search_query(prediction, anomalous_features, audio_features, text_features)
        
        # Semantic search
        results = self.kb.search(query, top_k=self.top_k * 2)
        
        # Filter and re-rank based on prediction context
        filtered_results = self._filter_results(results, prediction, anomalous_features)
        
        return filtered_results[:self.top_k]
    
    def _build_search_query(
        self,
        prediction: int,
        anomalous_features: List[str],
        audio_features: Optional[Dict],
        text_features: Optional[Dict]
    ) -> str:
        """Build a search query from the prediction context."""
        query_parts = []
        
        # Base on prediction
        if prediction == 1:  # AI-generated
            query_parts.append("AI generated speech TTS detection artifacts")
        else:
            query_parts.append("human natural speech characteristics")
        
        # Add anomalous features
        for anomaly in anomalous_features[:3]:  # Limit to avoid overly long query
            # Extract key terms
            anomaly_lower = anomaly.lower()
            if 'disfluency' in anomaly_lower:
                query_parts.append("disfluency filler words speech patterns")
            elif 'perplexity' in anomaly_lower:
                query_parts.append("text perplexity LLM generated")
            elif 'pitch' in anomaly_lower:
                query_parts.append("pitch variation prosody monotone")
            elif 'jitter' in anomaly_lower:
                query_parts.append("pitch jitter natural variation")
            elif 'spectral' in anomaly_lower:
                query_parts.append("spectral analysis vocoder artifacts")
            elif 'sentence' in anomaly_lower:
                query_parts.append("sentence length uniformity AI text")
        
        # Add specific feature-based terms
        if audio_features:
            if audio_features.get('pitch_jitter', 1) < 0.02:
                query_parts.append("low pitch jitter synthetic speech")
            if audio_features.get('spectral_flatness_mean', 0) > 0.4:
                query_parts.append("spectral flatness vocoder")
        
        if text_features:
            if text_features.get('disfluency_rate', 10) < 1:
                query_parts.append("absence disfluency markers")
            if text_features.get('perplexity', 100) < 30:
                query_parts.append("low perplexity LLM generated text")
        
        return " ".join(query_parts)
    
    def _filter_results(
        self,
        results: List[Document],
        prediction: int,
        anomalous_features: List[str]
    ) -> List[Document]:
        """Filter and re-rank results based on prediction context."""
        # Determine relevant categories
        relevant_categories = set()
        
        if prediction == 1:  # AI-generated
            relevant_categories.update(['tts_artifacts', 'attribution', 'research'])
        else:
            relevant_categories.add('linguistic_patterns')
        
        # Add categories based on anomalous features
        for anomaly in anomalous_features:
            anomaly_lower = anomaly.lower()
            for keyword, categories in self.FEATURE_CATEGORIES.items():
                if keyword in anomaly_lower:
                    relevant_categories.update(categories)
        
        # Filter and boost relevant documents
        filtered = []
        for doc in results:
            category = doc.metadata.get('category', '')
            score = doc.metadata.get('score', 0)
            
            # Boost score for relevant categories
            if category in relevant_categories:
                score *= 1.5
            
            doc.metadata['adjusted_score'] = score
            filtered.append(doc)
        
        # Sort by adjusted score
        filtered.sort(key=lambda x: x.metadata.get('adjusted_score', 0), reverse=True)
        
        return filtered
    
    def retrieve_by_query(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Direct semantic search with a query string.
        
        Args:
            query: Search query
            top_k: Number of results (defaults to self.top_k)
            
        Returns:
            List of relevant Documents
        """
        k = top_k or self.top_k
        return self.kb.search(query, top_k=k)
    
    def retrieve_by_category(self, category: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve documents by category.
        
        Args:
            category: Category to filter by
            top_k: Number of results
            
        Returns:
            List of Documents in the category
        """
        results = [doc for doc in self.kb.documents if doc.metadata.get('category') == category]
        return results[:top_k]
    
    def get_tts_model_info(self, model_hint: Optional[str] = None) -> List[Document]:
        """
        Get information about specific TTS models.
        
        Args:
            model_hint: Optional hint about the TTS model
            
        Returns:
            Relevant TTS model documentation
        """
        if model_hint:
            model_lower = model_hint.lower()
            for keyword, doc_id in self.TTS_KEYWORDS.items():
                if keyword in model_lower:
                    return [doc for doc in self.kb.documents if doc.id == doc_id]
        
        # Return all TTS artifact documents
        return self.retrieve_by_category('tts_artifacts')
    
    def format_context(self, documents: List[Document], max_length: int = 2000) -> str:
        """
        Format retrieved documents as context for LLM.
        
        Args:
            documents: List of documents to format
            max_length: Maximum total length
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for doc in documents:
            title = doc.metadata.get('title', 'Reference')
            category = doc.metadata.get('category', 'general')
            
            # Truncate text if needed
            max_doc_length = (max_length - current_length) // (len(documents) - len(context_parts))
            text = doc.text[:max_doc_length].strip()
            if len(doc.text) > max_doc_length:
                text += "..."
            
            section = f"### {title} [{category}]\n{text}"
            context_parts.append(section)
            current_length += len(section)
            
            if current_length >= max_length:
                break
        
        return "\n\n".join(context_parts)


def create_retriever(index_path: str = "rag/index") -> DocumentRetriever:
    """
    Create a document retriever instance.
    
    Args:
        index_path: Path to the index
        
    Returns:
        Configured DocumentRetriever
    """
    return DocumentRetriever(index_path=index_path)


if __name__ == "__main__":
    # Test the retriever
    print("Document Retriever Test")
    print("=" * 50)
    
    retriever = DocumentRetriever()
    
    # Test 1: Retrieve for AI prediction with low disfluency
    print("\nTest 1: AI prediction with low disfluency")
    results = retriever.retrieve(
        prediction=1,
        anomalous_features=["Very low disfluency rate", "Low pitch jitter"],
        audio_features={'pitch_jitter': 0.01},
        text_features={'disfluency_rate': 0.5}
    )
    
    print(f"Found {len(results)} relevant documents:")
    for doc in results:
        print(f"  - {doc.metadata.get('title', 'Untitled')} (score: {doc.metadata.get('score', 0):.3f})")
    
    # Test 2: Direct query
    print("\nTest 2: Direct query for vocoder artifacts")
    results = retriever.retrieve_by_query("vocoder artifacts spectral analysis")
    
    print(f"Found {len(results)} relevant documents:")
    for doc in results:
        print(f"  - {doc.metadata.get('title', 'Untitled')} (score: {doc.metadata.get('score', 0):.3f})")
    
    # Test 3: Get formatted context
    print("\nTest 3: Formatted context for LLM")
    context = retriever.format_context(results[:2], max_length=500)
    print(context[:500] + "...")
