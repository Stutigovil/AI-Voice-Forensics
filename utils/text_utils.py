"""
Text Utilities
==============
Common text processing functions for transcript analysis.

Features:
- Text cleaning and normalization
- Disfluency detection
- Sentence segmentation
- POS tagging utilities
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('tagsets/upenn_tagset.pickle')
except LookupError:
    nltk.download('tagsets', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Common disfluency markers in speech
DISFLUENCY_MARKERS = {
    'fillers': ['uh', 'um', 'er', 'ah', 'eh', 'hmm', 'hm', 'mm', 'mhm', 'erm'],
    'discourse': ['like', 'you know', 'i mean', 'well', 'so', 'right', 'okay', 'actually'],
    'repairs': ['i mean', 'rather', 'sorry', 'no wait', 'let me rephrase'],
    'false_starts_patterns': [
        r'\b(\w+)\s+\1\b',  # Word repetition
        r'\b(i|we|he|she|they)\s+(i|we|he|she|they)\b',  # Pronoun restart
    ]
}


class TextProcessor:
    """
    Text processing utilities for analyzing transcripts.
    
    Provides functions for cleaning, normalizing, and analyzing
    text from speech transcriptions.
    """
    
    def __init__(self, use_spacy: bool = False, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the text processor.
        
        Args:
            use_spacy: Whether to use spaCy for advanced NLP
            spacy_model: spaCy model to load
        """
        self.use_spacy = use_spacy
        self.nlp = None
        
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load(spacy_model)
                logger.info(f"Loaded spaCy model: {spacy_model}")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}. Using NLTK fallback.")
                self.use_spacy = False
    
    def clean_text(
        self,
        text: str,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        remove_extra_spaces: bool = True
    ) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation marks
            remove_extra_spaces: Collapse multiple spaces
            
        Returns:
            Cleaned text
        """
        if lowercase:
            text = text.lower()
        
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        if remove_extra_spaces:
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            return [token.text for token in doc]
        return word_tokenize(text)
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences."""
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            return [sent.text for sent in doc.sents]
        return sent_tokenize(text)
    
    def get_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        """
        Get part-of-speech tags for text.
        
        Args:
            text: Input text
            
        Returns:
            List of (word, tag) tuples
        """
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            return [(token.text, token.pos_) for token in doc]
        
        words = word_tokenize(text)
        return nltk.pos_tag(words)
    
    def detect_disfluencies(self, text: str) -> Dict[str, List[str]]:
        """
        Detect disfluency markers in text.
        
        Args:
            text: Input text (transcript)
            
        Returns:
            Dictionary with detected disfluencies by type
        """
        text_lower = text.lower()
        words = self.tokenize_words(text_lower)
        
        detected = {
            'fillers': [],
            'discourse_markers': [],
            'repetitions': [],
            'false_starts': []
        }
        
        # Detect fillers (uh, um, etc.)
        for filler in DISFLUENCY_MARKERS['fillers']:
            count = words.count(filler)
            if count > 0:
                detected['fillers'].extend([filler] * count)
        
        # Detect discourse markers
        for marker in DISFLUENCY_MARKERS['discourse']:
            if marker in text_lower:
                count = text_lower.count(marker)
                detected['discourse_markers'].extend([marker] * count)
        
        # Detect word repetitions
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                detected['repetitions'].append(words[i])
        
        # Detect patterns indicating false starts
        for pattern in DISFLUENCY_MARKERS['false_starts_patterns']:
            matches = re.findall(pattern, text_lower)
            detected['false_starts'].extend(matches)
        
        return detected
    
    def calculate_disfluency_rate(self, text: str) -> float:
        """
        Calculate the rate of disfluencies per 100 words.
        
        Args:
            text: Input text
            
        Returns:
            Disfluency rate (disfluencies per 100 words)
        """
        words = self.tokenize_words(text)
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        disfluencies = self.detect_disfluencies(text)
        total_disfluencies = sum(len(v) for v in disfluencies.values())
        
        return (total_disfluencies / word_count) * 100
    
    def get_ngrams(self, text: str, n: int = 2) -> List[Tuple[str, ...]]:
        """
        Generate n-grams from text.
        
        Args:
            text: Input text
            n: N-gram size
            
        Returns:
            List of n-gram tuples
        """
        words = self.tokenize_words(self.clean_text(text, remove_punctuation=True))
        return list(nltk.ngrams(words, n))
    
    def calculate_repetition_score(self, text: str, n: int = 2) -> float:
        """
        Calculate n-gram repetition score.
        
        Higher scores indicate more repetition (potentially AI-generated).
        
        Args:
            text: Input text
            n: N-gram size to analyze
            
        Returns:
            Repetition score (0-1, higher = more repetition)
        """
        ngrams = self.get_ngrams(text, n)
        
        if len(ngrams) == 0:
            return 0.0
        
        ngram_counts = Counter(ngrams)
        repeated = sum(1 for count in ngram_counts.values() if count > 1)
        
        return repeated / len(ngram_counts) if ngram_counts else 0.0
    
    def get_sentence_lengths(self, text: str) -> List[int]:
        """
        Get word counts for each sentence.
        
        Args:
            text: Input text
            
        Returns:
            List of sentence lengths (word counts)
        """
        sentences = self.tokenize_sentences(text)
        return [len(self.tokenize_words(sent)) for sent in sentences]
    
    def extract_named_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of (entity, label) tuples
        """
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            return [(ent.text, ent.label_) for ent in doc.ents]
        
        # Fallback: simple capitalized word detection
        words = word_tokenize(text)
        entities = []
        for word in words:
            if word[0].isupper() and word.lower() not in ['i', 'the', 'a', 'an']:
                entities.append((word, 'ENTITY'))
        return entities


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = False
) -> str:
    """
    Convenience function to clean text.
    
    Args:
        text: Input text
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation
        
    Returns:
        Cleaned text
    """
    processor = TextProcessor()
    return processor.clean_text(text, lowercase, remove_punctuation)


def get_disfluencies(text: str) -> Dict[str, List[str]]:
    """
    Convenience function to detect disfluencies.
    
    Args:
        text: Input text (transcript)
        
    Returns:
        Dictionary of detected disfluencies
    """
    processor = TextProcessor()
    return processor.detect_disfluencies(text)


def analyze_transcript(text: str, use_spacy: bool = False) -> Dict:
    """
    Perform comprehensive analysis of a transcript.
    
    Args:
        text: Input transcript text
        use_spacy: Whether to use spaCy for advanced analysis
        
    Returns:
        Dictionary with analysis results
    """
    processor = TextProcessor(use_spacy=use_spacy)
    
    words = processor.tokenize_words(text)
    sentences = processor.tokenize_sentences(text)
    disfluencies = processor.detect_disfluencies(text)
    
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sentence_length": sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0,
        "disfluency_rate": processor.calculate_disfluency_rate(text),
        "repetition_score_bigram": processor.calculate_repetition_score(text, 2),
        "repetition_score_trigram": processor.calculate_repetition_score(text, 3),
        "disfluencies": disfluencies,
        "unique_words": len(set(words)),
        "lexical_diversity": len(set(words)) / len(words) if words else 0
    }


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Um, so like, I was thinking about, you know, the the project we discussed. 
    I mean, it's actually pretty interesting. The the main idea is, well, 
    it's about detecting AI generated speech. Uh, we use various features like, 
    like spectral analysis and, er, linguistic patterns.
    """
    
    print("Transcript Analysis")
    print("=" * 50)
    
    analysis = analyze_transcript(sample_text)
    
    for key, value in analysis.items():
        if key != "disfluencies":
            print(f"{key}: {value}")
    
    print("\nDisfluencies detected:")
    for dtype, items in analysis["disfluencies"].items():
        if items:
            print(f"  {dtype}: {items}")
