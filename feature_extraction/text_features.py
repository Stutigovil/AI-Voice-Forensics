"""
Text Feature Extraction Module
==============================
Extracts linguistic features from transcribed text for AI voice detection.

Features extracted:
- Perplexity (using GPT-2 or KenLM)
- Sentence length statistics
- POS tag distribution and entropy
- Disfluency rate
- Repetition patterns
- Lexical diversity measures

These features help detect LLM-generated text that was then converted
to speech using TTS systems.
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from scipy import stats

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextFeatures:
    """Container for extracted text features."""
    # Perplexity
    perplexity: float
    log_perplexity: float
    
    # Sentence statistics
    sentence_count: int
    avg_sentence_length: float
    sentence_length_var: float
    sentence_length_std: float
    max_sentence_length: int
    min_sentence_length: int
    
    # Word statistics
    word_count: int
    avg_word_length: float
    unique_word_ratio: float  # Lexical diversity
    
    # POS tag features
    pos_entropy: float
    noun_ratio: float
    verb_ratio: float
    adj_ratio: float
    adv_ratio: float
    function_word_ratio: float
    
    # Disfluency features
    disfluency_rate: float
    filler_count: int
    repetition_rate: float
    
    # Repetition patterns
    bigram_repetition_score: float
    trigram_repetition_score: float
    
    # Coherence metrics
    sentence_similarity_mean: float
    transition_word_ratio: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_flat_array(self) -> np.ndarray:
        """Convert features to flat numpy array for model input."""
        return np.array(list(self.to_dict().values()), dtype=np.float32)
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get names of all features."""
        return [
            'perplexity', 'log_perplexity',
            'sentence_count', 'avg_sentence_length', 'sentence_length_var',
            'sentence_length_std', 'max_sentence_length', 'min_sentence_length',
            'word_count', 'avg_word_length', 'unique_word_ratio',
            'pos_entropy', 'noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio',
            'function_word_ratio', 'disfluency_rate', 'filler_count',
            'repetition_rate', 'bigram_repetition_score', 'trigram_repetition_score',
            'sentence_similarity_mean', 'transition_word_ratio'
        ]


class TextFeatureExtractor:
    """
    Extracts linguistic features from text for AI detection.
    
    Key detection signals:
    - LLM-generated text often has lower perplexity (more predictable)
    - AI text typically lacks natural disfluencies (uh, um)
    - TTS input scripts may have unusual sentence length patterns
    - Repetitive n-gram patterns can indicate AI generation
    """
    
    # Common filler words/disfluencies
    FILLERS = ['uh', 'um', 'er', 'ah', 'eh', 'hmm', 'hm', 'mm', 'mhm', 'erm', 'like']
    
    # Transition words indicating natural flow
    TRANSITIONS = [
        'however', 'therefore', 'moreover', 'furthermore', 'additionally',
        'meanwhile', 'consequently', 'nevertheless', 'although', 'because',
        'since', 'while', 'when', 'if', 'unless', 'but', 'and', 'or', 'so',
        'then', 'thus', 'hence', 'yet', 'still', 'also', 'first', 'second',
        'finally', 'next', 'lastly'
    ]
    
    # Function words (closed class words)
    FUNCTION_WORD_TAGS = ['IN', 'DT', 'CC', 'TO', 'PRP', 'PRP$', 'WP', 'WP$', 'MD']
    
    def __init__(
        self,
        use_transformers: bool = True,
        perplexity_model: str = "gpt2",
        use_spacy: bool = False,
        spacy_model: str = "en_core_web_sm"
    ):
        """
        Initialize the text feature extractor.
        
        Args:
            use_transformers: Use transformers for perplexity calculation
            perplexity_model: Model name for perplexity (gpt2, etc.)
            use_spacy: Use spaCy for advanced NLP
            spacy_model: spaCy model to use
        """
        self.use_transformers = use_transformers
        self.perplexity_model_name = perplexity_model
        self.tokenizer = None
        self.perplexity_model = None
        
        # Initialize perplexity model
        if use_transformers:
            self._init_perplexity_model()
        
        # Initialize spaCy if requested
        self.nlp = None
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load(spacy_model)
            except Exception as e:
                logger.warning(f"Failed to load spaCy: {e}")
    
    def _init_perplexity_model(self):
        """Initialize the transformer model for perplexity calculation."""
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            import torch
            
            logger.info(f"Loading {self.perplexity_model_name} for perplexity calculation...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.perplexity_model_name)
            self.perplexity_model = GPT2LMHeadModel.from_pretrained(self.perplexity_model_name)
            self.perplexity_model.eval()
            
            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.perplexity_model.to(self.device)
            
            logger.info("Perplexity model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load perplexity model: {e}")
            self.use_transformers = False
    
    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity using GPT-2.
        
        Lower perplexity indicates more predictable text,
        which can be a sign of LLM generation.
        
        Args:
            text: Input text
            
        Returns:
            Perplexity score
        """
        if not self.use_transformers or self.perplexity_model is None:
            return self._simple_perplexity(text)
        
        import torch
        
        try:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Calculate loss
            with torch.no_grad():
                outputs = self.perplexity_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
            
            # Convert loss to perplexity
            perplexity = math.exp(loss)
            
            return min(perplexity, 10000)  # Cap very high values
            
        except Exception as e:
            logger.warning(f"Perplexity calculation failed: {e}")
            return self._simple_perplexity(text)
    
    def _simple_perplexity(self, text: str) -> float:
        """Simple perplexity approximation using word frequency."""
        words = word_tokenize(text.lower())
        if len(words) == 0:
            return 1.0
        
        word_freq = Counter(words)
        total = sum(word_freq.values())
        
        # Calculate entropy-based approximation
        entropy = 0
        for count in word_freq.values():
            p = count / total
            entropy -= p * math.log2(p)
        
        return 2 ** entropy
    
    def calculate_pos_entropy(self, text: str) -> Tuple[float, Dict[str, float]]:
        """
        Calculate POS tag distribution entropy and ratios.
        
        AI-generated text may have different POS distributions
        compared to natural speech transcripts.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (entropy, POS ratios dict)
        """
        words = word_tokenize(text)
        pos_tags = nltk.pos_tag(words)
        
        if len(pos_tags) == 0:
            return 0.0, {}
        
        # Count POS tags
        tag_counts = Counter(tag for word, tag in pos_tags)
        total = sum(tag_counts.values())
        
        # Calculate entropy
        entropy = 0
        for count in tag_counts.values():
            p = count / total
            entropy -= p * math.log2(p) if p > 0 else 0
        
        # Calculate specific ratios
        noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
        verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        adj_tags = ['JJ', 'JJR', 'JJS']
        adv_tags = ['RB', 'RBR', 'RBS']
        
        ratios = {
            'noun_ratio': sum(tag_counts.get(t, 0) for t in noun_tags) / total,
            'verb_ratio': sum(tag_counts.get(t, 0) for t in verb_tags) / total,
            'adj_ratio': sum(tag_counts.get(t, 0) for t in adj_tags) / total,
            'adv_ratio': sum(tag_counts.get(t, 0) for t in adv_tags) / total,
            'function_word_ratio': sum(tag_counts.get(t, 0) for t in self.FUNCTION_WORD_TAGS) / total
        }
        
        return entropy, ratios
    
    def calculate_disfluency_features(self, text: str) -> Dict[str, float]:
        """
        Calculate disfluency-related features.
        
        Human speech naturally contains disfluencies (uh, um, false starts).
        AI-generated audio often lacks these markers.
        
        Args:
            text: Input text (transcript)
            
        Returns:
            Dictionary of disfluency features
        """
        words = word_tokenize(text.lower())
        word_count = len(words)
        
        if word_count == 0:
            return {'disfluency_rate': 0.0, 'filler_count': 0, 'repetition_rate': 0.0}
        
        # Count fillers
        filler_count = sum(1 for w in words if w in self.FILLERS)
        
        # Count immediate word repetitions
        repetition_count = sum(1 for i in range(len(words) - 1) if words[i] == words[i + 1])
        
        # Disfluency rate per 100 words
        disfluency_rate = ((filler_count + repetition_count) / word_count) * 100
        
        return {
            'disfluency_rate': disfluency_rate,
            'filler_count': filler_count,
            'repetition_rate': (repetition_count / word_count) * 100 if word_count > 0 else 0
        }
    
    def calculate_ngram_repetition(self, text: str, n: int = 2) -> float:
        """
        Calculate n-gram repetition score.
        
        High repetition can indicate AI-generated content.
        
        Args:
            text: Input text
            n: N-gram size
            
        Returns:
            Repetition score (0-1)
        """
        words = word_tokenize(text.lower())
        
        if len(words) < n:
            return 0.0
        
        # Generate n-grams
        ngrams = list(nltk.ngrams(words, n))
        
        if len(ngrams) == 0:
            return 0.0
        
        ngram_counts = Counter(ngrams)
        repeated = sum(1 for count in ngram_counts.values() if count > 1)
        
        return repeated / len(ngram_counts)
    
    def calculate_sentence_features(self, text: str) -> Dict[str, float]:
        """
        Calculate sentence-level statistics.
        
        AI-generated text may have more uniform sentence lengths
        compared to natural speech.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of sentence features
        """
        sentences = sent_tokenize(text)
        
        if len(sentences) == 0:
            return {
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'sentence_length_var': 0,
                'sentence_length_std': 0,
                'max_sentence_length': 0,
                'min_sentence_length': 0
            }
        
        sentence_lengths = [len(word_tokenize(s)) for s in sentences]
        
        return {
            'sentence_count': len(sentences),
            'avg_sentence_length': np.mean(sentence_lengths),
            'sentence_length_var': np.var(sentence_lengths),
            'sentence_length_std': np.std(sentence_lengths),
            'max_sentence_length': max(sentence_lengths),
            'min_sentence_length': min(sentence_lengths)
        }
    
    def calculate_word_features(self, text: str) -> Dict[str, float]:
        """
        Calculate word-level statistics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of word features
        """
        words = word_tokenize(text.lower())
        word_lengths = [len(w) for w in words if w.isalpha()]
        
        if len(words) == 0:
            return {
                'word_count': 0,
                'avg_word_length': 0,
                'unique_word_ratio': 0
            }
        
        unique_words = set(words)
        
        return {
            'word_count': len(words),
            'avg_word_length': np.mean(word_lengths) if word_lengths else 0,
            'unique_word_ratio': len(unique_words) / len(words)
        }
    
    def calculate_coherence_features(self, text: str) -> Dict[str, float]:
        """
        Calculate text coherence features.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of coherence features
        """
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        # Transition word ratio
        transition_count = sum(1 for w in words if w in self.TRANSITIONS)
        transition_ratio = transition_count / len(words) if words else 0
        
        # Simple sentence similarity (Jaccard-based)
        if len(sentences) < 2:
            similarity_mean = 0.0
        else:
            similarities = []
            for i in range(len(sentences) - 1):
                words1 = set(word_tokenize(sentences[i].lower()))
                words2 = set(word_tokenize(sentences[i + 1].lower()))
                
                if len(words1 | words2) > 0:
                    jacc = len(words1 & words2) / len(words1 | words2)
                    similarities.append(jacc)
            
            similarity_mean = np.mean(similarities) if similarities else 0.0
        
        return {
            'sentence_similarity_mean': similarity_mean,
            'transition_word_ratio': transition_ratio
        }
    
    def extract(self, text: str) -> TextFeatures:
        """
        Extract all features from text.
        
        Args:
            text: Input text (transcript)
            
        Returns:
            TextFeatures object with all extracted features
        """
        logger.info("Extracting text features...")
        
        # Calculate all feature groups
        perplexity = self.calculate_perplexity(text)
        pos_entropy, pos_ratios = self.calculate_pos_entropy(text)
        disfluency_features = self.calculate_disfluency_features(text)
        sentence_features = self.calculate_sentence_features(text)
        word_features = self.calculate_word_features(text)
        coherence_features = self.calculate_coherence_features(text)
        
        # N-gram repetition
        bigram_rep = self.calculate_ngram_repetition(text, 2)
        trigram_rep = self.calculate_ngram_repetition(text, 3)
        
        # Combine into TextFeatures object
        features = TextFeatures(
            perplexity=perplexity,
            log_perplexity=math.log(perplexity) if perplexity > 0 else 0,
            sentence_count=sentence_features['sentence_count'],
            avg_sentence_length=sentence_features['avg_sentence_length'],
            sentence_length_var=sentence_features['sentence_length_var'],
            sentence_length_std=sentence_features['sentence_length_std'],
            max_sentence_length=sentence_features['max_sentence_length'],
            min_sentence_length=sentence_features['min_sentence_length'],
            word_count=word_features['word_count'],
            avg_word_length=word_features['avg_word_length'],
            unique_word_ratio=word_features['unique_word_ratio'],
            pos_entropy=pos_entropy,
            noun_ratio=pos_ratios.get('noun_ratio', 0),
            verb_ratio=pos_ratios.get('verb_ratio', 0),
            adj_ratio=pos_ratios.get('adj_ratio', 0),
            adv_ratio=pos_ratios.get('adv_ratio', 0),
            function_word_ratio=pos_ratios.get('function_word_ratio', 0),
            disfluency_rate=disfluency_features['disfluency_rate'],
            filler_count=disfluency_features['filler_count'],
            repetition_rate=disfluency_features['repetition_rate'],
            bigram_repetition_score=bigram_rep,
            trigram_repetition_score=trigram_rep,
            sentence_similarity_mean=coherence_features['sentence_similarity_mean'],
            transition_word_ratio=coherence_features['transition_word_ratio']
        )
        
        logger.info(f"Extracted {len(features.to_flat_array())} text features")
        return features
    
    def extract_batch(
        self,
        texts: List[str],
        return_array: bool = True
    ) -> Union[List[TextFeatures], np.ndarray]:
        """
        Extract features from multiple texts.
        
        Args:
            texts: List of text strings
            return_array: If True, return as numpy array
            
        Returns:
            List of TextFeatures or numpy array
        """
        features_list = []
        
        for text in texts:
            try:
                features = self.extract(text)
                features_list.append(features)
            except Exception as e:
                logger.error(f"Failed to extract features: {e}")
                continue
        
        if return_array:
            return np.vstack([f.to_flat_array() for f in features_list])
        
        return features_list


def extract_text_features(
    text: str,
    use_transformers: bool = True
) -> Dict:
    """
    Convenience function to extract text features.
    
    Args:
        text: Input text
        use_transformers: Use transformers for perplexity
        
    Returns:
        Dictionary of extracted features
    """
    extractor = TextFeatureExtractor(use_transformers=use_transformers)
    features = extractor.extract(text)
    return features.to_dict()


if __name__ == "__main__":
    # Example usage
    sample_text = """
    So I was thinking about the project we discussed yesterday. 
    Um, it's actually quite interesting, you know. The main idea is 
    about detecting AI generated speech using various linguistic features.
    We can look at, like, sentence patterns and stuff. Er, let me explain 
    how it works. First, we extract the text from audio. Then we analyze 
    various patterns in the text. Finally, we use machine learning to 
    classify whether it's human or AI generated.
    """
    
    print("Text Feature Extraction")
    print("=" * 50)
    
    extractor = TextFeatureExtractor(use_transformers=True)
    features = extractor.extract(sample_text)
    
    print("\nExtracted Features:")
    for key, value in features.to_dict().items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
