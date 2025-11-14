"""
Vocabulary Sophistication Analysis Module

This module analyzes the sophistication level of vocabulary used in text,
including syllable complexity, abstract vs concrete word ratios, and register indicators.
"""

import re
from typing import Dict, Any, List, Set
from .base import AnalysisModule
from ..config_manager import AnalysisConfig

class VocabularySophisticationMetric(AnalysisModule):
    """Analyzes vocabulary sophistication and complexity patterns."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        
        # Abstract vs concrete word indicators
        self.abstract_indicators = {
            'concept', 'idea', 'notion', 'theory', 'principle', 'belief', 'thought',
            'philosophy', 'ideology', 'perspective', 'viewpoint', 'understanding',
            'knowledge', 'wisdom', 'intelligence', 'consciousness', 'awareness',
            'reality', 'truth', 'existence', 'meaning', 'purpose', 'significance',
            'importance', 'value', 'worth', 'quality', 'nature', 'essence',
            'freedom', 'justice', 'equality', 'democracy', 'liberty', 'rights',
            'responsibility', 'duty', 'obligation', 'morality', 'ethics', 'virtue',
            'progress', 'development', 'evolution', 'change', 'transformation',
            'potential', 'possibility', 'opportunity', 'challenge', 'problem'
        }
        
        # Formal register indicators
        self.formal_indicators = {
            'furthermore', 'moreover', 'nonetheless', 'consequently', 'therefore',
            'thus', 'hence', 'accordingly', 'nevertheless', 'however', 'whereas',
            'notwithstanding', 'albeit', 'pursuant', 'aforementioned', 'heretofore',
            'henceforth', 'whereby', 'wherein', 'thereof', 'thereof', 'herein',
            'constitute', 'establish', 'demonstrate', 'indicate', 'suggest',
            'propose', 'examine', 'analyze', 'evaluate', 'assess', 'determine',
            'utilize', 'implement', 'facilitate', 'endeavor', 'ascertain'
        }
        
        # Informal/colloquial indicators
        self.informal_indicators = {
            'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'yeah', 'nah', 'yep',
            'nope', 'ok', 'okay', 'cool', 'awesome', 'great', 'nice', 'good',
            'bad', 'stuff', 'things', 'guys', 'folks', 'dude', 'man', 'like',
            'you know', 'i mean', 'basically', 'actually', 'really', 'pretty',
            'quite', 'very', 'super', 'totally', 'absolutely', 'definitely',
            'probably', 'maybe', 'perhaps', 'guess', 'think', 'feel', 'seems'
        }
        
        # Academic/technical vocabulary
        self.academic_indicators = {
            'hypothesis', 'methodology', 'paradigm', 'phenomena', 'criterion',
            'empirical', 'theoretical', 'analytical', 'systematic', 'comprehensive',
            'fundamental', 'substantial', 'significant', 'considerable', 'extensive',
            'inherent', 'intrinsic', 'extrinsic', 'implicit', 'explicit',
            'preliminary', 'subsequent', 'concurrent', 'corresponding', 'respective',
            'comparative', 'alternative', 'potential', 'optimal', 'adequate',
            'appropriate', 'relevant', 'consistent', 'distinct', 'specific'
        }

    @property
    def name(self) -> str:
        return "vocabulary_sophistication"

    def compute(self, payload, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute vocabulary sophistication metrics.
        
        Args:
            payload: TextPayload containing the text to analyze
            context: Additional context for analysis
            
        Returns:
            Dictionary containing vocabulary sophistication metrics
        """
        return self.analyze(payload.content, **context)

    def calculate_syllables(self, word: str) -> int:
        """
        Estimate syllable count using heuristic rules.
        
        Args:
            word: Word to count syllables for
            
        Returns:
            Estimated syllable count
        """
        word = word.lower().strip()
        if not word:
            return 0
        
        # Remove common suffixes that don't add syllables
        word = re.sub(r'(ed|es|s)$', '', word)
        
        # Count vowel groups
        vowel_groups = re.findall(r'[aeiouy]+', word)
        syllable_count = len(vowel_groups)
        
        # Adjust for silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        # Minimum of 1 syllable
        return max(1, syllable_count)

    def analyze(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Analyze vocabulary sophistication patterns.
        
        Args:
            text: Text to analyze
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing vocabulary sophistication metrics
        """
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        if not words:
            return self._empty_result()
        
        # Calculate syllable metrics
        syllable_counts = [self.calculate_syllables(word) for word in words]
        avg_syllables = sum(syllable_counts) / len(syllable_counts)
        complex_words = sum(1 for count in syllable_counts if count >= 3)
        complex_word_ratio = complex_words / len(words)
        
        # Calculate word length metrics
        word_lengths = [len(word) for word in words]
        avg_word_length = sum(word_lengths) / len(word_lengths)
        long_words = sum(1 for length in word_lengths if length >= 7)
        long_word_ratio = long_words / len(words)
        
        # Analyze register (formal vs informal)
        formal_count = sum(1 for word in words if word in self.formal_indicators)
        informal_count = sum(1 for word in words if word in self.informal_indicators)
        academic_count = sum(1 for word in words if word in self.academic_indicators)
        
        formal_ratio = formal_count / len(words)
        informal_ratio = informal_count / len(words)
        academic_ratio = academic_count / len(words)
        
        # Calculate formality score
        formality_score = (formal_ratio + academic_ratio - informal_ratio)
        
        # Analyze abstract vs concrete vocabulary
        abstract_count = sum(1 for word in words if word in self.abstract_indicators)
        abstract_ratio = abstract_count / len(words)
        
        # Vocabulary diversity (unique sophisticated words)
        unique_words = set(words)
        sophisticated_words = {
            word for word in unique_words 
            if self.calculate_syllables(word) >= 3 or len(word) >= 7
        }
        sophistication_ratio = len(sophisticated_words) / len(unique_words) if unique_words else 0
        
        return {
            'avg_syllables_per_word': round(avg_syllables, 2),
            'complex_word_ratio': round(complex_word_ratio, 3),
            'avg_word_length': round(avg_word_length, 2),
            'long_word_ratio': round(long_word_ratio, 3),
            'formality_score': round(formality_score, 3),
            'formal_word_ratio': round(formal_ratio, 3),
            'informal_word_ratio': round(informal_ratio, 3),
            'academic_word_ratio': round(academic_ratio, 3),
            'abstract_word_ratio': round(abstract_ratio, 3),
            'vocabulary_sophistication': round(sophistication_ratio, 3),
            'total_words_analyzed': len(words)
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'avg_syllables_per_word': 0.0,
            'complex_word_ratio': 0.0,
            'avg_word_length': 0.0,
            'long_word_ratio': 0.0,
            'formality_score': 0.0,
            'formal_word_ratio': 0.0,
            'informal_word_ratio': 0.0,
            'academic_word_ratio': 0.0,
            'abstract_word_ratio': 0.0,
            'vocabulary_sophistication': 0.0,
            'total_words_analyzed': 0
        }

    def get_interpretation(self, metrics: Dict[str, Any]) -> str:
        """
        Provide human-readable interpretation of vocabulary sophistication metrics.
        
        Args:
            metrics: Dictionary of calculated metrics
            
        Returns:
            Interpretation string
        """
        sophistication = metrics.get('vocabulary_sophistication', 0)
        formality = metrics.get('formality_score', 0)
        complexity = metrics.get('complex_word_ratio', 0)
        
        # Sophistication level
        if sophistication > 0.4:
            soph_level = "High"
        elif sophistication > 0.25:
            soph_level = "Moderate" 
        else:
            soph_level = "Basic"
        
        # Formality level
        if formality > 0.01:
            form_level = "Formal"
        elif formality < -0.01:
            form_level = "Informal"
        else:
            form_level = "Neutral"
        
        # Complexity level
        if complexity > 0.2:
            comp_level = "Complex"
        elif complexity > 0.1:
            comp_level = "Moderate"
        else:
            comp_level = "Simple"
        
        return f"{soph_level} sophistication, {form_level} register, {comp_level} vocabulary"