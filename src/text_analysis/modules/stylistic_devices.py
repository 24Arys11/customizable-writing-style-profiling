"""
Stylistic Device Detection Module

This module detects various stylistic devices and rhetorical patterns
including repetition, alliteration, parallel structures, and metaphorical language.
"""

import re
from typing import Dict, Any, List, Set, Tuple
from collections import Counter
from .base import AnalysisModule
from ..config_manager import AnalysisConfig

class StylisticDeviceMetric(AnalysisModule):
    """Detects stylistic devices and rhetorical patterns."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        
        # Metaphor/comparison indicators
        self.metaphor_patterns = [
            r'\bis\s+(?:like|as)\s+',  # simile patterns
            r'\b(?:seems?|appears?|feels?)\s+(?:like|as)\s+',
            r'\b(?:reminds?\s+me\s+of|brings\s+to\s+mind)\b',
            r'\bmetaphor(?:ically)?\b',
            r'\bsymbol(?:iz|is)es?\b',
            r'\brepresents?\b',
            r'\bembod(?:ies|y)\b'
        ]
        
        # Repetition patterns for rhetorical effect
        self.repetition_patterns = [
            r'\b(\w+)\s+\1\b',  # immediate repetition
            r'\b(\w{4,})\b.*?\b\1\b',  # word repetition within sentence
        ]
        
        # Parallel structure indicators
        self.parallel_markers = [
            r'\b(?:not\s+only|neither|either)\b.*?\b(?:but\s+(?:also)?|nor|or)\b',
            r'\b(?:first|second|third|finally|lastly|in\s+conclusion)\b',
            r'\b(?:on\s+one\s+hand|on\s+the\s+other\s+hand|meanwhile|likewise|similarly)\b',
            r'\b(?:both|whether)\b.*?\b(?:and|or)\b'
        ]
        
        # Rhetorical devices
        self.rhetorical_patterns = [
            r'\b(?:imagine|picture\s+this|consider|suppose)\b',  # hypothetical scenarios
            r'\b(?:what\s+if|but\s+what\s+about|how\s+about)\b',  # rhetorical questions starters
            r'\b(?:clearly|obviously|certainly|undoubtedly|without\s+question)\b',  # emphasis
            r'\b(?:in\s+fact|indeed|actually|really)\b',  # assertion intensifiers
        ]
        
        # Emotional intensifiers
        self.intensifier_patterns = [
            r'\b(?:very|extremely|incredibly|amazingly|absolutely|completely|totally)\b',
            r'\b(?:never|always|forever|constantly|continuously)\b',
            r'\b(?:all|every|each|any|no)\s+(?:single|one|person|thing)\b',
        ]
        
        # Literary devices
        self.literary_devices = {
            'alliteration': r'\b([a-zA-Z])\w*\s+\1\w*(?:\s+\1\w*)*\b',
            'anaphora': r'^(\w+(?:\s+\w+)?)\s+.*?\n.*?^\1\s+',  # repetition at start of lines
            'epistrophe': r'(\w+(?:\s+\w+)?)\s*\.?\s*\n.*?\1\s*\.?\s*$',  # repetition at end of lines
        }

    @property
    def name(self) -> str:
        return "stylistic_devices"

    def compute(self, payload, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute stylistic device metrics.
        
        Args:
            payload: TextPayload containing the text to analyze
            context: Additional context for analysis
            
        Returns:
            Dictionary containing stylistic device metrics
        """
        return self.analyze_devices(payload.content, **context)

    def detect_alliteration(self, text: str) -> List[str]:
        """Detect alliterative phrases."""
        alliterations = []
        pattern = self.literary_devices['alliteration']
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            alliterations.append(match.group().strip())
        
        return alliterations

    def detect_repetition(self, text: str) -> Dict[str, Any]:
        """Detect various forms of repetition."""
        sentences = re.split(r'[.!?]+', text)
        
        # Word repetition within sentences
        word_repetition_count = 0
        repeated_words = []
        
        for pattern in self.repetition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            word_repetition_count += len(matches)
            repeated_words.extend(matches)
        
        # Phrase repetition across sentences
        phrase_repetition = 0
        common_phrases = []
        
        # Look for 3+ word phrases that repeat
        for sentence in sentences:
            if len(sentence.strip()) > 10:
                words = sentence.strip().split()
                for i in range(len(words) - 2):
                    phrase = ' '.join(words[i:i+3]).lower()
                    if text.lower().count(phrase) > 1:
                        phrase_repetition += 1
                        if phrase not in common_phrases:
                            common_phrases.append(phrase)
        
        return {
            'word_repetition_count': word_repetition_count,
            'phrase_repetition_count': phrase_repetition,
            'repeated_words': repeated_words[:10],  # top 10 for brevity
            'repeated_phrases': common_phrases[:5]  # top 5 for brevity
        }

    def detect_parallel_structures(self, text: str) -> int:
        """Detect parallel grammatical structures."""
        parallel_count = 0
        
        for pattern in self.parallel_markers:
            matches = re.findall(pattern, text, re.IGNORECASE)
            parallel_count += len(matches)
        
        return parallel_count

    def detect_rhetorical_devices(self, text: str) -> Dict[str, int]:
        """Detect various rhetorical devices."""
        device_counts = {}
        
        # Metaphor/simile detection
        metaphor_count = 0
        for pattern in self.metaphor_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            metaphor_count += len(matches)
        device_counts['metaphors_similes'] = metaphor_count
        
        # Rhetorical techniques
        rhetorical_count = 0
        for pattern in self.rhetorical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            rhetorical_count += len(matches)
        device_counts['rhetorical_techniques'] = rhetorical_count
        
        # Intensifiers
        intensifier_count = 0
        for pattern in self.intensifier_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            intensifier_count += len(matches)
        device_counts['intensifiers'] = intensifier_count
        
        return device_counts

    def analyze_sentence_variety(self, text: str) -> Dict[str, Any]:
        """Analyze sentence structure variety."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        if not sentences:
            return {'sentence_variety_score': 0, 'avg_sentence_complexity': 0}
        
        # Analyze sentence beginnings
        sentence_starts = []
        for sentence in sentences:
            words = sentence.strip().split()
            if words:
                first_word = words[0].lower()
                sentence_starts.append(first_word)
        
        # Calculate variety in sentence openings
        unique_starts = len(set(sentence_starts))
        total_sentences = len(sentence_starts)
        variety_score = unique_starts / total_sentences if total_sentences > 0 else 0
        
        # Analyze sentence complexity (presence of conjunctions, subordinate clauses)
        complex_markers = ['because', 'although', 'while', 'whereas', 'since', 'unless', 'until', 'before', 'after']
        complex_sentences = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(marker in sentence_lower for marker in complex_markers):
                complex_sentences += 1
        
        complexity_ratio = complex_sentences / len(sentences) if sentences else 0
        
        return {
            'sentence_variety_score': round(variety_score, 3),
            'avg_sentence_complexity': round(complexity_ratio, 3),
            'unique_sentence_starts': unique_starts,
            'total_sentences_analyzed': total_sentences
        }

    def analyze_devices(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Analyze stylistic devices in the text.
        
        Args:
            text: Text to analyze
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing stylistic device metrics
        """
        if not text or not text.strip():
            return self._empty_result()
        
        # Detect alliteration
        alliterations = self.detect_alliteration(text)
        
        # Detect repetition patterns
        repetition_data = self.detect_repetition(text)
        
        # Detect parallel structures
        parallel_count = self.detect_parallel_structures(text)
        
        # Detect rhetorical devices
        rhetorical_devices = self.detect_rhetorical_devices(text)
        
        # Analyze sentence variety
        sentence_analysis = self.analyze_sentence_variety(text)
        
        # Calculate text length for normalization
        word_count = len(text.split())
        
        return {
            'alliteration_instances': len(alliterations),
            'alliterative_phrases': alliterations[:5],  # top 5 examples
            'word_repetition_count': repetition_data['word_repetition_count'],
            'phrase_repetition_count': repetition_data['phrase_repetition_count'],
            'parallel_structures': parallel_count,
            'metaphors_similes': rhetorical_devices.get('metaphors_similes', 0),
            'rhetorical_techniques': rhetorical_devices.get('rhetorical_techniques', 0),
            'intensifiers': rhetorical_devices.get('intensifiers', 0),
            'sentence_variety_score': sentence_analysis['sentence_variety_score'],
            'sentence_complexity_ratio': sentence_analysis['avg_sentence_complexity'],
            'stylistic_density': round((len(alliterations) + repetition_data['word_repetition_count'] + 
                                     parallel_count + rhetorical_devices.get('metaphors_similes', 0)) / 
                                    max(word_count / 100, 1), 2),
            'word_count_analyzed': word_count
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'alliteration_instances': 0,
            'alliterative_phrases': [],
            'word_repetition_count': 0,
            'phrase_repetition_count': 0,
            'parallel_structures': 0,
            'metaphors_similes': 0,
            'rhetorical_techniques': 0,
            'intensifiers': 0,
            'sentence_variety_score': 0.0,
            'sentence_complexity_ratio': 0.0,
            'stylistic_density': 0.0,
            'word_count_analyzed': 0
        }

    def get_interpretation(self, metrics: Dict[str, Any]) -> str:
        """
        Provide human-readable interpretation of stylistic device metrics.
        
        Args:
            metrics: Dictionary of calculated metrics
            
        Returns:
            Interpretation string
        """
        density = metrics.get('stylistic_density', 0)
        variety = metrics.get('sentence_variety_score', 0)
        complexity = metrics.get('sentence_complexity_ratio', 0)
        
        # Style richness assessment
        if density > 2.0:
            richness = "Rich"
        elif density > 1.0:
            richness = "Moderate"
        else:
            richness = "Plain"
        
        # Sentence variety assessment
        if variety > 0.7:
            variety_level = "High variety"
        elif variety > 0.5:
            variety_level = "Moderate variety"
        else:
            variety_level = "Low variety"
        
        # Complexity assessment
        if complexity > 0.3:
            comp_level = "Complex"
        elif complexity > 0.15:
            comp_level = "Moderate"
        else:
            comp_level = "Simple"
        
        return f"{richness} stylistic devices, {variety_level}, {comp_level} sentences"