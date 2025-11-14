"""
Discourse Flow Analysis Module

This module analyzes discourse patterns, transition usage, and text flow coherence.
Focuses on how ideas are connected and presented throughout the text.
"""

import re
from typing import Dict, Any, List, Set, Tuple
from collections import Counter, defaultdict
from .base import AnalysisModule
from ..config_manager import AnalysisConfig

class DiscourseFlowMetric(AnalysisModule):
    """Analyzes discourse patterns and text flow coherence."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        
        # Transition word categories
        self.transitions = {
            'additive': ['also', 'furthermore', 'moreover', 'additionally', 'besides', 'in addition', 
                        'what is more', 'as well', 'too', 'and', 'plus'],
            'causal': ['because', 'since', 'as', 'due to', 'owing to', 'thanks to', 'as a result', 
                      'consequently', 'therefore', 'thus', 'hence', 'so', 'accordingly'],
            'contrastive': ['however', 'nevertheless', 'nonetheless', 'on the other hand', 'in contrast',
                           'whereas', 'while', 'but', 'yet', 'although', 'though', 'despite', 'instead',
                           'rather', 'alternatively', 'on the contrary'],
            'temporal': ['first', 'second', 'then', 'next', 'later', 'finally', 'meanwhile', 'subsequently',
                        'previously', 'before', 'after', 'during', 'while', 'when', 'until', 'since'],
            'exemplification': ['for example', 'for instance', 'such as', 'namely', 'specifically',
                              'in particular', 'that is', 'i.e.', 'e.g.', 'including'],
            'emphasis': ['indeed', 'in fact', 'certainly', 'clearly', 'obviously', 'undoubtedly',
                        'without doubt', 'of course', 'naturally', 'surely', 'definitely'],
            'summary': ['in conclusion', 'to conclude', 'in summary', 'to summarize', 'overall',
                       'in brief', 'all in all', 'in short', 'finally', 'lastly']
        }
        
        # Paragraph opening patterns
        self.opening_patterns = {
            'question': r'^\s*(?:what|why|how|when|where|who|which|can|could|would|should|do|does|did|is|are|was|were)',
            'statement': r'^\s*(?:the|this|that|these|those|a|an)',
            'temporal': r'^\s*(?:first|second|then|next|now|today|yesterday|tomorrow|recently|currently)',
            'personal': r'^\s*(?:i|we|you|my|our|your)',
            'contrast': r'^\s*(?:however|but|yet|although|though|despite|on the other hand)',
            'causal': r'^\s*(?:because|since|as|therefore|thus|consequently|as a result)'
        }
        
        # Discourse markers for coherence
        self.coherence_markers = {
            'referential': ['this', 'that', 'these', 'those', 'it', 'they', 'such', 'the former', 'the latter'],
            'lexical_chains': [],  # Will be populated dynamically
            'conjunctive': ['and', 'but', 'or', 'nor', 'for', 'yet', 'so']
        }

    @property
    def name(self) -> str:
        return "discourse_flow"

    def compute(self, payload, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute discourse flow metrics.
        
        Args:
            payload: TextPayload containing the text to analyze
            context: Additional context for analysis
            
        Returns:
            Dictionary containing discourse flow metrics
        """
        return self.analyze_discourse(payload.content, **context)

    def analyze_transitions(self, text: str) -> Dict[str, Any]:
        """Analyze transition word usage and distribution."""
        text_lower = text.lower()
        
        transition_counts = {}
        total_transitions = 0
        
        for category, words in self.transitions.items():
            count = 0
            for word in words:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(word) + r'\b'
                count += len(re.findall(pattern, text_lower))
            
            transition_counts[category] = count
            total_transitions += count
        
        # Calculate transition density (per 100 words)
        word_count = len(text.split())
        transition_density = (total_transitions / word_count * 100) if word_count > 0 else 0
        
        # Find most used transition category
        most_used_category = max(transition_counts.items(), key=lambda x: x[1])[0] if total_transitions > 0 else 'none'
        
        return {
            'transition_counts': transition_counts,
            'total_transitions': total_transitions,
            'transition_density': round(transition_density, 2),
            'most_used_transition_type': most_used_category,
            'transition_distribution': {k: round(v/total_transitions*100, 1) if total_transitions > 0 else 0 
                                     for k, v in transition_counts.items()}
        }

    def analyze_paragraph_structure(self, text: str) -> Dict[str, Any]:
        """Analyze paragraph opening patterns and structure."""
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        if not paragraphs:
            return {'paragraph_count': 0, 'opening_patterns': {}, 'avg_paragraph_length': 0}
        
        opening_pattern_counts = {pattern: 0 for pattern in self.opening_patterns.keys()}
        paragraph_lengths = []
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            paragraph_lengths.append(len(paragraph.split()))
            
            # Check opening patterns
            for pattern_name, pattern_regex in self.opening_patterns.items():
                if re.match(pattern_regex, paragraph_lower):
                    opening_pattern_counts[pattern_name] += 1
                    break
        
        avg_paragraph_length = sum(paragraph_lengths) / len(paragraph_lengths) if paragraph_lengths else 0
        
        return {
            'paragraph_count': len(paragraphs),
            'opening_patterns': opening_pattern_counts,
            'avg_paragraph_length': round(avg_paragraph_length, 1),
            'paragraph_length_variation': round(
                (max(paragraph_lengths) - min(paragraph_lengths)) / avg_paragraph_length if avg_paragraph_length > 0 else 0, 2
            )
        }

    def analyze_coherence(self, text: str) -> Dict[str, Any]:
        """Analyze text coherence through referential and lexical chains."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        if not sentences:
            return {'referential_density': 0, 'lexical_chain_strength': 0, 'topic_consistency': 0}
        
        # Count referential markers
        referential_count = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for marker in self.coherence_markers['referential']:
                referential_count += len(re.findall(r'\b' + re.escape(marker) + r'\b', sentence_lower))
        
        # Calculate referential density
        total_words = len(text.split())
        referential_density = (referential_count / total_words * 100) if total_words > 0 else 0
        
        # Analyze lexical chains (repeated content words)
        content_words = []
        for sentence in sentences:
            words = re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower())  # 4+ letter words
            content_words.extend(words)
        
        # Remove common function words
        function_words = {'this', 'that', 'they', 'them', 'their', 'there', 'then', 'when', 'what', 'where',
                         'which', 'while', 'with', 'would', 'will', 'were', 'was', 'very', 'much', 'more',
                         'most', 'many', 'make', 'made', 'like', 'just', 'into', 'have', 'been', 'from'}
        
        content_words = [w for w in content_words if w not in function_words]
        
        # Calculate lexical chain strength (repetition of content words)
        word_freq = Counter(content_words)
        repeated_words = sum(1 for count in word_freq.values() if count > 1)
        lexical_chain_strength = (repeated_words / len(set(content_words))) if content_words else 0
        
        # Topic consistency (how many sentences contain core topic words)
        if word_freq:
            top_words = [word for word, count in word_freq.most_common(5)]
            sentences_with_topic = 0
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in top_words):
                    sentences_with_topic += 1
            
            topic_consistency = sentences_with_topic / len(sentences)
        else:
            topic_consistency = 0
        
        return {
            'referential_density': round(referential_density, 2),
            'lexical_chain_strength': round(lexical_chain_strength, 3),
            'topic_consistency': round(topic_consistency, 3),
            'content_word_repetition': repeated_words,
            'unique_content_words': len(set(content_words))
        }

    def analyze_flow_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze overall discourse flow patterns."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        if len(sentences) < 2:
            return {'flow_consistency': 0, 'topic_shifts': 0, 'discourse_complexity': 0}
        
        # Analyze sentence connections
        connected_sentences = 0
        transition_words_all = []
        for category, words in self.transitions.items():
            transition_words_all.extend(words)
        
        for i in range(1, len(sentences)):
            sentence_lower = sentences[i].lower()
            
            # Check if sentence starts with transition or reference
            if any(sentence_lower.startswith(word) for word in transition_words_all[:20]):  # Check common ones
                connected_sentences += 1
            elif any(ref in sentence_lower[:20] for ref in self.coherence_markers['referential'][:5]):
                connected_sentences += 1
        
        flow_consistency = connected_sentences / (len(sentences) - 1) if len(sentences) > 1 else 0
        
        # Estimate topic shifts (simplified)
        # Look for abrupt changes in vocabulary between adjacent sentences
        topic_shifts = 0
        for i in range(1, len(sentences)):
            prev_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', sentences[i-1].lower()))
            curr_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', sentences[i].lower()))
            
            if prev_words and curr_words:
                overlap = len(prev_words & curr_words) / len(prev_words | curr_words)
                if overlap < 0.1:  # Very little overlap suggests topic shift
                    topic_shifts += 1
        
        topic_shift_rate = topic_shifts / (len(sentences) - 1) if len(sentences) > 1 else 0
        
        # Discourse complexity (variety of discourse markers)
        complexity_score = len(set(re.findall(r'\b(?:' + '|'.join(transition_words_all[:30]) + r')\b', text.lower())))
        
        return {
            'flow_consistency': round(flow_consistency, 3),
            'topic_shift_rate': round(topic_shift_rate, 3),
            'discourse_complexity': complexity_score,
            'connected_sentences': connected_sentences,
            'total_sentence_pairs': len(sentences) - 1
        }

    def analyze_discourse(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Analyze discourse flow patterns.
        
        Args:
            text: Text to analyze
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing discourse flow metrics
        """
        if not text or not text.strip():
            return self._empty_result()
        
        # Analyze transitions
        transition_data = self.analyze_transitions(text)
        
        # Analyze paragraph structure
        paragraph_data = self.analyze_paragraph_structure(text)
        
        # Analyze coherence
        coherence_data = self.analyze_coherence(text)
        
        # Analyze flow patterns
        flow_data = self.analyze_flow_patterns(text)
        
        # Combine all metrics
        return {
            **transition_data,
            **paragraph_data,
            **coherence_data,
            **flow_data,
            'discourse_score': round((
                flow_data['flow_consistency'] * 0.3 +
                coherence_data['topic_consistency'] * 0.3 +
                coherence_data['lexical_chain_strength'] * 0.2 +
                (transition_data['transition_density'] / 10) * 0.2  # Normalize to 0-1 scale
            ), 3)
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'transition_counts': {k: 0 for k in self.transitions.keys()},
            'total_transitions': 0,
            'transition_density': 0.0,
            'most_used_transition_type': 'none',
            'transition_distribution': {k: 0 for k in self.transitions.keys()},
            'paragraph_count': 0,
            'opening_patterns': {k: 0 for k in self.opening_patterns.keys()},
            'avg_paragraph_length': 0.0,
            'paragraph_length_variation': 0.0,
            'referential_density': 0.0,
            'lexical_chain_strength': 0.0,
            'topic_consistency': 0.0,
            'content_word_repetition': 0,
            'unique_content_words': 0,
            'flow_consistency': 0.0,
            'topic_shift_rate': 0.0,
            'discourse_complexity': 0,
            'connected_sentences': 0,
            'total_sentence_pairs': 0,
            'discourse_score': 0.0
        }

    def get_interpretation(self, metrics: Dict[str, Any]) -> str:
        """
        Provide human-readable interpretation of discourse flow metrics.
        
        Args:
            metrics: Dictionary of calculated metrics
            
        Returns:
            Interpretation string
        """
        score = metrics.get('discourse_score', 0)
        flow = metrics.get('flow_consistency', 0)
        coherence = metrics.get('topic_consistency', 0)
        
        # Overall discourse quality
        if score > 0.7:
            quality = "Excellent"
        elif score > 0.5:
            quality = "Good"
        elif score > 0.3:
            quality = "Moderate"
        else:
            quality = "Poor"
        
        # Flow assessment
        if flow > 0.6:
            flow_level = "Well-connected"
        elif flow > 0.3:
            flow_level = "Moderately connected"
        else:
            flow_level = "Disconnected"
        
        # Coherence assessment
        if coherence > 0.7:
            coherence_level = "Highly coherent"
        elif coherence > 0.5:
            coherence_level = "Coherent"
        else:
            coherence_level = "Fragmented"
        
        return f"{quality} discourse flow, {flow_level}, {coherence_level}"