"""
AI Detection Pattern Module

Enhanced pattern matching specifically designed to detect AI-generated text patterns
including sycophancy, over-qualification, and formulaic responses.
"""

import re
from typing import Dict, Any, List, Set
from collections import Counter
from .base import AnalysisModule
from ..config_manager import AnalysisConfig

class AIDetectionMetric(AnalysisModule):
    """Detects patterns characteristic of AI-generated text."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        
        # AI Sycophancy patterns
        self.sycophancy_patterns = [
            r"i'?d?\s+be\s+(?:happy|glad|delighted|pleased)\s+to\s+help",
            r"i'?m?\s+(?:here\s+to\s+help|at\s+your\s+service|excited\s+to\s+assist)",
            r"(?:of\s+course|certainly|absolutely),?\s+i'?d?\s+be\s+(?:happy|glad|delighted)",
            r"(?:thank\s+you\s+for|i\s+appreciate)\s+(?:bringing\s+this\s+up|your\s+question|asking)",
            r"it'?s\s+my\s+pleasure\s+to",
            r"i\s+understand\s+your\s+(?:concern|question|point)",
            r"(?:great|excellent|wonderful|fantastic)\s+(?:question|point|idea|observation)",
            r"that'?s\s+(?:a\s+)?(?:great|excellent|wonderful|fantastic)\s+(?:question|point|idea)",
        ]
        
        # Contrastive negation patterns (AI over-qualification)
        self.contrastive_patterns = [
            r"while\s+i\s+can'?t\s+.*\s+i\s+can",
            r"although\s+i\s+don'?t\s+.*\s+i\s+do",
            r"i\s+can'?t\s+.*\s+but\s+i\s+can",
            r"i'?m?\s+not\s+.*\s+but\s+i\s+am",
            r"i\s+don'?t\s+.*\s+however\s+i",
            r"i\s+won'?t\s+.*\s+but\s+i\s+will",
            r"it'?s\s+not\s+.*\s+but\s+it\s+is",
            r"this\s+isn'?t\s+.*\s+but\s+this\s+is",
            r"rather\s+than\s+.*\s+instead",
            r"not\s+only\s+.*\s+but\s+also",
        ]
        
        # Hedging and over-qualification patterns
        self.hedging_patterns = [
            r"it\s+(?:seems|appears)\s+that",
            r"it\s+(?:might|could|may)\s+be\s+that",
            r"it'?s\s+(?:worth\s+noting|important\s+to\s+note|worth\s+mentioning)",
            r"it\s+should\s+be\s+noted\s+that",
            r"(?:generally|typically|usually)\s+speaking",
            r"in\s+(?:most\s+cases|many\s+instances)",
            r"(?:potentially|possibly|perhaps)\s+",
            r"to\s+some\s+extent",
            r"in\s+a\s+sense",
            r"from\s+(?:one\s+perspective|a\s+certain\s+viewpoint)",
        ]
        
        # Excessive structuring language
        self.structure_patterns = [
            r"here\s+(?:are|is)\s+.*\s+ways?\s+to",
            r"here'?s\s+how\s+you\s+can",
            r"let\s+me\s+(?:break\s+this\s+down|explain|walk\s+you\s+through)",
            r"(?:to\s+summarize|in\s+summary|in\s+conclusion)",
            r"(?:moving|going)\s+forward",
            r"(?:next\s+steps|action\s+items|key\s+takeaways)",
            r"(?:first|second|third|finally),",
            r"on\s+the\s+one\s+hand.*on\s+the\s+other\s+hand",
            r"step\s+\d+:",
            r"option\s+\d+:",
        ]
        
        # Meta-commentary patterns (AI talking about its process)
        self.meta_patterns = [
            r"i\s+understand\s+(?:you'?re\s+asking|your\s+question)",
            r"i\s+see\s+what\s+you'?re\s+(?:looking\s+for|getting\s+at)",
            r"based\s+on\s+your\s+question",
            r"from\s+what\s+i\s+understand",
            r"if\s+i\s+understand\s+correctly",
            r"to\s+answer\s+your\s+question",
            r"regarding\s+your\s+(?:inquiry|request|question)",
            r"in\s+response\s+to\s+your",
            r"as\s+you\s+mentioned",
            r"building\s+on\s+what\s+you\s+said",
        ]
        
        # Formulaic response patterns
        self.formulaic_patterns = [
            r"(?:great|excellent|good)\s+question",
            r"thanks?\s+for\s+asking",
            r"i'?m?\s+happy\s+to\s+clarify",
            r"let\s+me\s+help\s+you\s+with\s+that",
            r"here'?s\s+what\s+you\s+need\s+to\s+know",
            r"the\s+short\s+answer\s+is",
            r"the\s+key\s+thing\s+to\s+remember\s+is",
            r"the\s+bottom\s+line\s+is",
            r"in\s+a\s+nutshell",
            r"to\s+put\s+it\s+simply",
        ]
        
        # Em-dash overuse (AI characteristic)
        self.punctuation_patterns = {
            'em_dashes': r'—',
            'double_dashes': r'--',
            'parenthetical_dashes': r'\s+—\s+.*?\s+—\s+',
            'excessive_colons': r':(?=\s+[A-Z])',  # Colons followed by capitalized explanations
            'semicolon_overuse': r';',
            'parenthetical_excess': r'\([^)]{20,}\)',  # Long parenthetical remarks
        }

    @property
    def name(self) -> str:
        return "ai_detection"

    def compute(self, payload, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute AI detection metrics.
        
        Args:
            payload: TextPayload containing the text to analyze
            context: Additional context for analysis
            
        Returns:
            Dictionary containing AI detection metrics
        """
        return self.analyze_ai_patterns(payload.content, **context)

    def count_pattern_matches(self, text: str, patterns: List[str], name: str) -> Dict[str, Any]:
        """Count matches for a specific pattern category."""
        text_lower = text.lower()
        matches = []
        total_matches = 0
        
        for pattern in patterns:
            pattern_matches = re.findall(pattern, text_lower, re.IGNORECASE)
            total_matches += len(pattern_matches)
            if pattern_matches:
                matches.extend(pattern_matches[:3])  # Keep first 3 examples
        
        return {
            f'{name}_count': total_matches,
            f'{name}_examples': matches[:5]  # Top 5 examples
        }

    def analyze_punctuation_tells(self, text: str) -> Dict[str, Any]:
        """Analyze punctuation patterns characteristic of AI."""
        results = {}
        word_count = len(text.split())
        
        for pattern_name, pattern in self.punctuation_patterns.items():
            matches = len(re.findall(pattern, text))
            density = (matches / word_count * 1000) if word_count > 0 else 0
            results[f'{pattern_name}_count'] = matches
            results[f'{pattern_name}_density'] = round(density, 2)
        
        return results

    def calculate_ai_likelihood_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall AI likelihood score based on detected patterns."""
        # Weights for different AI indicators (these could be tuned based on research)
        weights = {
            'sycophancy_count': 0.15,
            'contrastive_count': 0.12,
            'hedging_count': 0.10,
            'meta_count': 0.10,
            'formulaic_count': 0.10,
            'structure_count': 0.08,
            'em_dashes_density': 0.05,
            'excessive_colons_density': 0.05,
            'parenthetical_excess_count': 0.05,
        }
        
        score = 0.0
        max_possible_score = sum(weights.values())
        
        for indicator, weight in weights.items():
            value = metrics.get(indicator, 0)
            
            # Normalize different metrics to 0-1 scale
            if 'density' in indicator:
                normalized_value = min(value / 10.0, 1.0)  # Density values
            elif 'count' in indicator:
                normalized_value = min(value / 5.0, 1.0)   # Count values
            else:
                normalized_value = min(value, 1.0)
            
            score += normalized_value * weight
        
        # Normalize to 0-1 scale
        return min(score / max_possible_score, 1.0)

    def analyze_ai_patterns(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Analyze AI-characteristic patterns in text.
        
        Args:
            text: Text to analyze
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing AI detection metrics
        """
        if not text or not text.strip():
            return self._empty_result()
        
        # Count different types of AI patterns
        sycophancy_data = self.count_pattern_matches(text, self.sycophancy_patterns, 'sycophancy')
        contrastive_data = self.count_pattern_matches(text, self.contrastive_patterns, 'contrastive')
        hedging_data = self.count_pattern_matches(text, self.hedging_patterns, 'hedging')
        structure_data = self.count_pattern_matches(text, self.structure_patterns, 'structure')
        meta_data = self.count_pattern_matches(text, self.meta_patterns, 'meta')
        formulaic_data = self.count_pattern_matches(text, self.formulaic_patterns, 'formulaic')
        
        # Analyze punctuation tells
        punctuation_data = self.analyze_punctuation_tells(text)
        
        # Combine all metrics
        all_metrics = {
            **sycophancy_data,
            **contrastive_data,
            **hedging_data,
            **structure_data,
            **meta_data,
            **formulaic_data,
            **punctuation_data
        }
        
        # Calculate overall AI likelihood score
        ai_likelihood = self.calculate_ai_likelihood_score(all_metrics)
        
        # Add summary metrics
        total_ai_patterns = (
            all_metrics.get('sycophancy_count', 0) +
            all_metrics.get('contrastive_count', 0) +
            all_metrics.get('hedging_count', 0) +
            all_metrics.get('structure_count', 0) +
            all_metrics.get('meta_count', 0) +
            all_metrics.get('formulaic_count', 0)
        )
        
        word_count = len(text.split())
        pattern_density = (total_ai_patterns / word_count * 100) if word_count > 0 else 0
        
        all_metrics.update({
            'ai_likelihood_score': round(ai_likelihood, 3),
            'total_ai_patterns': total_ai_patterns,
            'ai_pattern_density': round(pattern_density, 2),
            'word_count_analyzed': word_count,
            'confidence_level': self._get_confidence_level(ai_likelihood)
        })
        
        return all_metrics

    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level description based on AI likelihood score."""
        if score >= 0.8:
            return "High confidence AI-generated"
        elif score >= 0.6:
            return "Likely AI-generated"
        elif score >= 0.4:
            return "Uncertain (mixed signals)"
        elif score >= 0.2:
            return "Likely human-written"
        else:
            return "High confidence human-written"

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'sycophancy_count': 0,
            'sycophancy_examples': [],
            'contrastive_count': 0,
            'contrastive_examples': [],
            'hedging_count': 0,
            'hedging_examples': [],
            'structure_count': 0,
            'structure_examples': [],
            'meta_count': 0,
            'meta_examples': [],
            'formulaic_count': 0,
            'formulaic_examples': [],
            'em_dashes_count': 0,
            'em_dashes_density': 0.0,
            'double_dashes_count': 0,
            'double_dashes_density': 0.0,
            'parenthetical_dashes_count': 0,
            'parenthetical_dashes_density': 0.0,
            'excessive_colons_count': 0,
            'excessive_colons_density': 0.0,
            'semicolon_overuse_count': 0,
            'semicolon_overuse_density': 0.0,
            'parenthetical_excess_count': 0,
            'parenthetical_excess_density': 0.0,
            'ai_likelihood_score': 0.0,
            'total_ai_patterns': 0,
            'ai_pattern_density': 0.0,
            'word_count_analyzed': 0,
            'confidence_level': 'No patterns detected'
        }

    def get_interpretation(self, metrics: Dict[str, Any]) -> str:
        """
        Provide human-readable interpretation of AI detection metrics.
        
        Args:
            metrics: Dictionary of calculated metrics
            
        Returns:
            Interpretation string
        """
        score = metrics.get('ai_likelihood_score', 0)
        confidence = metrics.get('confidence_level', 'Unknown')
        density = metrics.get('ai_pattern_density', 0)
        
        # Risk assessment
        if score >= 0.8:
            risk = "High AI risk"
        elif score >= 0.6:
            risk = "Moderate AI risk"
        elif score >= 0.4:
            risk = "Low AI risk"
        else:
            risk = "Minimal AI risk"
        
        return f"{risk}, {confidence}, {density:.1f}% pattern density"