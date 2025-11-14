"""Rhythm and variability analysis module for temporal patterns and text flow."""

from __future__ import annotations

import math
from typing import Any
import statistics

from .base import AnalysisModule
from ..utils.stats import summarize_numeric


class RhythmMetrics(AnalysisModule):
    """Analyzes rhythm, variability, and temporal patterns in text structure."""

    @property
    def name(self) -> str:
        return "rhythm_metrics"

    def compute(self, payload, context: dict[str, Any]) -> dict[str, Any]:
        """Calculate rhythm and variability statistics."""
        return self.analyze(context)

    def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        """Calculate rhythm patterns from sentences and structure."""
        sentences = context["sentences"]
        doc = context.get("spacy_doc")
        
        if not sentences:
            return {}

        # Sentence-level rhythm analysis
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        sentence_char_lengths = [len(sentence) for sentence in sentences]
        
        sentence_rhythm = self._analyze_sentence_rhythm(sentence_lengths)
        character_rhythm = self._analyze_character_rhythm(sentence_char_lengths)
        
        # Paragraph-level analysis (if available)
        paragraphs = self._extract_paragraphs(context.get("content", ""))
        paragraph_analysis = self._analyze_paragraph_patterns(paragraphs)
        
        # Autocorrelation analysis
        autocorr_analysis = self._analyze_autocorrelation(sentence_lengths)
        
        # Burstiness and spectral analysis
        burstiness = self._calculate_burstiness(sentence_lengths)
        spectral_features = self._analyze_spectral_features(sentence_lengths)
        
        # Temporal pattern analysis
        temporal_patterns = self._analyze_temporal_patterns(sentences)

        return {
            "sentence_rhythm": sentence_rhythm,
            "character_rhythm": character_rhythm,
            "paragraph_analysis": paragraph_analysis,
            "autocorrelation": autocorr_analysis,
            "burstiness": burstiness,
            "spectral_features": spectral_features,
            "temporal_patterns": temporal_patterns
        }

    def _analyze_sentence_rhythm(self, lengths: list[int]) -> dict[str, Any]:
        """Analyze rhythm patterns in sentence word lengths."""
        if not lengths:
            return {}
        
        stats = summarize_numeric(lengths)
        
        # Calculate rhythm-specific metrics
        range_span = max(lengths) - min(lengths) if len(lengths) > 1 else 0
        
        # Coefficient of variation (normalized variability)
        cv = stats["std_dev"] / stats["mean_value"] if stats["mean_value"] > 0 else 0.0
        
        # Sequential variability (average absolute difference between consecutive sentences)
        if len(lengths) > 1:
            sequential_diffs = [abs(lengths[i] - lengths[i-1]) for i in range(1, len(lengths))]
            avg_sequential_diff = sum(sequential_diffs) / len(sequential_diffs)
            max_sequential_diff = max(sequential_diffs)
        else:
            avg_sequential_diff = 0.0
            max_sequential_diff = 0.0
        
        # Pattern regularity (how consistently patterned the lengths are)
        if len(lengths) >= 3:
            pattern_score = self._calculate_pattern_regularity(lengths)
        else:
            pattern_score = 0.0

        return {
            **stats,
            "coefficient_of_variation": cv,
            "range_span": range_span,
            "average_sequential_difference": avg_sequential_diff,
            "max_sequential_difference": max_sequential_diff,
            "pattern_regularity": pattern_score
        }

    def _analyze_character_rhythm(self, char_lengths: list[int]) -> dict[str, Any]:
        """Analyze rhythm patterns in sentence character lengths."""
        if not char_lengths:
            return {}
        
        stats = summarize_numeric(char_lengths)
        
        # Character-specific rhythm metrics
        cv = stats["std_dev"] / stats["mean_value"] if stats["mean_value"] > 0 else 0.0
        
        return {
            "char_mean_length": stats["mean_value"],
            "char_std_dev": stats["std_dev"],
            "char_coefficient_of_variation": cv,
            "char_min_length": stats["min_value"],
            "char_max_length": stats["max_value"]
        }

    def _extract_paragraphs(self, content: str) -> list[str]:
        """Extract paragraphs from content."""
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        return paragraphs if paragraphs else [content]

    def _analyze_paragraph_patterns(self, paragraphs: list[str]) -> dict[str, Any]:
        """Analyze paragraph-level structure patterns."""
        if not paragraphs:
            return {}
        
        para_lengths = [len(p.split()) for p in paragraphs]
        para_sentence_counts = [len([s for s in p.split('.') if s.strip()]) for p in paragraphs]
        
        para_stats = summarize_numeric(para_lengths) if para_lengths else {}
        sentence_count_stats = summarize_numeric(para_sentence_counts) if para_sentence_counts else {}
        
        return {
            "paragraph_count": len(paragraphs),
            "avg_words_per_paragraph": para_stats.get("mean_value", 0),
            "paragraph_length_variation": para_stats.get("coefficient_of_variation", 0),
            "avg_sentences_per_paragraph": sentence_count_stats.get("mean_value", 0),
            "paragraph_sentence_variation": sentence_count_stats.get("coefficient_of_variation", 0)
        }

    def _analyze_autocorrelation(self, lengths: list[int]) -> dict[str, Any]:
        """Analyze autocorrelation in sentence length patterns."""
        if len(lengths) < 3:
            return {"lag_1_autocorrelation": 0.0, "lag_2_autocorrelation": 0.0}
        
        def autocorrelation_at_lag(data: list[int], lag: int) -> float:
            if len(data) <= lag:
                return 0.0
            
            mean_val = statistics.mean(data)
            
            # Calculate autocorrelation
            numerator = 0.0
            denominator = 0.0
            
            for i in range(len(data) - lag):
                numerator += (data[i] - mean_val) * (data[i + lag] - mean_val)
            
            for i in range(len(data)):
                denominator += (data[i] - mean_val) ** 2
            
            return numerator / denominator if denominator > 0 else 0.0
        
        lag_1 = autocorrelation_at_lag(lengths, 1)
        lag_2 = autocorrelation_at_lag(lengths, 2) if len(lengths) >= 4 else 0.0
        
        return {
            "lag_1_autocorrelation": lag_1,
            "lag_2_autocorrelation": lag_2
        }

    def _calculate_burstiness(self, lengths: list[int]) -> dict[str, Any]:
        """Calculate burstiness measures (temporal clustering patterns)."""
        if len(lengths) < 2:
            return {"burstiness_coefficient": 0.0, "memory_coefficient": 0.0}
        
        # Burstiness coefficient B = (σ - μ) / (σ + μ)
        # where σ is standard deviation and μ is mean
        mean_length = statistics.mean(lengths)
        std_length = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
        
        if mean_length + std_length > 0:
            burstiness = (std_length - mean_length) / (std_length + mean_length)
        else:
            burstiness = 0.0
        
        # Memory coefficient (correlation between consecutive intervals)
        if len(lengths) >= 3:
            memory = self._analyze_autocorrelation(lengths)["lag_1_autocorrelation"]
        else:
            memory = 0.0
        
        # Gini coefficient for inequality in length distribution
        gini = self._calculate_gini_coefficient(lengths)
        
        return {
            "burstiness_coefficient": burstiness,
            "memory_coefficient": memory,
            "gini_coefficient": gini
        }

    def _calculate_gini_coefficient(self, values: list[int]) -> float:
        """Calculate Gini coefficient for measuring inequality."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        if n == 0:
            return 0.0
        
        # Calculate Gini coefficient
        index = range(1, n + 1)
        gini = (2 * sum(index[i] * sorted_values[i] for i in range(n))) / (n * sum(sorted_values)) - (n + 1) / n
        
        return gini

    def _analyze_spectral_features(self, lengths: list[int]) -> dict[str, Any]:
        """Analyze frequency domain features of length sequences."""
        if len(lengths) < 4:
            return {"dominant_frequency": 0.0, "spectral_entropy": 0.0}
        
        # Simple spectral analysis using discrete differences
        # Calculate power in different "frequency" bands
        
        # High-frequency component (rapid changes)
        high_freq_power = 0.0
        for i in range(1, len(lengths)):
            high_freq_power += (lengths[i] - lengths[i-1]) ** 2
        high_freq_power /= len(lengths) - 1
        
        # Mid-frequency component (medium-term trends)
        mid_freq_power = 0.0
        window_size = min(3, len(lengths) // 2)
        if window_size > 0:
            for i in range(window_size, len(lengths)):
                window_mean = statistics.mean(lengths[i-window_size:i])
                mid_freq_power += (lengths[i] - window_mean) ** 2
            mid_freq_power /= len(lengths) - window_size
        
        # Calculate spectral entropy (measure of frequency diversity)
        total_power = high_freq_power + mid_freq_power
        if total_power > 0:
            p_high = high_freq_power / total_power
            p_mid = mid_freq_power / total_power
            spectral_entropy = -(p_high * math.log2(p_high + 1e-10) + p_mid * math.log2(p_mid + 1e-10))
        else:
            spectral_entropy = 0.0
        
        return {
            "high_frequency_power": high_freq_power,
            "mid_frequency_power": mid_freq_power,
            "spectral_entropy": spectral_entropy
        }

    def _calculate_pattern_regularity(self, lengths: list[int]) -> float:
        """Calculate how regular/predictable the length pattern is."""
        if len(lengths) < 3:
            return 0.0
        
        # Calculate how well the sequence follows a simple pattern
        # by measuring prediction error using moving average
        prediction_errors = []
        
        for i in range(2, len(lengths)):
            # Simple prediction: average of previous two values
            predicted = (lengths[i-1] + lengths[i-2]) / 2
            actual = lengths[i]
            error = abs(predicted - actual)
            prediction_errors.append(error)
        
        if not prediction_errors:
            return 0.0
        
        # Normalize by the mean length to get relative regularity
        mean_length = statistics.mean(lengths)
        if mean_length > 0:
            avg_error = statistics.mean(prediction_errors)
            regularity = max(0.0, 1.0 - (avg_error / mean_length))
        else:
            regularity = 0.0
        
        return regularity

    def _analyze_temporal_patterns(self, sentences: list[str]) -> dict[str, Any]:
        """Analyze temporal flow and transition patterns."""
        if len(sentences) < 2:
            return {}
        
        # Analyze sentence beginnings for temporal markers
        temporal_markers = {
            'sequence': ['first', 'second', 'third', 'next', 'then', 'finally', 'last'],
            'time': ['when', 'while', 'during', 'after', 'before', 'since', 'until'],
            'causality': ['because', 'since', 'therefore', 'thus', 'hence', 'so'],
            'contrast': ['however', 'but', 'yet', 'nevertheless', 'although', 'despite']
        }
        
        marker_counts = {category: 0 for category in temporal_markers.keys()}
        total_sentences = len(sentences)
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            words = sentence_lower.split()
            
            if words:
                first_word = words[0]
                first_two_words = ' '.join(words[:2]) if len(words) >= 2 else first_word
                
                for category, markers in temporal_markers.items():
                    for marker in markers:
                        if first_word == marker or first_two_words.startswith(marker):
                            marker_counts[category] += 1
                            break
        
        # Calculate ratios and flow characteristics
        results = {}
        for category, count in marker_counts.items():
            results[f"{category}_marker_ratio"] = count / total_sentences if total_sentences > 0 else 0.0
        
        # Overall temporal flow score
        total_temporal_markers = sum(marker_counts.values())
        results["temporal_flow_density"] = total_temporal_markers / total_sentences if total_sentences > 0 else 0.0
        
        return results