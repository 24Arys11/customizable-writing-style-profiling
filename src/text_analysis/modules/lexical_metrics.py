from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict

from .base import AnalysisModule
from ..utils.stats import summarize_numeric


class LexicalMetrics(AnalysisModule):
    """Computes lexical richness and distribution statistics."""

    @property
    def name(self) -> str:
        return "lexical_metrics"

    def compute(self, payload, context: Dict[str, Any]) -> Dict[str, Any]:
        tokens = [self._normalize_token(token) for token in context.get("tokens", []) if token]
        token_count = len(tokens)
        if token_count == 0:
            empty_stats = summarize_numeric([])
            return {
                "token_count": 0,
                "type_count": 0,
                "type_token_ratio": 0.0,
                "hapax_ratio": 0.0,
                "word_length_mean_characters": empty_stats["mean_value"],
                "word_length_std_dev_characters": empty_stats["std_dev"],
                "word_length_variance_characters_squared": empty_stats["variance"],
                "word_length_min_characters": empty_stats["min_value"],
                "word_length_max_characters": empty_stats["max_value"],
            }

        counter = Counter(tokens)
        type_count = len(counter)
        ttr = type_count / token_count if token_count else 0.0
        hapax = sum(1 for word, freq in counter.items() if freq == 1)
        hapax_ratio = hapax / token_count if token_count else 0.0

        word_lengths = [len(word) for word in tokens if word]
        length_stats = summarize_numeric(word_lengths)

        # Advanced lexical diversity metrics
        yules_k = self._calculate_yules_k(counter, token_count)
        simpsons_d = self._calculate_simpsons_d(counter, token_count)
        
        # Character n-grams analysis
        char_ngrams = self._analyze_character_ngrams(tokens)
        
        # Function word analysis
        function_word_analysis = self._analyze_function_words(tokens)

        base_metrics = {
            "token_count": token_count,
            "type_count": type_count,
            "type_token_ratio": ttr,
            "hapax_ratio": hapax_ratio,
            "word_length_mean_characters": length_stats["mean_value"],
            "word_length_std_dev_characters": length_stats["std_dev"],
            "word_length_variance_characters_squared": length_stats["variance"],
            "word_length_min_characters": length_stats["min_value"],
            "word_length_max_characters": length_stats["max_value"],
        }
        
        # Combine all metrics
        advanced_metrics = {
            "yules_k": yules_k,
            "simpsons_d": simpsons_d,
            **char_ngrams,
            **function_word_analysis
        }
        
        return {**base_metrics, **advanced_metrics}

    @staticmethod
    def _normalize_token(token: str) -> str:
        return re.sub(r"[^\w']", "", token.lower())

    def _calculate_yules_k(self, counter: Counter, token_count: int) -> float:
        """Calculate Yule's K diversity index."""
        if token_count == 0:
            return 0.0
        
        # Yule's K = 10^4 * (sum(i^2 * V_i) - N) / N^2
        # where V_i = number of word types with frequency i, N = total tokens
        frequency_distribution = Counter(counter.values())
        
        # Sum of (frequency^2 * count of words with that frequency)
        sum_freq_squared = sum(freq * freq * count for freq, count in frequency_distribution.items())
        yules_k = 10000 * (sum_freq_squared - token_count) / (token_count * token_count)
        
        return yules_k

    def _calculate_simpsons_d(self, counter: Counter, token_count: int) -> float:
        """Calculate Simpson's D diversity index."""
        if token_count == 0:
            return 0.0
        
        # Simpson's D = sum((n_i * (n_i - 1)) / (N * (N - 1)))
        # where n_i = frequency of word type i, N = total tokens
        if token_count == 1:
            return 0.0
        
        sum_n_squared = sum(freq * (freq - 1) for freq in counter.values())
        simpsons_d = sum_n_squared / (token_count * (token_count - 1))
        
        return simpsons_d

    def _analyze_character_ngrams(self, tokens: list[str]) -> dict[str, Any]:
        """Analyze character n-gram patterns."""
        if not tokens:
            return {
                "char_bigram_diversity": 0.0,
                "char_trigram_diversity": 0.0,
                "average_char_bigrams_per_word": 0.0,
                "average_char_trigrams_per_word": 0.0
            }
        
        all_bigrams = []
        all_trigrams = []
        
        for token in tokens:
            if len(token) >= 2:
                bigrams = [token[i:i+2] for i in range(len(token) - 1)]
                all_bigrams.extend(bigrams)
            
            if len(token) >= 3:
                trigrams = [token[i:i+3] for i in range(len(token) - 2)]
                all_trigrams.extend(trigrams)
        
        # Calculate diversity (unique n-grams / total n-grams)
        bigram_diversity = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0.0
        trigram_diversity = len(set(all_trigrams)) / len(all_trigrams) if all_trigrams else 0.0
        
        return {
            "char_bigram_diversity": bigram_diversity,
            "char_trigram_diversity": trigram_diversity,
            "average_char_bigrams_per_word": len(all_bigrams) / len(tokens) if tokens else 0.0,
            "average_char_trigrams_per_word": len(all_trigrams) / len(tokens) if tokens else 0.0
        }

    def _analyze_function_words(self, tokens: list[str]) -> dict[str, Any]:
        """Analyze function word usage patterns."""
        # Common function words (determiners, prepositions, pronouns, conjunctions)
        function_words = {
            'determiners': {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their'},
            'prepositions': {'in', 'on', 'at', 'by', 'for', 'with', 'to', 'of', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among'},
            'pronouns': {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves'},
            'conjunctions': {'and', 'or', 'but', 'if', 'when', 'while', 'because', 'since', 'although', 'though', 'unless', 'until', 'wherever', 'whenever'},
            'auxiliaries': {'be', 'am', 'is', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'may', 'might', 'can', 'could', 'must'}
        }
        
        if not tokens:
            return {f"{category}_ratio": 0.0 for category in function_words.keys()}
        
        token_count = len(tokens)
        normalized_tokens = [self._normalize_token(token) for token in tokens]
        
        results = {}
        for category, word_set in function_words.items():
            count = sum(1 for token in normalized_tokens if token in word_set)
            results[f"{category}_ratio"] = count / token_count
        
        # Calculate overall function word ratio
        all_function_words = set()
        for word_set in function_words.values():
            all_function_words.update(word_set)
        
        total_function_words = sum(1 for token in normalized_tokens if token in all_function_words)
        results["total_function_word_ratio"] = total_function_words / token_count
        
        return results
