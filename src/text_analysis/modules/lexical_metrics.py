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

        return {
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

    @staticmethod
    def _normalize_token(token: str) -> str:
        return re.sub(r"[^\w']", "", token.lower())
