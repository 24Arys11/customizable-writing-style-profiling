from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List

from .base import AnalysisModule
from ..utils.stats import summarize_counts


class WordListMetrics(AnalysisModule):
    """Computes counts and dispersion for configured word lists."""

    @property
    def name(self) -> str:
        return "word_list_metrics"

    def compute(self, payload, context: Dict[str, Any]) -> Dict[str, Any]:
        config_lists = self._config.raw.get("word_lists", {})
        tokens = [self._normalize_token(token) for token in context.get("tokens", [])]
        sentence_tokens = [
            [self._normalize_token(token) for token in sentence.split()]
            for sentence in context.get("sentences", [])
        ]
        token_total = len(tokens)
        normalization_basis = self._config.raw.get("metrics", {}).get("normalization_basis")

        token_counter = Counter(tokens)
        results: Dict[str, Any] = {}

        for list_name, entry in config_lists.items():
            terms = self._extract_terms(entry)
            if not terms:
                continue
            term_set = set(terms)

            per_sentence_counts = [
                sum(1 for token in sentence if token in term_set)
                for sentence in sentence_tokens
            ]
            stats = summarize_counts(
                per_sentence_counts,
                total_reference=token_total,
                normalization_basis=normalization_basis,
            )

            term_counts = {term: token_counter.get(term, 0) for term in terms}

            list_report = {
                "description": entry.get("description") if isinstance(entry, dict) else None,
                "terms": terms,
                "term_counts": term_counts,
                "total_occurrences": stats["total"],
                "mean_occurrences_per_sentence": stats["mean_per_unit"],
                "std_dev_occurrences_per_sentence": stats["std_dev_per_unit"],
                "variance_occurrences_per_sentence": stats["variance"],
                "occurrence_coefficient_of_variation": stats["coefficient_of_variation"],
            }

            frequency_key = next(
                (key for key in stats.keys() if key.startswith("frequency_per_")),
                None,
            )
            if frequency_key:
                list_report[frequency_key] = stats[frequency_key]

            results[list_name] = list_report

        return results

    @staticmethod
    def _normalize_token(token: str) -> str:
        return re.sub(r"[^\w']", "", token.lower())

    @staticmethod
    def _extract_terms(entry: Any) -> List[str]:
        if isinstance(entry, dict):
            terms = entry.get("terms") or entry.get("words") or []
        elif isinstance(entry, list):
            terms = entry
        else:
            terms = []
        return [term.lower() for term in terms if term]
