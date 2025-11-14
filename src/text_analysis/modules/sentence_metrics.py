from __future__ import annotations

from typing import Any, Dict, List

from .base import AnalysisModule
from ..utils.stats import summarize_numeric


class SentenceMetrics(AnalysisModule):
    """Computes basic sentence-level statistics."""

    @property
    def name(self) -> str:
        return "sentence_stats"

    def compute(self, payload, context: Dict[str, Any]) -> Dict[str, Any]:
        sentences: List[str] = context.get("sentences", [])
        paragraphs: List[str] = context.get("paragraphs", [])
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        summary = summarize_numeric(sentence_lengths)
        return {
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "mean_sentence_length_tokens": summary["mean_value"],
            "std_sentence_length_tokens": summary["std_dev"],
            "variance_sentence_length_tokens": summary["variance"],
            "min_sentence_length_tokens": summary["min_value"],
            "max_sentence_length_tokens": summary["max_value"],
            "sentence_length_coefficient_of_variation": summary["coefficient_of_variation"],
        }
