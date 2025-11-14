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
        sentence_summary = summarize_numeric(sentence_lengths)
        
        # Paragraph length analysis
        paragraph_lengths_words = [len(paragraph.split()) for paragraph in paragraphs if paragraph.strip()]
        paragraph_lengths_sentences = []
        
        # Calculate sentences per paragraph
        if paragraphs:
            # Simple approximation: count sentence-ending punctuation
            for paragraph in paragraphs:
                if paragraph.strip():
                    sentence_endings = paragraph.count('.') + paragraph.count('!') + paragraph.count('?')
                    paragraph_lengths_sentences.append(max(1, sentence_endings))  # At least 1 sentence
        
        paragraph_word_summary = summarize_numeric(paragraph_lengths_words) if paragraph_lengths_words else summarize_numeric([])
        paragraph_sentence_summary = summarize_numeric(paragraph_lengths_sentences) if paragraph_lengths_sentences else summarize_numeric([])
        
        return {
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "mean_sentence_length_tokens": sentence_summary["mean_value"],
            "std_sentence_length_tokens": sentence_summary["std_dev"],
            "variance_sentence_length_tokens": sentence_summary["variance"],
            "min_sentence_length_tokens": sentence_summary["min_value"],
            "max_sentence_length_tokens": sentence_summary["max_value"],
            "sentence_length_coefficient_of_variation": sentence_summary["coefficient_of_variation"],
            
            # Paragraph analysis
            "paragraph_length_mean_words": paragraph_word_summary["mean_value"],
            "paragraph_length_std_words": paragraph_word_summary["std_dev"],
            "paragraph_length_coefficient_of_variation_words": paragraph_word_summary["coefficient_of_variation"],
            "paragraph_length_min_words": paragraph_word_summary["min_value"],
            "paragraph_length_max_words": paragraph_word_summary["max_value"],
            
            "paragraph_length_mean_sentences": paragraph_sentence_summary["mean_value"],
            "paragraph_length_std_sentences": paragraph_sentence_summary["std_dev"],
            "paragraph_length_coefficient_of_variation_sentences": paragraph_sentence_summary["coefficient_of_variation"],
        }
