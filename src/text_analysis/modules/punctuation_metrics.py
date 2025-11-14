"""Punctuation analysis module for orthography and style patterns."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .base import AnalysisModule
from ..utils.stats import summarize_counts


class PunctuationMetrics(AnalysisModule):
    """Analyzes punctuation patterns and orthography styles."""

    @property
    def name(self) -> str:
        return "punctuation_metrics"

    def compute(self, payload, context: dict[str, Any]) -> dict[str, Any]:
        """Calculate punctuation statistics from text and spaCy doc."""
        return self.analyze(context)

    def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        """Calculate punctuation statistics from text and spaCy doc."""
        # Get text from spaCy doc if available, otherwise reconstruct from tokens
        doc = context.get("spacy_doc")
        if doc:
            text = doc.text
        else:
            # Fallback: reconstruct text from tokens (won't be perfect but workable)
            tokens = context.get("tokens", [])
            text = " ".join(tokens)
        
        sentences = context["sentences"]
        
        # Basic punctuation counts
        punct_counts = self._count_punctuation(text)
        
        # Sentence-level punctuation analysis
        sent_punct_stats = self._analyze_sentence_punctuation(sentences)
        
        # Quotation and dialogue analysis
        quote_stats = self._analyze_quotations(text, sentences)
        
        # Capitalization patterns
        cap_stats = self._analyze_capitalization(text, doc) if doc else {}
        
        # Punctuation density and ratios
        density_stats = self._calculate_punctuation_density(text, punct_counts)
        
        return {
            "punctuation_counts": dict(punct_counts),
            "sentence_punctuation": sent_punct_stats,
            "quotation_analysis": quote_stats,
            "capitalization_patterns": cap_stats,
            "punctuation_density": density_stats
        }

    def _count_punctuation(self, text: str) -> Counter:
        """Count all punctuation marks in text."""
        punct_pattern = re.compile(r'[^\w\s]')
        punctuation_chars = punct_pattern.findall(text)
        return Counter(punctuation_chars)

    def _analyze_sentence_punctuation(self, sentences: list[str]) -> dict[str, Any]:
        """Analyze punctuation patterns at sentence level."""
        end_punct_counts = Counter()
        exclamation_sentences = 0
        question_sentences = 0
        multi_punct_sentences = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Count ending punctuation
            if sentence.endswith('!'):
                end_punct_counts['!'] += 1
                exclamation_sentences += 1
            elif sentence.endswith('?'):
                end_punct_counts['?'] += 1
                question_sentences += 1
            elif sentence.endswith('.'):
                end_punct_counts['.'] += 1
                
            # Check for multiple punctuation marks
            if re.search(r'[.!?]{2,}', sentence):
                multi_punct_sentences += 1

        total_sentences = len([s for s in sentences if s.strip()])
        
        return {
            "ending_punctuation": dict(end_punct_counts),
            "exclamation_ratio": exclamation_sentences / total_sentences if total_sentences > 0 else 0.0,
            "question_ratio": question_sentences / total_sentences if total_sentences > 0 else 0.0,
            "multiple_punctuation_ratio": multi_punct_sentences / total_sentences if total_sentences > 0 else 0.0,
            "exclamation_sentences": exclamation_sentences,
            "question_sentences": question_sentences,
            "multiple_punctuation_sentences": multi_punct_sentences
        }

    def _analyze_quotations(self, text: str, sentences: list[str]) -> dict[str, Any]:
        """Analyze quotation marks and dialogue patterns."""
        # Different quote types
        quote_patterns = {
            'double_quotes': r'"[^"]*"',
            'single_quotes': r"'[^']*'",
            'curly_double_open': r'"[^"]*"',
            'curly_single_open': r"'[^']*'"
        }
        
        quote_counts = {}
        for quote_type, pattern in quote_patterns.items():
            matches = re.findall(pattern, text)
            quote_counts[quote_type] = len(matches)
        
        # Dialogue analysis
        dialogue_sentences = 0
        for sentence in sentences:
            if re.search(r'["\'""][^"\'""]*["\'""]', sentence):
                dialogue_sentences += 1
        
        total_sentences = len([s for s in sentences if s.strip()])
        
        return {
            "quote_counts": quote_counts,
            "dialogue_sentences": dialogue_sentences,
            "dialogue_ratio": dialogue_sentences / total_sentences if total_sentences > 0 else 0.0,
            "total_quoted_passages": sum(quote_counts.values())
        }

    def _analyze_capitalization(self, text: str, doc) -> dict[str, Any]:
        """Analyze capitalization patterns using spaCy tokens."""
        if not doc:
            return {}
            
        cap_stats = {
            'all_caps_words': 0,
            'title_case_words': 0, 
            'lowercase_sentence_starts': 0,
            'proper_nouns': 0,
            'capitalized_non_proper': 0
        }
        
        sentences = list(doc.sents)
        
        for sent in sentences:
            sent_tokens = [token for token in sent if not token.is_space and not token.is_punct]
            
            if sent_tokens:
                first_token = sent_tokens[0]
                # Check if sentence starts with lowercase (unusual)
                if first_token.text.islower() and first_token.pos_ != "PROPN":
                    cap_stats['lowercase_sentence_starts'] += 1
            
            for token in sent_tokens:
                text_token = token.text
                
                # Skip single characters and punctuation
                if len(text_token) <= 1:
                    continue
                    
                if text_token.isupper():
                    cap_stats['all_caps_words'] += 1
                elif text_token.istitle():
                    if token.pos_ == "PROPN":
                        cap_stats['proper_nouns'] += 1
                    else:
                        cap_stats['title_case_words'] += 1
                elif text_token[0].isupper() and not token.pos_ == "PROPN":
                    cap_stats['capitalized_non_proper'] += 1
        
        return cap_stats

    def _calculate_punctuation_density(self, text: str, punct_counts: Counter) -> dict[str, Any]:
        """Calculate punctuation density and distribution metrics."""
        text_chars = len(text)
        total_punct = sum(punct_counts.values())
        word_count = len(text.split())
        
        # Calculate ratios
        density_stats = {
            "punctuation_density": total_punct / text_chars if text_chars > 0 else 0.0,
            "punctuation_per_word": total_punct / word_count if word_count > 0 else 0.0,
            "total_punctuation_marks": total_punct
        }
        
        # Most common punctuation
        if punct_counts:
            most_common = punct_counts.most_common(5)
            density_stats["most_common_punctuation"] = {
                mark: {"count": count, "ratio": count / total_punct}
                for mark, count in most_common
            }
        
        # Specific punctuation ratios
        key_punctuation = ['.', ',', '!', '?', ';', ':', '-', '—']
        for punct in key_punctuation:
            count = punct_counts.get(punct, 0)
            density_stats[f"{punct}_ratio"] = count / total_punct if total_punct > 0 else 0.0
            density_stats[f"{punct}_per_1000_chars"] = (count / text_chars * 1000) if text_chars > 0 else 0.0
        
        return density_stats