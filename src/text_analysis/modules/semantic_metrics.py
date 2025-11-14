"""Semantic analysis module for topic coherence, drift, and semantic relationships."""

from __future__ import annotations

import math
from typing import Any
from collections import Counter

from .base import AnalysisModule
from ..utils.stats import summarize_counts


class SemanticMetrics(AnalysisModule):
    """Analyzes semantic patterns, topic coherence, and conceptual relationships."""

    @property
    def name(self) -> str:
        return "semantic_metrics"

    def compute(self, payload, context: dict[str, Any]) -> dict[str, Any]:
        """Calculate semantic coherence and topic analysis."""
        return self.analyze(context)

    def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        """Calculate semantic patterns from spaCy doc and sentences."""
        sentences = context["sentences"]
        doc = context.get("spacy_doc")
        
        if not sentences or not doc:
            return {}

        # Extract semantic features
        content_words = self._extract_content_words(doc)
        topic_coherence = self._analyze_topic_coherence(sentences, content_words)
        semantic_density = self._calculate_semantic_density(doc)
        entity_patterns = self._analyze_entities(doc)
        conceptual_overlap = self._analyze_conceptual_overlap(sentences, content_words)
        
        return {
            "topic_coherence": topic_coherence,
            "semantic_density": semantic_density,
            "entity_patterns": entity_patterns,
            "conceptual_overlap": conceptual_overlap,
            "content_word_stats": self._analyze_content_words(content_words)
        }

    def _extract_content_words(self, doc) -> list[str]:
        """Extract content words (nouns, verbs, adjectives) excluding stop words."""
        content_words = []
        
        for token in doc:
            if (not token.is_stop and 
                not token.is_punct and 
                not token.is_space and
                token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'} and
                len(token.text) > 2):
                content_words.append(token.lemma_.lower())
        
        return content_words

    def _analyze_topic_coherence(self, sentences: list[str], content_words: list[str]) -> dict[str, Any]:
        """Analyze topic coherence and drift across the text."""
        if len(sentences) < 2:
            return {"topic_drift": 0.0, "coherence_score": 1.0}
        
        # Divide text into segments for drift analysis
        segment_size = max(len(sentences) // 5, 2)  # 5 segments minimum
        segments = [
            sentences[i:i + segment_size]
            for i in range(0, len(sentences), segment_size)
        ]
        
        # Calculate word frequency for each segment
        segment_word_counts = []
        for segment in segments:
            segment_text = " ".join(segment).lower()
            words = [word for word in segment_text.split() if word in content_words]
            segment_word_counts.append(Counter(words))
        
        # Calculate pairwise similarity between adjacent segments
        drift_scores = []
        for i in range(len(segment_word_counts) - 1):
            similarity = self._calculate_jaccard_similarity(
                segment_word_counts[i], segment_word_counts[i + 1]
            )
            drift_scores.append(1.0 - similarity)  # Drift is inverse of similarity
        
        # Calculate overall coherence
        total_word_count = Counter(content_words)
        vocabulary_size = len(total_word_count)
        
        # Calculate lexical cohesion using content word repetition
        repeated_words = sum(1 for count in total_word_count.values() if count > 1)
        lexical_cohesion = repeated_words / vocabulary_size if vocabulary_size > 0 else 0.0
        
        # Topic concentration (how focused the vocabulary is)
        if content_words:
            # Calculate entropy of content word distribution
            total_content_words = len(content_words)
            word_probs = [count / total_content_words for count in total_word_count.values()]
            topic_entropy = -sum(p * math.log2(p) for p in word_probs if p > 0)
            max_entropy = math.log2(vocabulary_size) if vocabulary_size > 0 else 1
            topic_concentration = 1.0 - (topic_entropy / max_entropy) if max_entropy > 0 else 0.0
        else:
            topic_concentration = 0.0
        
        return {
            "topic_drift": sum(drift_scores) / len(drift_scores) if drift_scores else 0.0,
            "coherence_score": lexical_cohesion,
            "topic_concentration": topic_concentration,
            "vocabulary_size": vocabulary_size,
            "segment_count": len(segments)
        }

    def _calculate_jaccard_similarity(self, counter1: Counter, counter2: Counter) -> float:
        """Calculate Jaccard similarity between two word frequency counters."""
        set1 = set(counter1.keys())
        set2 = set(counter2.keys())
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0

    def _calculate_semantic_density(self, doc) -> dict[str, Any]:
        """Calculate semantic density metrics."""
        total_tokens = len([token for token in doc if not token.is_space])
        
        # Count different types of meaningful content
        content_words = sum(1 for token in doc if (
            not token.is_stop and 
            not token.is_punct and 
            not token.is_space and
            token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'}
        ))
        
        nouns = sum(1 for token in doc if token.pos_ == 'NOUN')
        verbs = sum(1 for token in doc if token.pos_ == 'VERB')
        adjectives = sum(1 for token in doc if token.pos_ == 'ADJ')
        
        # Calculate ratios
        return {
            "content_word_density": content_words / total_tokens if total_tokens > 0 else 0.0,
            "noun_density": nouns / total_tokens if total_tokens > 0 else 0.0,
            "verb_density": verbs / total_tokens if total_tokens > 0 else 0.0,
            "adjective_density": adjectives / total_tokens if total_tokens > 0 else 0.0,
            "semantic_load": (nouns + verbs + adjectives) / total_tokens if total_tokens > 0 else 0.0
        }

    def _analyze_entities(self, doc) -> dict[str, Any]:
        """Analyze named entities and their patterns."""
        entities = [ent for ent in doc.ents]
        entity_types = Counter(ent.label_ for ent in entities)
        
        # Calculate entity density and diversity
        total_tokens = len([token for token in doc if not token.is_space])
        entity_tokens = sum(len(ent.text.split()) for ent in entities)
        
        entity_diversity = len(entity_types) / len(entities) if entities else 0.0
        
        # Most common entity types
        most_common_types = dict(entity_types.most_common(5))
        
        return {
            "entity_count": len(entities),
            "entity_density": entity_tokens / total_tokens if total_tokens > 0 else 0.0,
            "entity_type_diversity": entity_diversity,
            "unique_entity_types": len(entity_types),
            "most_common_entity_types": most_common_types
        }

    def _analyze_conceptual_overlap(self, sentences: list[str], content_words: list[str]) -> dict[str, Any]:
        """Analyze conceptual overlap between sentences."""
        if len(sentences) < 2:
            return {"average_overlap": 0.0, "max_overlap": 0.0, "min_overlap": 0.0}
        
        # Calculate word sets for each sentence
        sentence_word_sets = []
        for sentence in sentences:
            words = set(word.lower() for word in sentence.split() if word.lower() in content_words)
            sentence_word_sets.append(words)
        
        # Calculate pairwise overlaps
        overlaps = []
        for i in range(len(sentence_word_sets)):
            for j in range(i + 1, len(sentence_word_sets)):
                if sentence_word_sets[i] or sentence_word_sets[j]:
                    overlap = len(sentence_word_sets[i] & sentence_word_sets[j])
                    union_size = len(sentence_word_sets[i] | sentence_word_sets[j])
                    overlap_ratio = overlap / union_size if union_size > 0 else 0.0
                    overlaps.append(overlap_ratio)
        
        if not overlaps:
            return {"average_overlap": 0.0, "max_overlap": 0.0, "min_overlap": 0.0}
        
        return {
            "average_overlap": sum(overlaps) / len(overlaps),
            "max_overlap": max(overlaps),
            "min_overlap": min(overlaps),
            "overlap_variance": self._calculate_variance(overlaps)
        }

    def _analyze_content_words(self, content_words: list[str]) -> dict[str, Any]:
        """Analyze content word statistics and diversity."""
        if not content_words:
            return {}
        
        word_counts = Counter(content_words)
        unique_words = len(word_counts)
        total_words = len(content_words)
        
        # Calculate TTR and other lexical diversity measures
        ttr = unique_words / total_words if total_words > 0 else 0.0
        
        # Calculate hapax legomena (words that occur only once)
        hapax_legomena = sum(1 for count in word_counts.values() if count == 1)
        hapax_ratio = hapax_legomena / unique_words if unique_words > 0 else 0.0
        
        # Calculate word frequency distribution
        frequency_distribution = summarize_counts(list(word_counts.values()))
        
        return {
            "unique_content_words": unique_words,
            "total_content_words": total_words,
            "content_word_ttr": ttr,
            "hapax_legomena": hapax_legomena,
            "hapax_ratio": hapax_ratio,
            "word_frequency_distribution": frequency_distribution
        }

    def _calculate_variance(self, values: list[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / (len(values) - 1)