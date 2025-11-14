"""Syntax analysis module using spaCy POS tags and dependency parsing."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .base import AnalysisModule
from ..utils.stats import summarize_counts


class SyntaxMetrics(AnalysisModule):
    """Analyzes syntactic patterns using POS tags and dependency relationships."""

    @property
    def name(self) -> str:
        return "syntax_metrics"

    def compute(self, payload, context: dict[str, Any]) -> dict[str, Any]:
        """Calculate syntax statistics from spaCy doc."""
        return self.analyze(context)

    def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        """Calculate syntax statistics from spaCy doc."""
        doc = context.get("spacy_doc")
        if not doc:
            return {}

        sentences = context["sentences"]
        
        # POS tag analysis
        pos_counts = Counter()
        pos_per_sentence = []
        
        # Dependency relation analysis  
        dep_counts = Counter()
        dep_per_sentence = []
        
        # Syntactic complexity metrics
        sentence_depths = []
        
        for sent in doc.sents:
            sent_pos = Counter()
            sent_deps = Counter()
            
            # Calculate dependency tree depth for this sentence
            max_depth = self._calculate_tree_depth(sent)
            sentence_depths.append(max_depth)
            
            for token in sent:
                if not token.is_space:
                    pos_counts[token.pos_] += 1
                    sent_pos[token.pos_] += 1
                    
                    dep_counts[token.dep_] += 1
                    sent_deps[token.dep_] += 1
            
            pos_per_sentence.append(dict(sent_pos))
            dep_per_sentence.append(dict(sent_deps))

        # Calculate statistics for key POS categories
        pos_stats = self._calculate_pos_statistics(pos_counts, sentences)
        dep_stats = self._calculate_dep_statistics(dep_counts, sentences)
        complexity_stats = self._calculate_complexity_statistics(sentence_depths, sentences)
        
        return {
            "pos_distribution": dict(pos_counts),
            "pos_statistics": pos_stats,
            "dependency_distribution": dict(dep_counts),
            "dependency_statistics": dep_stats,
            "syntactic_complexity": complexity_stats,
            # Internal data for debugging/research (not displayed in summary/table)
            "_debug": {
                "per_sentence_pos": pos_per_sentence,
                "per_sentence_dependencies": dep_per_sentence
            }
        }

    def _calculate_tree_depth(self, sent) -> int:
        """Calculate maximum dependency tree depth for a sentence."""
        def get_depth(token, current_depth=0):
            if not list(token.children):
                return current_depth
            return max(get_depth(child, current_depth + 1) for child in token.children)
        
        # Find the root token
        roots = [token for token in sent if token.dep_ == "ROOT"]
        if not roots:
            return 0
        
        return max(get_depth(root) for root in roots)

    def _calculate_pos_statistics(self, pos_counts: Counter, sentences: list[str]) -> dict[str, Any]:
        """Calculate statistics for important POS categories."""
        total_tokens = sum(pos_counts.values())
        sentence_count = len(sentences)
        
        # Key grammatical categories
        categories = {
            "nouns": ["NOUN", "PROPN"],
            "verbs": ["VERB", "AUX"], 
            "adjectives": ["ADJ"],
            "adverbs": ["ADV"],
            "pronouns": ["PRON"],
            "determiners": ["DET"],
            "prepositions": ["ADP"],
            "conjunctions": ["CCONJ", "SCONJ"],
            "punctuation": ["PUNCT"]
        }
        
        stats = {}
        for category, pos_tags in categories.items():
            count = sum(pos_counts.get(tag, 0) for tag in pos_tags)
            if total_tokens > 0:
                stats[f"{category}_ratio"] = count / total_tokens
                stats[f"{category}_per_sentence"] = count / sentence_count
            else:
                stats[f"{category}_ratio"] = 0.0
                stats[f"{category}_per_sentence"] = 0.0
            stats[f"{category}_total"] = count
            
        return stats

    def _calculate_dep_statistics(self, dep_counts: Counter, sentences: list[str]) -> dict[str, Any]:
        """Calculate statistics for dependency relationships."""
        total_deps = sum(dep_counts.values())
        sentence_count = len(sentences)
        
        # Important dependency relations
        key_deps = {
            "subjects": ["nsubj", "nsubjpass", "csubj", "csubjpass"],
            "objects": ["dobj", "iobj", "pobj"],
            "modifiers": ["amod", "advmod", "det"],
            "clauses": ["relcl", "advcl", "ccomp", "xcomp"],
            "conjunctions": ["conj", "cc"]
        }
        
        stats = {}
        for category, dep_labels in key_deps.items():
            count = sum(dep_counts.get(label, 0) for label in dep_labels)
            if total_deps > 0:
                stats[f"{category}_ratio"] = count / total_deps
                stats[f"{category}_per_sentence"] = count / sentence_count
            else:
                stats[f"{category}_ratio"] = 0.0
                stats[f"{category}_per_sentence"] = 0.0
            stats[f"{category}_total"] = count
            
        return stats

    def _calculate_complexity_statistics(self, depths: list[int], sentences: list[str]) -> dict[str, Any]:
        """Calculate syntactic complexity metrics."""
        if not depths:
            return {
                "mean_tree_depth": 0.0,
                "std_tree_depth": 0.0,
                "variance_tree_depth": 0.0,
                "max_tree_depth": 0,
                "min_tree_depth": 0,
                "depth_coefficient_of_variation": 0.0
            }
        
        stats = summarize_counts(depths)
        return {
            "mean_tree_depth": stats["mean_per_unit"],
            "std_tree_depth": stats["std_dev_per_unit"], 
            "variance_tree_depth": stats["variance"],
            "max_tree_depth": max(depths),
            "min_tree_depth": min(depths),
            "depth_coefficient_of_variation": stats["coefficient_of_variation"]
        }