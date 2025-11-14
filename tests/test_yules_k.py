#!/usr/bin/env python3
"""Test for Yule's K calculation in lexical metrics."""

import unittest
from pathlib import Path
from collections import Counter
import sys
sys.path.append('src')

from text_analysis.config_manager import AnalysisConfig
from text_analysis.modules.lexical_metrics import LexicalMetrics
from text_analysis.text_analyzer import TextPayload


class TestYulesK(unittest.TestCase):
    """Test Yule's K calculation."""
    
    def setUp(self):
        self.config = AnalysisConfig({})
        self.module = LexicalMetrics(self.config)
        self.payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="the quick brown fox jumps"
        )
    
    def test_yules_k_all_unique(self):
        """Test Yule's K with all unique words (should be 0)."""
        tokens = ["the", "quick", "brown", "fox", "jumps"]
        counter = Counter(tokens)
        result = self.module._calculate_yules_k(counter, len(tokens))
        self.assertAlmostEqual(result, 0.0, places=3)
    
    def test_yules_k_repeated_words(self):
        """Test Yule's K with repeated words."""
        tokens = ["the", "the", "the", "quick", "brown", "quick"]
        counter = Counter(tokens)
        result = self.module._calculate_yules_k(counter, len(tokens))
        # Should be positive (around 2222)
        self.assertGreater(result, 2000)
        self.assertLess(result, 3000)
    
    def test_yules_k_very_repetitive(self):
        """Test Yule's K with very repetitive text."""
        tokens = ["the"] * 5 + ["cat"]
        counter = Counter(tokens)
        result = self.module._calculate_yules_k(counter, len(tokens))
        # Should be high (around 5556)
        self.assertGreater(result, 5000)
        self.assertLess(result, 6000)
    
    def test_yules_k_empty_text(self):
        """Test Yule's K with empty text."""
        counter = Counter()
        result = self.module._calculate_yules_k(counter, 0)
        self.assertEqual(result, 0.0)
    
    def test_full_compute_includes_yules_k(self):
        """Test that full compute() includes Yule's K in results."""
        # Mock context with simple tokens
        context = {
            "tokens": ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
        }
        
        result = self.module.compute(self.payload, context)
        
        # Should include yules_k in results
        self.assertIn("yules_k", result)
        self.assertIsInstance(result["yules_k"], (int, float))
        self.assertGreaterEqual(result["yules_k"], 0)


if __name__ == "__main__":
    unittest.main()