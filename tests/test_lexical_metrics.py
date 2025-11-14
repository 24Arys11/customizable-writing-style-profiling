"""Comprehensive tests for lexical metrics calculations."""
import unittest
from pathlib import Path
from collections import Counter

from src.text_analysis.config_manager import AnalysisConfig
from src.text_analysis.modules.lexical_metrics import LexicalMetrics
from src.text_analysis.text_analyzer import TextPayload


class TestLexicalMetrics(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test configuration."""
        self.config = AnalysisConfig(raw={
            "metrics": {"normalization_basis": 1000}
        })
        self.module = LexicalMetrics(self.config)

    def test_type_token_ratio_basic(self) -> None:
        """Test basic type-token ratio calculation."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="the cat sat on the mat"
        )
        context = {
            "tokens": ["the", "cat", "sat", "on", "the", "mat"],
            "word_frequencies": Counter(["the", "cat", "sat", "on", "the", "mat"])
        }
        
        result = self.module.compute(payload, context)
        
        # 5 unique words / 6 total tokens = 0.8333...
        expected_ttr = 5 / 6
        self.assertAlmostEqual(result["type_token_ratio"], expected_ttr, places=4)

    def test_type_token_ratio_edge_cases(self) -> None:
        """Test type-token ratio edge cases."""
        # All unique words
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="cat dog bird fish"
        )
        context = {
            "tokens": ["cat", "dog", "bird", "fish"],
            "word_frequencies": Counter(["cat", "dog", "bird", "fish"])
        }
        
        result = self.module.compute(payload, context)
        self.assertAlmostEqual(result["type_token_ratio"], 1.0, places=4)

        # All same word
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="the the the the"
        )
        context = {
            "tokens": ["the", "the", "the", "the"],
            "word_frequencies": Counter(["the", "the", "the", "the"])
        }
        
        result = self.module.compute(payload, context)
        self.assertAlmostEqual(result["type_token_ratio"], 0.25, places=4)

        # Empty text
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content=""
        )
        context = {
            "tokens": [],
            "word_frequencies": Counter()
        }
        
        result = self.module.compute(payload, context)
        self.assertEqual(result["type_token_ratio"], 0.0)

    def test_yules_k_calculation(self) -> None:
        """Test Yule's K calculation with known values."""
        # Test case: words with frequency 1: [a, b], frequency 2: [c], frequency 3: [d]
        # Expected: sum(i² × V_i) = 1²×2 + 2²×1 + 3²×1 = 2 + 4 + 9 = 15
        # N = 7 total tokens
        # K = 10000 × (15 - 7) / 49 = 10000 × 8 / 49 = 1632.65
        
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="a b c c d d d"
        )
        
        word_freq = Counter(["a", "b", "c", "c", "d", "d", "d"])
        context = {
            "tokens": ["a", "b", "c", "c", "d", "d", "d"],
            "word_frequencies": word_freq
        }
        
        result = self.module.compute(payload, context)
        
        # Manual calculation verification
        freq_of_freq = {}
        for word, freq in word_freq.items():
            freq_of_freq[freq] = freq_of_freq.get(freq, 0) + 1
        
        sum_freq_squared = sum(i * i * count for i, count in freq_of_freq.items())
        n = len(context["tokens"])
        expected_k = 10000 * (sum_freq_squared - n) / (n * n)
        
        self.assertAlmostEqual(result["yules_k"], expected_k, places=2)
        # Updated expected value based on correct calculation
        self.assertAlmostEqual(result["yules_k"], 1632.65, places=1)

    def test_yules_k_edge_cases(self) -> None:
        """Test Yule's K edge cases."""
        # All words unique - should give K = 0
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="each word appears once only here"
        )
        context = {
            "tokens": ["each", "word", "appears", "once", "only", "here"],
            "word_frequencies": Counter(["each", "word", "appears", "once", "only", "here"])
        }
        
        result = self.module.compute(payload, context)
        self.assertAlmostEqual(result["yules_k"], 0.0, places=2)

        # Very repetitive text
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="the the the the the the the the the"
        )
        context = {
            "tokens": ["the"] * 9,
            "word_frequencies": Counter(["the"] * 9)
        }
        
        result = self.module.compute(payload, context)
        # 1 word appearing 9 times: sum = 9² × 1 = 81, N = 9
        # K = 10000 × (81 - 9) / 81 = 10000 × 72 / 81 ≈ 8888.89
        expected_k = 10000 * (81 - 9) / 81
        self.assertAlmostEqual(result["yules_k"], expected_k, places=2)

        # Empty text - should not have yules_k in result (early return)
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content=""
        )
        context = {
            "tokens": [],
            "word_frequencies": Counter()
        }
        
        result = self.module.compute(payload, context)
        # yules_k is not included in empty text results, that's expected behavior
        self.assertEqual(result["token_count"], 0)

    def test_average_word_length(self) -> None:
        """Test average word length calculation."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="cat elephant dog"
        )
        context = {
            "tokens": ["cat", "elephant", "dog"]
        }
        
        result = self.module.compute(payload, context)
        
        # cat=3, elephant=8, dog=3 -> (3+8+3)/3 = 4.67
        expected_avg = (3 + 8 + 3) / 3
        self.assertAlmostEqual(result["word_length_mean_characters"], expected_avg, places=2)

    def test_average_word_length_edge_cases(self) -> None:
        """Test average word length edge cases."""
        # Empty text
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content=""
        )
        context = {
            "tokens": []
        }
        
        result = self.module.compute(payload, context)
        self.assertEqual(result["word_length_mean_characters"], 0.0)

        # Single character words
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="a b c"
        )
        context = {
            "tokens": ["a", "b", "c"]
        }
        
        result = self.module.compute(payload, context)
        self.assertAlmostEqual(result["word_length_mean_characters"], 1.0, places=2)

    def test_lexical_diversity_measures(self) -> None:
        """Test lexical diversity related measures."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="the quick brown fox jumps over the lazy dog"
        )
        context = {
            "tokens": ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
            "word_frequencies": Counter(["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"])
        }
        
        result = self.module.compute(payload, context)
        
        # Verify all expected metrics are present
        self.assertIn("type_token_ratio", result)
        self.assertIn("yules_k", result)
        self.assertIn("word_length_mean_characters", result)
        
        # Basic sanity checks
        self.assertGreaterEqual(result["type_token_ratio"], 0.0)
        self.assertLessEqual(result["type_token_ratio"], 1.0)
        self.assertGreaterEqual(result["yules_k"], 0.0)
        self.assertGreaterEqual(result["word_length_mean_characters"], 0.0)


if __name__ == "__main__":
    unittest.main()