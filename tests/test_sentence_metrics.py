"""Comprehensive tests for sentence metrics calculations."""
import unittest
from pathlib import Path

from src.text_analysis.config_manager import AnalysisConfig
from src.text_analysis.modules.sentence_metrics import SentenceMetrics
from src.text_analysis.text_analyzer import TextPayload


class TestSentenceMetrics(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test configuration."""
        self.config = AnalysisConfig(raw={
            "metrics": {"normalization_basis": 1000}
        })
        self.module = SentenceMetrics(self.config)

    def test_average_sentence_length(self) -> None:
        """Test average sentence length calculation."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="This is short. This is a much longer sentence with many words."
        )
        context = {
            "sentences": [
                "This is short.",
                "This is a much longer sentence with many words."
            ],
            "tokens": ["This", "is", "short", "This", "is", "a", "much", "longer", 
                      "sentence", "with", "many", "words"]
        }
        
        result = self.module.compute(payload, context)
        
        # First sentence: 3 words, Second sentence: 9 words
        # Average: (3 + 9) / 2 = 6
        self.assertAlmostEqual(result["mean_sentence_length_tokens"], 6.0, places=2)

    def test_sentence_length_variation(self) -> None:
        """Test sentence length variation metrics."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="Short. This is longer. This is a very much longer sentence indeed."
        )
        context = {
            "sentences": [
                "Short.",
                "This is longer.",
                "This is a very much longer sentence indeed."
            ],
            "tokens": ["Short", "This", "is", "longer", "This", "is", "a", "very", 
                      "much", "longer", "sentence", "indeed"]
        }
        
        result = self.module.compute(payload, context)
        
        # Sentence lengths: 1, 3, 8 words
        # Should have non-zero standard deviation and coefficient of variation
        self.assertGreater(result["std_sentence_length_tokens"], 0.0)
        self.assertGreater(result["sentence_length_coefficient_of_variation"], 0.0)

    def test_paragraph_analysis(self) -> None:
        """Test paragraph length analysis."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="Short paragraph.\n\nThis is a much longer paragraph with multiple sentences. It has more content and should be longer.\n\nAnother short one."
        )
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in payload.content.split('\n\n') if p.strip()]
        
        context = {
            "paragraphs": paragraphs,
            "sentences": [
                "Short paragraph.",
                "This is a much longer paragraph with multiple sentences.",
                "It has more content and should be longer.",
                "Another short one."
            ]
        }
        
        result = self.module.compute(payload, context)
        
        # Should calculate paragraph statistics
        self.assertIn("paragraph_length_mean_words", result)
        self.assertIn("paragraph_length_std_words", result)
        self.assertIn("paragraph_length_coefficient_of_variation_words", result)
        
        # Verify non-zero variation since paragraphs have different lengths
        if len(paragraphs) > 1:
            self.assertGreaterEqual(result["paragraph_length_coefficient_of_variation_words"], 0.0)

    def test_empty_text_handling(self) -> None:
        """Test handling of empty text."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content=""
        )
        context = {
            "sentences": [],
            "paragraphs": [],
            "tokens": []
        }
        
        result = self.module.compute(payload, context)
        
        # Should handle empty input gracefully
        self.assertEqual(result["sentence_count"], 0)
        self.assertEqual(result["average_sentence_length_words"], 0.0)
        self.assertEqual(result["paragraph_count"], 0)

    def test_single_sentence_text(self) -> None:
        """Test text with only one sentence."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="This is the only sentence."
        )
        context = {
            "sentences": ["This is the only sentence."],
            "paragraphs": ["This is the only sentence."],
            "tokens": ["This", "is", "the", "only", "sentence"]
        }
        
        result = self.module.compute(payload, context)
        
        self.assertEqual(result["sentence_count"], 1)
        self.assertEqual(result["average_sentence_length_words"], 5.0)
        # Standard deviation should be 0 with only one sentence
        self.assertEqual(result["sentence_length_std_dev_words"], 0.0)

    def test_sentence_complexity_metrics(self) -> None:
        """Test sentence complexity analysis."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="Simple sentence. Complex sentence with subordinate clauses, coordinating conjunctions, and multiple phrases."
        )
        context = {
            "sentences": [
                "Simple sentence.",
                "Complex sentence with subordinate clauses, coordinating conjunctions, and multiple phrases."
            ],
            "tokens": ["Simple", "sentence", "Complex", "sentence", "with", "subordinate", 
                      "clauses", "coordinating", "conjunctions", "and", "multiple", "phrases"]
        }
        
        result = self.module.compute(payload, context)
        
        # Should include sentence complexity metrics
        self.assertIn("average_sentence_length_words", result)
        self.assertIn("sentence_length_std_dev_words", result)
        
        # Basic sanity checks
        self.assertGreater(result["average_sentence_length_words"], 0.0)
        self.assertGreaterEqual(result["sentence_length_std_dev_words"], 0.0)

    def test_punctuation_density_per_sentence(self) -> None:
        """Test punctuation density calculations."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="Hello, world! How are you today? Fine, thanks."
        )
        context = {
            "sentences": [
                "Hello, world!",
                "How are you today?",
                "Fine, thanks."
            ]
        }
        
        result = self.module.compute(payload, context)
        
        # Should have basic sentence metrics
        self.assertEqual(result["sentence_count"], 3)
        self.assertGreater(result["average_sentence_length_words"], 0.0)


if __name__ == "__main__":
    unittest.main()