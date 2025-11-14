"""Comprehensive tests for punctuation metrics calculations."""
import unittest
from pathlib import Path

from src.text_analysis.config_manager import AnalysisConfig
from src.text_analysis.modules.punctuation_metrics import PunctuationMetrics
from src.text_analysis.text_analyzer import TextPayload


class TestPunctuationMetrics(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test configuration."""
        self.config = AnalysisConfig(raw={
            "metrics": {"normalization_basis": 1000}
        })
        self.module = PunctuationMetrics(self.config)

    def test_basic_punctuation_counts(self) -> None:
        """Test basic punctuation counting."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="Hello, world! How are you? Fine; thanks: excellent."
        )
        context = {
            "content": payload.content
        }
        
        result = self.module.compute(payload, context)
        
        # Count expected punctuation
        self.assertEqual(result["comma_count"], 1)
        self.assertEqual(result["period_count"], 1)
        self.assertEqual(result["exclamation_count"], 1)
        self.assertEqual(result["question_count"], 1)
        self.assertEqual(result["semicolon_count"], 1)
        self.assertEqual(result["colon_count"], 1)

    def test_punctuation_density(self) -> None:
        """Test punctuation density calculations."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="Hello! How are you? Fine."
        )
        context = {
            "content": payload.content,
            "tokens": ["Hello", "How", "are", "you", "Fine"]
        }
        
        result = self.module.compute(payload, context)
        
        # 3 punctuation marks (!?.) with 5 words
        # Per 1000 words: 3/5 * 1000 = 600
        expected_density = (3 / 5) * 1000
        self.assertAlmostEqual(result["punctuation_density_per_1000_words"], expected_density, places=1)

    def test_quotation_marks(self) -> None:
        """Test quotation mark counting."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content='He said "Hello there" and she replied \'Good morning\'.'
        )
        context = {
            "content": payload.content
        }
        
        result = self.module.compute(payload, context)
        
        # Should count both types of quotes
        self.assertEqual(result["quotation_count"], 4)  # 2 double + 2 single

    def test_ellipsis_and_dash_counting(self) -> None:
        """Test ellipsis and dash counting."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="Well... I think—no, wait—that's not right."
        )
        context = {
            "content": payload.content
        }
        
        result = self.module.compute(payload, context)
        
        # Should detect ellipsis and dashes
        self.assertIn("ellipsis_count", result)
        self.assertIn("dash_count", result)

    def test_empty_text_handling(self) -> None:
        """Test handling of empty text."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content=""
        )
        context = {
            "content": "",
            "tokens": []
        }
        
        result = self.module.compute(payload, context)
        
        # All counts should be zero
        self.assertEqual(result["comma_count"], 0)
        self.assertEqual(result["period_count"], 0)
        self.assertEqual(result["exclamation_count"], 0)
        self.assertEqual(result["question_count"], 0)
        self.assertEqual(result["punctuation_density_per_1000_words"], 0.0)

    def test_text_without_punctuation(self) -> None:
        """Test text with no punctuation."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="hello world this is plain text"
        )
        context = {
            "content": payload.content,
            "tokens": ["hello", "world", "this", "is", "plain", "text"]
        }
        
        result = self.module.compute(payload, context)
        
        # All punctuation counts should be zero
        self.assertEqual(result["comma_count"], 0)
        self.assertEqual(result["period_count"], 0)
        self.assertEqual(result["exclamation_count"], 0)
        self.assertEqual(result["question_count"], 0)
        self.assertEqual(result["punctuation_density_per_1000_words"], 0.0)

    def test_heavy_punctuation_text(self) -> None:
        """Test text with heavy punctuation use."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="Wow! Really? Yes, indeed; absolutely: amazing... Well—maybe not!"
        )
        context = {
            "content": payload.content,
            "tokens": ["Wow", "Really", "Yes", "indeed", "absolutely", "amazing", "Well", "maybe", "not"]
        }
        
        result = self.module.compute(payload, context)
        
        # Should handle high punctuation density
        self.assertGreater(result["punctuation_density_per_1000_words"], 0)
        
        # Verify specific counts
        self.assertEqual(result["exclamation_count"], 2)
        self.assertEqual(result["question_count"], 1)
        self.assertEqual(result["comma_count"], 1)
        self.assertEqual(result["semicolon_count"], 1)
        self.assertEqual(result["colon_count"], 1)

    def test_parentheses_and_brackets(self) -> None:
        """Test parentheses and bracket counting."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="This (is a test) with [square brackets] and {curly braces}."
        )
        context = {
            "content": payload.content
        }
        
        result = self.module.compute(payload, context)
        
        # Should count opening and closing brackets
        self.assertIn("parenthesis_count", result)

    def test_punctuation_ratios(self) -> None:
        """Test punctuation ratio calculations."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="Statement. Question? Exclamation!"
        )
        context = {
            "content": payload.content,
            "sentences": ["Statement.", "Question?", "Exclamation!"]
        }
        
        result = self.module.compute(payload, context)
        
        # Should calculate ratios relative to sentence count
        if "sentences" in context and len(context["sentences"]) > 0:
            sentence_count = len(context["sentences"])
            expected_question_ratio = 1 / sentence_count
            expected_exclamation_ratio = 1 / sentence_count
            
            self.assertAlmostEqual(
                result["question_count"] / sentence_count, 
                expected_question_ratio, 
                places=3
            )


if __name__ == "__main__":
    unittest.main()