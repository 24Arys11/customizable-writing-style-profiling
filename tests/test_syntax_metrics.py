"""Comprehensive tests for syntax metrics calculations."""
import unittest
from pathlib import Path

from src.text_analysis.config_manager import AnalysisConfig
from src.text_analysis.modules.syntax_metrics import SyntaxMetrics
from src.text_analysis.text_analyzer import TextPayload


class TestSyntaxMetrics(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test configuration."""
        self.config = AnalysisConfig(raw={
            "metrics": {"normalization_basis": 1000}
        })
        self.module = SyntaxMetrics(self.config)

    def test_parse_depth_calculation(self) -> None:
        """Test parse depth calculation."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="The cat sat. The big brown dog that was sleeping peacefully woke up."
        )
        
        # Mock dependency parse data
        context = {
            "content": payload.content,
            "sentences": [
                "The cat sat.",
                "The big brown dog that was sleeping peacefully woke up."
            ],
            "dependency_parse": [
                [
                    {"head": 2, "dep": "det", "text": "The"},
                    {"head": 2, "dep": "nsubj", "text": "cat"},
                    {"head": 0, "dep": "ROOT", "text": "sat"}
                ],
                [
                    {"head": 4, "dep": "det", "text": "The"},
                    {"head": 4, "dep": "amod", "text": "big"},
                    {"head": 4, "dep": "amod", "text": "brown"},
                    {"head": 11, "dep": "nsubj", "text": "dog"},
                    {"head": 4, "dep": "nsubj", "text": "that"},
                    {"head": 7, "dep": "aux", "text": "was"},
                    {"head": 4, "dep": "acl:relcl", "text": "sleeping"},
                    {"head": 7, "dep": "advmod", "text": "peacefully"},
                    {"head": 0, "dep": "ROOT", "text": "woke"},
                    {"head": 11, "dep": "compound:prt", "text": "up"}
                ]
            ]
        }
        
        result = self.module.compute(payload, context)
        
        # Should calculate parse depth metrics
        self.assertIn("average_parse_depth", result)
        self.assertIn("max_parse_depth", result)
        self.assertGreater(result["average_parse_depth"], 0.0)
        self.assertGreater(result["max_parse_depth"], 0.0)

    def test_coordination_detection(self) -> None:
        """Test coordination pattern detection."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="I like cats and dogs but not birds or fish."
        )
        
        context = {
            "content": payload.content,
            "tokens": ["I", "like", "cats", "and", "dogs", "but", "not", "birds", "or", "fish"],
            "dependency_parse": [
                [
                    {"head": 2, "dep": "nsubj", "text": "I"},
                    {"head": 0, "dep": "ROOT", "text": "like"},
                    {"head": 2, "dep": "dobj", "text": "cats"},
                    {"head": 5, "dep": "cc", "text": "and"},
                    {"head": 3, "dep": "conj", "text": "dogs"},
                    {"head": 2, "dep": "cc", "text": "but"},
                    {"head": 8, "dep": "neg", "text": "not"},
                    {"head": 2, "dep": "conj", "text": "birds"},
                    {"head": 10, "dep": "cc", "text": "or"},
                    {"head": 8, "dep": "conj", "text": "fish"}
                ]
            ]
        }
        
        result = self.module.compute(payload, context)
        
        # Should detect coordination patterns
        self.assertIn("coordination_density_per_1000_words", result)
        
        # Should find coordinating conjunctions (and, but, or)
        expected_coords = 3  # "and", "but", "or"
        word_count = len(context["tokens"])
        expected_density = (expected_coords / word_count) * 1000
        
        self.assertAlmostEqual(
            result["coordination_density_per_1000_words"], 
            expected_density, 
            places=1
        )

    def test_subordination_detection(self) -> None:
        """Test subordination pattern detection."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="Although it was raining, we went outside because we needed exercise."
        )
        
        context = {
            "content": payload.content,
            "tokens": ["Although", "it", "was", "raining", "we", "went", "outside", "because", "we", "needed", "exercise"],
            "dependency_parse": [
                [
                    {"head": 4, "dep": "mark", "text": "Although"},
                    {"head": 4, "dep": "nsubj", "text": "it"},
                    {"head": 4, "dep": "aux", "text": "was"},
                    {"head": 6, "dep": "advcl", "text": "raining"},
                    {"head": 6, "dep": "nsubj", "text": "we"},
                    {"head": 0, "dep": "ROOT", "text": "went"},
                    {"head": 6, "dep": "advmod", "text": "outside"},
                    {"head": 10, "dep": "mark", "text": "because"},
                    {"head": 10, "dep": "nsubj", "text": "we"},
                    {"head": 6, "dep": "advcl", "text": "needed"},
                    {"head": 10, "dep": "dobj", "text": "exercise"}
                ]
            ]
        }
        
        result = self.module.compute(payload, context)
        
        # Should detect subordination patterns
        self.assertIn("subordination_density_per_1000_words", result)
        
        # Should find subordinating markers
        word_count = len(context["tokens"])
        self.assertGreater(result["subordination_density_per_1000_words"], 0.0)

    def test_empty_text_handling(self) -> None:
        """Test handling of empty text."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content=""
        )
        context = {
            "content": "",
            "sentences": [],
            "tokens": [],
            "dependency_parse": []
        }
        
        result = self.module.compute(payload, context)
        
        # Should handle empty input gracefully
        self.assertEqual(result["average_parse_depth"], 0.0)
        self.assertEqual(result["max_parse_depth"], 0.0)
        self.assertEqual(result["coordination_density_per_1000_words"], 0.0)
        self.assertEqual(result["subordination_density_per_1000_words"], 0.0)

    def test_simple_sentences(self) -> None:
        """Test simple sentences with minimal complexity."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="Cat sleeps. Dog runs."
        )
        
        context = {
            "content": payload.content,
            "sentences": ["Cat sleeps.", "Dog runs."],
            "tokens": ["Cat", "sleeps", "Dog", "runs"],
            "dependency_parse": [
                [
                    {"head": 2, "dep": "nsubj", "text": "Cat"},
                    {"head": 0, "dep": "ROOT", "text": "sleeps"}
                ],
                [
                    {"head": 2, "dep": "nsubj", "text": "Dog"},
                    {"head": 0, "dep": "ROOT", "text": "runs"}
                ]
            ]
        }
        
        result = self.module.compute(payload, context)
        
        # Simple sentences should have low complexity
        self.assertGreater(result["average_parse_depth"], 0.0)
        self.assertLessEqual(result["average_parse_depth"], 3.0)  # Should be relatively low

    def test_complex_sentences(self) -> None:
        """Test complex sentences with high syntactic complexity."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="The students who were studying diligently for their final exams, which were scheduled next week, felt confident."
        )
        
        # Mock complex dependency structure
        context = {
            "content": payload.content,
            "sentences": [payload.content],
            "tokens": ["The", "students", "who", "were", "studying", "diligently", "for", "their", "final", "exams", "which", "were", "scheduled", "next", "week", "felt", "confident"],
            "dependency_parse": [
                [
                    {"head": 2, "dep": "det", "text": "The"},
                    {"head": 16, "dep": "nsubj", "text": "students"},
                    {"head": 5, "dep": "nsubj", "text": "who"},
                    {"head": 5, "dep": "aux", "text": "were"},
                    {"head": 2, "dep": "acl:relcl", "text": "studying"},
                    {"head": 5, "dep": "advmod", "text": "diligently"},
                    {"head": 5, "dep": "prep", "text": "for"},
                    {"head": 10, "dep": "poss", "text": "their"},
                    {"head": 10, "dep": "amod", "text": "final"},
                    {"head": 7, "dep": "pobj", "text": "exams"},
                    {"head": 13, "dep": "nsubjpass", "text": "which"},
                    {"head": 13, "dep": "auxpass", "text": "were"},
                    {"head": 10, "dep": "acl:relcl", "text": "scheduled"},
                    {"head": 15, "dep": "amod", "text": "next"},
                    {"head": 13, "dep": "npadvmod", "text": "week"},
                    {"head": 0, "dep": "ROOT", "text": "felt"},
                    {"head": 16, "dep": "acomp", "text": "confident"}
                ]
            ]
        }
        
        result = self.module.compute(payload, context)
        
        # Complex sentences should have higher parse depth
        self.assertGreater(result["average_parse_depth"], 2.0)
        self.assertGreater(result["max_parse_depth"], 3.0)

    def test_dependency_relationship_counting(self) -> None:
        """Test counting of specific dependency relationships."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="The quick brown fox jumps over the lazy dog."
        )
        
        context = {
            "content": payload.content,
            "dependency_parse": [
                [
                    {"head": 4, "dep": "det", "text": "The"},
                    {"head": 4, "dep": "amod", "text": "quick"},
                    {"head": 4, "dep": "amod", "text": "brown"},
                    {"head": 5, "dep": "nsubj", "text": "fox"},
                    {"head": 0, "dep": "ROOT", "text": "jumps"},
                    {"head": 5, "dep": "prep", "text": "over"},
                    {"head": 9, "dep": "det", "text": "the"},
                    {"head": 9, "dep": "amod", "text": "lazy"},
                    {"head": 6, "dep": "pobj", "text": "dog"}
                ]
            ]
        }
        
        result = self.module.compute(payload, context)
        
        # Should process dependency relationships
        self.assertIn("average_parse_depth", result)
        self.assertGreater(result["average_parse_depth"], 0.0)


if __name__ == "__main__":
    unittest.main()