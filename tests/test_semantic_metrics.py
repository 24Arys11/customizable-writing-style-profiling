"""Comprehensive tests for semantic metrics calculations."""
import unittest
from pathlib import Path

from src.text_analysis.config_manager import AnalysisConfig
from src.text_analysis.modules.semantic_metrics import SemanticMetrics
from src.text_analysis.text_analyzer import TextPayload


class TestSemanticMetrics(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test configuration."""
        self.config = AnalysisConfig(raw={
            "metrics": {"normalization_basis": 1000}
        })
        self.module = SemanticMetrics(self.config)

    def test_semantic_coherence_basic(self) -> None:
        """Test basic semantic coherence calculation."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="The cat sat on the mat. The dog ran in the park."
        )
        context = {
            "sentences": [
                "The cat sat on the mat.",
                "The dog ran in the park."
            ],
            "tokens": ["The", "cat", "sat", "on", "the", "mat", "The", "dog", "ran", "in", "the", "park"]
        }
        
        result = self.module.compute(payload, context)
        
        # Should calculate semantic coherence metrics
        self.assertIn("semantic_coherence", result)
        self.assertIsInstance(result["semantic_coherence"], float)
        self.assertGreaterEqual(result["semantic_coherence"], 0.0)
        self.assertLessEqual(result["semantic_coherence"], 1.0)

    def test_topic_consistency(self) -> None:
        """Test topic consistency measurements."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="Dogs are loyal pets. Cats are independent animals. Fish need water to survive."
        )
        context = {
            "sentences": [
                "Dogs are loyal pets.",
                "Cats are independent animals.", 
                "Fish need water to survive."
            ],
            "tokens": ["Dogs", "are", "loyal", "pets", "Cats", "are", "independent", "animals", "Fish", "need", "water", "to", "survive"]
        }
        
        result = self.module.compute(payload, context)
        
        # Should provide topic-related metrics
        self.assertIn("semantic_coherence", result)
        # Animal-related sentences should have reasonable coherence
        self.assertGreater(result["semantic_coherence"], 0.0)

    def test_empty_text_handling(self) -> None:
        """Test handling of empty text."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content=""
        )
        context = {
            "sentences": [],
            "tokens": []
        }
        
        result = self.module.compute(payload, context)
        
        # Should handle empty input gracefully
        self.assertEqual(result["semantic_coherence"], 0.0)

    def test_single_sentence_text(self) -> None:
        """Test text with only one sentence."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="This is a single sentence."
        )
        context = {
            "sentences": ["This is a single sentence."],
            "tokens": ["This", "is", "a", "single", "sentence"]
        }
        
        result = self.module.compute(payload, context)
        
        # Single sentence should have perfect coherence with itself
        self.assertEqual(result["semantic_coherence"], 1.0)

    def test_highly_coherent_text(self) -> None:
        """Test text with high semantic coherence."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="The library contains many books. Students read books in the library. Books provide knowledge to readers."
        )
        context = {
            "sentences": [
                "The library contains many books.",
                "Students read books in the library.",
                "Books provide knowledge to readers."
            ],
            "tokens": ["The", "library", "contains", "many", "books", "Students", "read", "books", "in", "the", "library", "Books", "provide", "knowledge", "to", "readers"]
        }
        
        result = self.module.compute(payload, context)
        
        # Related sentences about books/library should have high coherence
        self.assertGreater(result["semantic_coherence"], 0.3)

    def test_incoherent_text(self) -> None:
        """Test text with low semantic coherence."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="The purple elephant danced. Mathematical equations solve themselves. Refrigerators dream of electric sheep."
        )
        context = {
            "sentences": [
                "The purple elephant danced.",
                "Mathematical equations solve themselves.",
                "Refrigerators dream of electric sheep."
            ],
            "tokens": ["The", "purple", "elephant", "danced", "Mathematical", "equations", "solve", "themselves", "Refrigerators", "dream", "of", "electric", "sheep"]
        }
        
        result = self.module.compute(payload, context)
        
        # Unrelated/nonsensical sentences should have lower coherence
        self.assertLessEqual(result["semantic_coherence"], 0.8)

    def test_technical_coherence(self) -> None:
        """Test coherence in technical text."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="Machine learning algorithms process data. Neural networks use backpropagation for training. Deep learning models require large datasets."
        )
        context = {
            "sentences": [
                "Machine learning algorithms process data.",
                "Neural networks use backpropagation for training.",
                "Deep learning models require large datasets."
            ],
            "tokens": ["Machine", "learning", "algorithms", "process", "data", "Neural", "networks", "use", "backpropagation", "for", "training", "Deep", "learning", "models", "require", "large", "datasets"]
        }
        
        result = self.module.compute(payload, context)
        
        # Technical domain-specific text should show coherence
        self.assertGreater(result["semantic_coherence"], 0.2)

    def test_narrative_coherence(self) -> None:
        """Test coherence in narrative text."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="Sarah walked to the store. She bought milk and bread. Then she returned home."
        )
        context = {
            "sentences": [
                "Sarah walked to the store.",
                "She bought milk and bread.",
                "Then she returned home."
            ],
            "tokens": ["Sarah", "walked", "to", "the", "store", "She", "bought", "milk", "and", "bread", "Then", "she", "returned", "home"]
        }
        
        result = self.module.compute(payload, context)
        
        # Narrative text should have good coherence
        self.assertGreater(result["semantic_coherence"], 0.3)

    def test_repeated_content_coherence(self) -> None:
        """Test coherence with repeated content."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="The cat sat. The cat sat. The cat sat."
        )
        context = {
            "sentences": [
                "The cat sat.",
                "The cat sat.",
                "The cat sat."
            ],
            "tokens": ["The", "cat", "sat", "The", "cat", "sat", "The", "cat", "sat"]
        }
        
        result = self.module.compute(payload, context)
        
        # Identical sentences should have very high coherence
        self.assertGreater(result["semantic_coherence"], 0.8)

    def test_mixed_topic_coherence(self) -> None:
        """Test coherence with mixed but related topics."""
        payload = TextPayload(
            path=Path("/tmp/test.txt"),
            content="Cooking requires fresh ingredients. The kitchen should be clean. Recipes help guide preparation."
        )
        context = {
            "sentences": [
                "Cooking requires fresh ingredients.",
                "The kitchen should be clean.",
                "Recipes help guide preparation."
            ],
            "tokens": ["Cooking", "requires", "fresh", "ingredients", "The", "kitchen", "should", "be", "clean", "Recipes", "help", "guide", "preparation"]
        }
        
        result = self.module.compute(payload, context)
        
        # Related cooking topics should show moderate coherence
        self.assertGreater(result["semantic_coherence"], 0.2)
        self.assertLess(result["semantic_coherence"], 1.0)


if __name__ == "__main__":
    unittest.main()