"""
Test suite for new style analysis modules: vocabulary sophistication, 
stylistic devices, discourse flow, and AI detection.
"""

import unittest
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from text_analysis.modules.vocabulary_sophistication import VocabularySophisticationMetric
from text_analysis.modules.stylistic_devices import StylisticDeviceMetric
from text_analysis.modules.discourse_flow import DiscourseFlowMetric
from text_analysis.modules.ai_detection import AIDetectionMetric
from text_analysis.config_manager import AnalysisConfig


class TestVocabularySophistication(unittest.TestCase):
    """Test vocabulary sophistication analysis."""
    
    def setUp(self):
        config = AnalysisConfig({})
        self.analyzer = VocabularySophisticationMetric(config)
    
    def test_syllable_counting(self):
        """Test syllable counting accuracy."""
        test_cases = [
            ("cat", 1),
            ("running", 2),
            ("beautiful", 3),
            ("university", 5),  # u-ni-ver-si-ty
            ("hello", 2),
            ("the", 1),
            ("", 0)
        ]
        
        for word, expected in test_cases:
            with self.subTest(word=word):
                result = self.analyzer.calculate_syllables(word)
                self.assertEqual(result, expected, f"Expected {expected} syllables for '{word}', got {result}")
    
    def test_basic_vocabulary_analysis(self):
        """Test basic vocabulary sophistication metrics."""
        text = "The sophisticated university professor delivered an exceptionally comprehensive lecture."
        result = self.analyzer.analyze(text)
        
        # Check that all required keys are present
        expected_keys = [
            'avg_syllables_per_word', 'complex_word_ratio', 'avg_word_length',
            'long_word_ratio', 'formality_score', 'vocabulary_sophistication'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result)
            self.assertIsInstance(result[key], (int, float))
    
    def test_formal_vs_informal_text(self):
        """Test formality score detection."""
        formal_text = "Furthermore, the aforementioned analysis demonstrates that the utilization of academic terminology facilitates comprehension."
        informal_text = "Yeah, so like, this stuff is really cool and awesome, you know? It's pretty great!"
        
        formal_result = self.analyzer.analyze(formal_text)
        informal_result = self.analyzer.analyze(informal_text)
        
        # Formal text should have higher formality score
        self.assertGreater(formal_result['formality_score'], informal_result['formality_score'])
        # Note: academic_word_ratio might both be 0 for short texts, just check formality
    
    def test_empty_text(self):
        """Test handling of empty text."""
        result = self.analyzer.analyze("")
        
        self.assertEqual(result['avg_syllables_per_word'], 0.0)
        self.assertEqual(result['total_words_analyzed'], 0)


class TestStylisticDevices(unittest.TestCase):
    """Test stylistic device detection."""
    
    def setUp(self):
        config = AnalysisConfig({})
        self.analyzer = StylisticDeviceMetric(config)
    
    def test_alliteration_detection(self):
        """Test alliteration detection accuracy."""
        # Clear alliteration example
        text = "Peter Piper picked a peck of pickled peppers."
        alliterations = self.analyzer.detect_alliteration(text)
        
        self.assertGreater(len(alliterations), 0, "Should detect alliteration")
        
        # Text without alliteration
        text_no_alliteration = "The cat sat on the mat."
        alliterations_none = self.analyzer.detect_alliteration(text_no_alliteration)
        
        self.assertEqual(len(alliterations_none), 0, "Should not detect alliteration")
    
    def test_repetition_detection(self):
        """Test word repetition detection."""
        text = "The quick quick brown fox jumps jumps over the lazy dog."
        result = self.analyzer.detect_repetition(text)
        
        self.assertGreater(result['word_repetition_count'], 0)
        self.assertIn('repeated_words', result)
    
    def test_stylistic_density_calculation(self):
        """Test stylistic density is properly normalized."""
        # Rich text with multiple devices
        rich_text = "Peter Piper picked pickled peppers. The quick quick brown fox jumps. This is like lightning."
        result = self.analyzer.analyze_devices(rich_text)
        
        self.assertGreater(result['stylistic_density'], 0)
        self.assertIsInstance(result['stylistic_density'], float)
        
        # Plain text should have lower density
        plain_text = "The cat sat on the mat. It was a normal day."
        plain_result = self.analyzer.analyze_devices(plain_text)
        
        self.assertLessEqual(plain_result['stylistic_density'], result['stylistic_density'])
    
    def test_sentence_variety(self):
        """Test sentence variety analysis."""
        # Varied sentence openings
        varied_text = "Today was beautiful. However, it rained. Because of this, I stayed inside. Finally, the sun came out."
        result = self.analyzer.analyze_sentence_variety(varied_text)
        
        self.assertGreater(result['sentence_variety_score'], 0.5)
        
        # Repetitive sentence openings
        repetitive_text = "The cat sat. The dog ran. The bird flew. The fish swam."
        repetitive_result = self.analyzer.analyze_sentence_variety(repetitive_text)
        
        self.assertLess(repetitive_result['sentence_variety_score'], result['sentence_variety_score'])


class TestDiscourseFlow(unittest.TestCase):
    """Test discourse flow analysis."""
    
    def setUp(self):
        config = AnalysisConfig({})
        self.analyzer = DiscourseFlowMetric(config)
    
    def test_transition_detection(self):
        """Test transition word detection across categories."""
        text = "First, we consider the problem. However, there are complications. Therefore, we need alternatives. Finally, we reach a conclusion."
        result = self.analyzer.analyze_transitions(text)
        
        # Should detect transitions from multiple categories
        self.assertGreater(result['total_transitions'], 0)
        self.assertIn('temporal', result['transition_counts'])
        self.assertIn('contrastive', result['transition_counts'])
        self.assertIn('causal', result['transition_counts'])
        
        # Check transition density calculation
        self.assertGreater(result['transition_density'], 0)
    
    def test_paragraph_structure(self):
        """Test paragraph opening pattern analysis."""
        text = """What is the solution?
        
        The answer lies in careful analysis.
        
        First, we must consider the evidence.
        
        I believe this approach will work."""
        
        result = self.analyzer.analyze_paragraph_structure(text)
        
        self.assertGreater(result['paragraph_count'], 0)
        self.assertIn('opening_patterns', result)
        
        # Should detect question, statement, temporal, and personal openings
        patterns = result['opening_patterns']
        self.assertGreater(patterns['question'] + patterns['statement'] + 
                          patterns['temporal'] + patterns['personal'], 0)
    
    def test_coherence_analysis(self):
        """Test text coherence measurement."""
        # Coherent text with referential markers
        coherent_text = "The scientist conducted experiments. These experiments revealed important findings. The results supported the hypothesis."
        result = self.analyzer.analyze_coherence(coherent_text)
        
        self.assertGreater(result['referential_density'], 0)
        self.assertGreaterEqual(result['topic_consistency'], 0)
        self.assertLessEqual(result['topic_consistency'], 1)
    
    def test_discourse_score_calculation(self):
        """Test overall discourse score calculation."""
        well_structured_text = """First, let me explain the concept. However, there are several considerations. 
        These factors are important. Therefore, we must analyze them carefully. 
        The results show clear patterns. Finally, we can draw conclusions."""
        
        result = self.analyzer.analyze_discourse(well_structured_text)
        
        self.assertIn('discourse_score', result)
        self.assertGreaterEqual(result['discourse_score'], 0)
        self.assertLessEqual(result['discourse_score'], 1)


class TestAIDetection(unittest.TestCase):
    """Test AI detection patterns."""
    
    def setUp(self):
        config = AnalysisConfig({})
        self.analyzer = AIDetectionMetric(config)
    
    def test_sycophancy_detection(self):
        """Test AI sycophancy pattern detection."""
        ai_text = "I'd be happy to help you with that! That's a great question. I appreciate you asking."
        result = self.analyzer.count_pattern_matches(ai_text, self.analyzer.sycophancy_patterns, 'sycophancy')
        
        self.assertGreater(result['sycophancy_count'], 0)
        self.assertIn('sycophancy_examples', result)
    
    def test_contrastive_negation_detection(self):
        """Test contrastive negation pattern detection."""
        ai_text = "While I can't provide specific advice, I can certainly offer general guidance. Although I don't have access to that, I can help in other ways."
        result = self.analyzer.count_pattern_matches(ai_text, self.analyzer.contrastive_patterns, 'contrastive')
        
        self.assertGreater(result['contrastive_count'], 0)
    
    def test_hedging_detection(self):
        """Test hedging pattern detection."""
        hedged_text = "It seems that this might be the case. It's worth noting that generally speaking, this could be true."
        result = self.analyzer.count_pattern_matches(hedged_text, self.analyzer.hedging_patterns, 'hedging')
        
        self.assertGreater(result['hedging_count'], 0)
    
    def test_ai_likelihood_scoring(self):
        """Test AI likelihood score calculation."""
        # High AI characteristics
        high_ai_text = """I'd be absolutely delighted to help! Thank you for this excellent question. 
        While I can't provide specific details, I can certainly offer general guidance. 
        Here's how you can approach this: First, consider the options. Second, evaluate the results."""
        
        result = self.analyzer.analyze_ai_patterns(high_ai_text)
        
        self.assertIn('ai_likelihood_score', result)
        self.assertGreaterEqual(result['ai_likelihood_score'], 0)
        self.assertLessEqual(result['ai_likelihood_score'], 1)
        
        # Human-like text should score lower
        human_text = "I think this is probably okay. Maybe we should try something else."
        human_result = self.analyzer.analyze_ai_patterns(human_text)
        
        self.assertLessEqual(human_result['ai_likelihood_score'], result['ai_likelihood_score'])
    
    def test_punctuation_analysis(self):
        """Test AI punctuation pattern detection."""
        text_with_em_dashes = "The solution—while complex—requires careful consideration. This approach—though challenging—offers benefits."
        result = self.analyzer.analyze_punctuation_tells(text_with_em_dashes)
        
        self.assertIn('em_dashes_count', result)
        self.assertIn('em_dashes_density', result)
        self.assertGreater(result['em_dashes_count'], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for all new modules working together."""
    
    def test_all_modules_return_valid_data(self):
        """Test that all modules return properly formatted data."""
        config = AnalysisConfig({})
        
        modules = [
            VocabularySophisticationMetric(config),
            StylisticDeviceMetric(config),
            DiscourseFlowMetric(config),
            AIDetectionMetric(config)
        ]
        
        test_text = """I'd be happy to help you understand this sophisticated concept. 
        The beautiful, brilliant analysis demonstrates exceptional clarity. 
        First, consider the evidence. However, we must also examine alternatives. 
        Therefore, the conclusion seems reasonable."""
        
        for module in modules:
            with self.subTest(module=module.name):
                # Create a mock payload
                class MockPayload:
                    def __init__(self, content):
                        self.content = content
                
                payload = MockPayload(test_text)
                result = module.compute(payload, {})
                
                self.assertIsInstance(result, dict)
                self.assertGreater(len(result), 0)
    
    def test_edge_case_empty_text(self):
        """Test all modules handle empty text gracefully."""
        config = AnalysisConfig({})
        
        modules = [
            VocabularySophisticationMetric(config),
            StylisticDeviceMetric(config),
            DiscourseFlowMetric(config),
            AIDetectionMetric(config)
        ]
        
        for module in modules:
            with self.subTest(module=module.name):
                class MockPayload:
                    def __init__(self, content):
                        self.content = content
                
                payload = MockPayload("")
                result = module.compute(payload, {})
                
                self.assertIsInstance(result, dict)
                # Should not crash and should return some structure


def run_new_module_tests():
    """Run all tests for the new modules."""
    print("🧪 Running comprehensive tests for new style analysis modules...")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestVocabularySophistication,
        TestStylisticDevices,
        TestDiscourseFlow,
        TestAIDetection,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"📊 TEST SUMMARY:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"   - {test}")
    
    if result.errors:
        print(f"\n💥 ERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"   - {test}")
    
    if not result.failures and not result.errors:
        print("\n✅ All tests passed! New modules are working correctly.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_new_module_tests()
    exit(0 if success else 1)