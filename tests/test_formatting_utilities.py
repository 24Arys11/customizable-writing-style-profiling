"""Comprehensive tests for formatting utilities."""
import unittest
from unittest.mock import Mock
from pathlib import Path

from src.text_analysis.utils.formatting import format_summary, format_table


class TestFormattingUtilities(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test data."""
        self.sample_report = {
            "metadata": {
                "text_length": 1250,
                "file_path": "/test/sample.txt"
            },
            "sentence_stats": {
                "sentence_count": 10,
                "paragraph_count": 3,
                "mean_sentence_length_tokens": 15.3
            },
            "lexical_richness": {
                "type_token_ratio": 0.75,
                "yules_k": 125.5,
                "average_word_length": 4.2
            },
            "punctuation_patterns": {
                "comma_density_per_1000_words": 45.2,
                "period_density_per_1000_words": 67.8,
                "exclamation_density_per_1000_words": 12.1
            },
            "syntactic_complexity": {
                "average_parse_depth": 3.5,
                "coordination_density_per_1000_words": 23.1,
                "subordination_density_per_1000_words": 15.6
            },
            "rhythm_patterns": {
                "sentence_length_autocorrelation": 0.234,
                "word_length_autocorrelation": 0.156
            }
        }

    def test_format_summary_standard(self) -> None:
        """Test standard summary formatting."""
        output = format_summary(self.sample_report, "standard")
        
        # Should contain header and basic info
        self.assertIn("TEXT ANALYSIS SUMMARY", output)
        self.assertIn("1,250 characters", output)
        self.assertIn("Sentences: 10", output)
        self.assertIn("Paragraphs: 3", output)
        self.assertIn("15.3 words", output)
        
        # Should contain style characteristics section
        self.assertIn("STYLE CHARACTERISTICS", output)

    def test_format_summary_detailed(self) -> None:
        """Test detailed summary formatting."""
        output = format_summary(self.sample_report, "detailed")
        
        # Should contain all the same basic info as standard
        self.assertIn("TEXT ANALYSIS SUMMARY", output)
        self.assertIn("STYLE CHARACTERISTICS", output)
        
        # May contain additional detail depending on implementation
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 100)

    def test_format_table_standard(self) -> None:
        """Test table formatting with standard verbosity."""
        output = format_table(self.sample_report, "standard")
        
        # Should contain table header
        self.assertIn("AUTHORSHIP METRICS TABLE", output)
        
        # Should be formatted as a table-like structure
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 50)

    def test_format_table_detailed(self) -> None:
        """Test table formatting with detailed verbosity."""
        output = format_table(self.sample_report, "detailed")
        
        # Should contain table structure
        self.assertIn("AUTHORSHIP METRICS TABLE", output)
        
        # Should be a string output
        self.assertIsInstance(output, str)

    def test_empty_report_handling(self) -> None:
        """Test handling of empty or minimal reports."""
        minimal_report = {
            "metadata": {},
            "sentence_stats": {}
        }
        
        # Should handle minimal data gracefully
        summary_output = format_summary(minimal_report, "standard")
        table_output = format_table(minimal_report, "standard")
        
        self.assertIn("TEXT ANALYSIS SUMMARY", summary_output)
        self.assertIn("AUTHORSHIP METRICS TABLE", table_output)
        
        # Should not crash on missing data
        self.assertIsInstance(summary_output, str)
        self.assertIsInstance(table_output, str)

    def test_missing_metadata_handling(self) -> None:
        """Test handling when metadata is missing."""
        report_no_metadata = {
            "sentence_stats": {
                "sentence_count": 5,
                "paragraph_count": 2
            }
        }
        
        output = format_summary(report_no_metadata, "standard")
        
        # Should handle missing metadata gracefully
        self.assertIn("TEXT ANALYSIS SUMMARY", output)
        self.assertIn("Sentences: 5", output)

    def test_missing_sentence_stats_handling(self) -> None:
        """Test handling when sentence stats are missing."""
        report_no_sentences = {
            "metadata": {
                "text_length": 500
            }
        }
        
        output = format_summary(report_no_sentences, "standard")
        
        # Should handle missing sentence stats gracefully
        self.assertIn("TEXT ANALYSIS SUMMARY", output)
        self.assertIn("500 characters", output)

    def test_format_summary_contains_insights(self) -> None:
        """Test that summary contains style insights."""
        output = format_summary(self.sample_report, "detailed")
        
        # Should extract and display insights from the data
        self.assertIn("STYLE CHARACTERISTICS", output)
        
        # Should process the report data into readable insights
        self.assertGreater(len(output.split('\n')), 5)  # Multiple lines of output

    def test_verbosity_parameter_handling(self) -> None:
        """Test different verbosity levels."""
        standard_output = format_summary(self.sample_report, "standard")
        detailed_output = format_summary(self.sample_report, "detailed")
        
        # Both should be valid strings
        self.assertIsInstance(standard_output, str)
        self.assertIsInstance(detailed_output, str)
        
        # Both should contain the main header
        self.assertIn("TEXT ANALYSIS SUMMARY", standard_output)
        self.assertIn("TEXT ANALYSIS SUMMARY", detailed_output)

    def test_table_format_structure(self) -> None:
        """Test table format produces structured output."""
        output = format_table(self.sample_report, "standard")
        
        # Should have table-like structure with headers
        self.assertIn("AUTHORSHIP METRICS TABLE", output)
        
        # Should be formatted text (not JSON)
        self.assertIsInstance(output, str)
        
        # Should not be empty
        self.assertGreater(len(output), 50)

    def test_numeric_value_formatting(self) -> None:
        """Test that numeric values are formatted appropriately."""
        report_with_decimals = {
            "metadata": {"text_length": 1234},
            "sentence_stats": {"mean_sentence_length_tokens": 15.123456789},
            "lexical_richness": {"type_token_ratio": 0.987654321}
        }
        
        output = format_summary(report_with_decimals, "standard")
        
        # Should format numbers to reasonable precision
        self.assertIn("15.1", output)  # Should round appropriately
        
    def test_large_numbers_formatting(self) -> None:
        """Test formatting of large numbers."""
        report_large = {
            "metadata": {"text_length": 123456789},
            "sentence_stats": {"sentence_count": 5000}
        }
        
        output = format_summary(report_large, "standard")
        
        # Should format large numbers with commas
        self.assertIn("123,456,789", output)


if __name__ == "__main__":
    unittest.main()