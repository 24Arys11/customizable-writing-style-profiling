#!/usr/bin/env python3
"""
Authorship comparison utility using the enhanced aggregation system.
Usage: python compare_authors.py <text1.txt> <text2.txt> <config.yaml>
"""

import sys
from pathlib import Path
import json

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.text_analysis.config_manager import AnalysisConfig
from src.text_analysis.text_analyzer import TextAnalyzer


def analyze_text(text_path: Path, config_path: Path):
    """Analyze a text file and return the aggregator."""
    config = AnalysisConfig.from_path(config_path)
    analyzer = TextAnalyzer(config)
    
    payload = analyzer.load_text(text_path)
    result = analyzer.analyze(payload)
    
    # Extract the aggregator from the analyzer
    # We need to reconstruct it from the results
    from src.text_analysis.aggregator import Aggregator
    aggregator = Aggregator()
    
    for module_name, module_data in result.items():
        aggregator.add(module_name, module_data)
    
    return aggregator


def main():
    if len(sys.argv) != 4:
        print("Usage: python compare_authors.py <text1.txt> <text2.txt> <config.yaml>")
        sys.exit(1)
    
    text1_path = Path(sys.argv[1])
    text2_path = Path(sys.argv[2])
    config_path = Path(sys.argv[3])
    
    print(f"Analyzing {text1_path.name}...")
    aggregator1 = analyze_text(text1_path, config_path)
    
    print(f"Analyzing {text2_path.name}...")
    aggregator2 = analyze_text(text2_path, config_path)
    
    print("\n" + "="*60)
    print("AUTHORSHIP COMPARISON ANALYSIS")
    print("="*60)
    
    # Generate style vectors and signatures
    vector1 = aggregator1.create_style_vector()
    vector2 = aggregator2.create_style_vector()
    
    signature1 = aggregator1.get_authorship_signature()
    signature2 = aggregator2.get_authorship_signature()
    
    print(f"\nText 1 ({text1_path.name}) - Authorship Signature:")
    print(f"  Features: {signature1['total_features']}")
    print(f"  Signature strength: {signature1['signature_strength']:.3f}")
    print("  Top distinguishing features:")
    for item in signature1['signature_summary'][:3]:
        print(f"    {item['category']}: {item['top_feature']} = {item['value']:.3f}")
    
    print(f"\nText 2 ({text2_path.name}) - Authorship Signature:")
    print(f"  Features: {signature2['total_features']}")
    print(f"  Signature strength: {signature2['signature_strength']:.3f}")
    print("  Top distinguishing features:")
    for item in signature2['signature_summary'][:3]:
        print(f"    {item['category']}: {item['top_feature']} = {item['value']:.3f}")
    
    # Compare the texts
    comparison = aggregator1.compare_authors(aggregator2)
    
    print(f"\n" + "-"*40)
    print("COMPARISON RESULTS")
    print("-"*40)
    print(f"Overall similarity: {comparison['overall_score']:.3f}")
    print(f"Interpretation: {comparison['interpretation']}")
    
    print("\nCategory-by-category analysis:")
    for category, data in comparison['feature_analysis'].items():
        print(f"  {category.capitalize()}: {data['similarity']:.3f} ({data['interpretation']})")
    
    print(f"\n" + "-"*40)
    print("DETAILED FEATURE COMPARISON")
    print("-"*40)
    
    # Show top differentiating features
    common_features = set(vector1.features.keys()) & set(vector2.features.keys())
    feature_diffs = []
    
    for feature in common_features:
        diff = abs(vector1.features[feature] - vector2.features[feature])
        feature_diffs.append((feature, diff, vector1.features[feature], vector2.features[feature]))
    
    # Sort by difference and show top 10
    feature_diffs.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 10 differentiating features:")
    for feature, diff, val1, val2 in feature_diffs[:10]:
        print(f"  {feature:<40} | {val1:8.3f} vs {val2:8.3f} (diff {diff:.3f})")


if __name__ == "__main__":
    main()