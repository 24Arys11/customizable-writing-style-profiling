#!/usr/bin/env python3
"""
Style fingerprinting utility for individual text analysis.
Usage: python style_fingerprint.py <text.txt> <config.yaml>
"""

import sys
from pathlib import Path
import json

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.text_analysis.config_manager import AnalysisConfig
from src.text_analysis.text_analyzer import TextAnalyzer


def main():
    if len(sys.argv) != 3:
        print("Usage: python style_fingerprint.py <text.txt> <config.yaml>")
        sys.exit(1)
    
    text_path = Path(sys.argv[1])
    config_path = Path(sys.argv[2])
    
    print(f"Generating style fingerprint for: {text_path.name}")
    print("="*60)
    
    # Analyze the text
    config = AnalysisConfig.from_path(config_path)
    analyzer = TextAnalyzer(config)
    
    payload = analyzer.load_text(text_path)
    result = analyzer.analyze(payload)
    
    # Create aggregator and style vector
    from src.text_analysis.aggregator import Aggregator
    aggregator = Aggregator()
    
    for module_name, module_data in result.items():
        aggregator.add(module_name, module_data)
    
    # Generate authorship signature
    signature = aggregator.get_authorship_signature()
    vector = aggregator.create_style_vector()
    
    print(f"Text: {text_path.name}")
    print(f"Length: {result.get('metadata', {}).get('text_length', 'unknown')} characters")
    print(f"Sentences: {result.get('sentence_stats', {}).get('sentence_count', 'unknown')}")
    print(f"Total features extracted: {signature['total_features']}")
    print(f"Signature strength: {signature['signature_strength']:.3f}")
    
    print("\n" + "-"*40)
    print("AUTHORSHIP SIGNATURE")
    print("-"*40)
    
    print("Top distinguishing features by category:")
    for item in signature['signature_summary']:
        print(f"  {item['category'].upper():<12} | {item['top_feature']:<35} = {item['value']:8.3f}")
    
    print("\n" + "-"*40)
    print("DETAILED STYLE PROFILE")
    print("-"*40)
    
    # Show category breakdowns
    categories = signature['category_breakdown']
    
    for category, features in categories.items():
        if features and category != 'other':
            print(f"\n{category.upper()} features ({len(features)}):")
            # Show top 5 features in each category
            for feature_name, value in features[:5]:
                feature_short = feature_name.split('_')[-2:] if '_' in feature_name else feature_name
                feature_display = '_'.join(feature_short)
                print(f"  {feature_display:<30} {value:8.3f}")
    
    print("\n" + "-"*40)
    print("STYLE CHARACTERISTICS SUMMARY")
    print("-"*40)
    
    # Generate human-readable insights
    insights = generate_style_insights(result, signature)
    for insight in insights:
        print(f"• {insight}")
    
    # Save detailed fingerprint to JSON
    output_file = text_path.with_suffix('.fingerprint.json')
    fingerprint_data = {
        'text_file': str(text_path),
        'signature': signature,
        'style_vector': vector.features,
        'metadata': vector.metadata,
        'insights': insights
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fingerprint_data, f, indent=2)
    
    print(f"\nDetailed fingerprint saved to: {output_file}")


def generate_style_insights(analysis_data: dict, signature: dict) -> list:
    """Generate human-readable style insights from analysis data."""
    insights = []
    
    # Lexical sophistication
    lexical = analysis_data.get('lexical_metrics', {})
    ttr = lexical.get('type_token_ratio', 0)
    if ttr > 0.8:
        insights.append(f"High lexical diversity (TTR: {ttr:.3f}) suggests sophisticated vocabulary")
    elif ttr < 0.5:
        insights.append(f"Lower lexical diversity (TTR: {ttr:.3f}) indicates repetitive word usage")
    
    # Sentence rhythm
    rhythm = analysis_data.get('rhythm_metrics', {}).get('sentence_rhythm', {})
    cv = rhythm.get('coefficient_of_variation', 0)
    if cv > 0.7:
        insights.append(f"Highly variable sentence lengths (CV: {cv:.3f}) create dynamic rhythm")
    elif cv < 0.3:
        insights.append(f"Consistent sentence lengths (CV: {cv:.3f}) suggest regular prose style")
    
    # Person perspective
    perspective = analysis_data.get('sentiment_metrics', {}).get('person_perspective', {})
    dominant = perspective.get('dominant_perspective', '')
    if 'first_person' in dominant:
        insights.append("Predominantly first-person narration suggests personal/memoir style")
    elif 'third_person' in dominant:
        insights.append("Third-person narration indicates formal/narrative prose style")
    
    # Emotional patterns
    emotion = analysis_data.get('sentiment_metrics', {}).get('emotion_patterns', {})
    emotional_diversity = emotion.get('emotional_diversity', 0)
    if emotional_diversity > 2:
        insights.append(f"Rich emotional vocabulary (diversity: {emotional_diversity:.2f}) suggests expressive writing")
    
    # Punctuation style
    punct = analysis_data.get('punctuation_metrics', {}).get('punctuation_density', {})
    question_ratio = analysis_data.get('punctuation_metrics', {}).get('sentence_punctuation', {}).get('question_ratio', 0)
    if question_ratio > 0.1:
        insights.append(f"Frequent rhetorical questions ({question_ratio:.1%}) suggest engaging style")
    
    # Pattern usage
    patterns = analysis_data.get('pattern_metrics', {})
    total_patterns = sum(data.get('total_occurrences', 0) for data in patterns.values() if isinstance(data, dict))
    sentences = analysis_data.get('sentence_stats', {}).get('sentence_count', 1)
    pattern_density = total_patterns / sentences
    if pattern_density > 0.2:
        insights.append(f"Frequent rhetorical patterns suggest structured argumentative style")
    
    return insights


if __name__ == "__main__":
    main()