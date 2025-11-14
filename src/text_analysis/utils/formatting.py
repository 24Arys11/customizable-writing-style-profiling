"""Enhanced formatting utilities for clean, signal-focused output."""

from __future__ import annotations

from typing import Any, Dict, List
import statistics


def format_summary(report: Dict[str, Any], verbosity: str = "standard") -> str:
    """Format analysis results as clean, conclusion-focused summary."""
    lines: list[str] = []
    
    # Text overview
    metadata = report.get('metadata', {})
    lines.append("=" * 60)
    lines.append("TEXT ANALYSIS SUMMARY")
    lines.append("=" * 60)
    
    text_length = metadata.get('text_length', 'unknown')
    if isinstance(text_length, (int, float)):
        lines.append(f"Text length: {text_length:,} characters")
    else:
        lines.append(f"Text length: {text_length} characters")
    
    # High-level structure
    sentence_stats = report.get('sentence_stats', {})
    if sentence_stats:
        lines.append(f"Sentences: {sentence_stats.get('sentence_count', 0)}")
        lines.append(f"Paragraphs: {sentence_stats.get('paragraph_count', 0)}")
        lines.append(f"Avg sentence length: {sentence_stats.get('mean_sentence_length_tokens', 0):.1f} words")
    
    lines.append("")
    lines.append("-" * 40)
    lines.append("STYLE CHARACTERISTICS")
    lines.append("-" * 40)
    
    # Extract key insights from each module
    insights = _extract_insights(report, verbosity)
    for category, category_insights in insights.items():
        lines.append(f"\n{category.upper()}:")
        for insight in category_insights:
            lines.append(f"  • {insight}")
    
    return "\n".join(lines)


def format_table(report: Dict[str, Any], verbosity: str = "standard") -> str:
    """Format analysis results as compact table view."""
    lines = []
    lines.append("=" * 95)
    lines.append("AUTHORSHIP METRICS TABLE")
    lines.append("=" * 95)
    
    # Extract key metrics for table
    metrics = _extract_key_metrics(report, verbosity)
    
    # Format as aligned table with better column separation (no | inside data)
    max_metric_width = max(len(name) for name in metrics.keys()) if metrics else 20
    
    lines.append(f"{'METRIC':<{max_metric_width}}   {'VALUE':<18}   INTERPRETATION")
    lines.append("-" * 95)
    
    for metric_name, (value, interpretation) in metrics.items():
        lines.append(f"{metric_name:<{max_metric_width}}   {value:<18}   {interpretation}")
    
    return "\n".join(lines)


def _extract_insights(report: Dict[str, Any], verbosity: str = "standard") -> Dict[str, List[str]]:
    """Extract human-readable insights from analysis data with configurable verbosity."""
    insights = {}
    
    # Lexical sophistication
    lexical = report.get('lexical_metrics', {})
    lexical_insights = []
    
    ttr = lexical.get('type_token_ratio', 0)
    if ttr > 0.8:
        lexical_insights.append(f"High lexical diversity (TTR: {ttr:.3f}) - sophisticated vocabulary")
    elif ttr < 0.4:
        lexical_insights.append(f"Low lexical diversity (TTR: {ttr:.3f}) - repetitive vocabulary")
    else:
        lexical_insights.append(f"Moderate lexical diversity (TTR: {ttr:.3f})")
    
    yules_k = lexical.get('yules_k', 0)
    if yules_k > 100:
        lexical_insights.append(f"High vocabulary richness (Yule's K: {yules_k:.0f})")
    elif yules_k > 0:
        lexical_insights.append(f"Moderate vocabulary richness (Yule's K: {yules_k:.1f})")
    
    function_ratio = lexical.get('total_function_word_ratio', 0)
    lexical_insights.append(f"Function words: {function_ratio:.1%} (grammatical complexity indicator)")
    
    # Add vocabulary sophistication metrics
    vocab_soph = report.get('vocabulary_sophistication', {})
    if vocab_soph:
        avg_syllables = vocab_soph.get('avg_syllables_per_word', 0)
        formality = vocab_soph.get('formality_score', 0)
        sophistication = vocab_soph.get('vocabulary_sophistication', 0)
        if avg_syllables > 0:
            if formality > 0.01:
                register = "formal"
            elif formality < -0.01:
                register = "informal"
            else:
                register = "neutral"
            lexical_insights.append(f"Vocabulary sophistication: {sophistication:.2f} ({register} register, {avg_syllables:.1f} avg syllables)")
    
    insights['Vocabulary'] = lexical_insights
    
    # Sentence rhythm and flow
    rhythm = report.get('rhythm_metrics', {}).get('sentence_rhythm', {})
    rhythm_insights = []
    
    cv = rhythm.get('coefficient_of_variation', 0)
    std_dev = rhythm.get('std_dev', 0)
    mean_len = rhythm.get('mean_value', 0)
    
    if cv > 0.7:
        rhythm_insights.append(f"Highly variable sentence lengths (CV: {cv:.2f}, SD: {std_dev:.1f}) - dynamic rhythm")
    elif cv < 0.3:
        rhythm_insights.append(f"Consistent sentence lengths (CV: {cv:.2f}, SD: {std_dev:.1f}) - regular flow")
    else:
        rhythm_insights.append(f"Moderate sentence variation (CV: {cv:.2f}, SD: {std_dev:.1f})")
    
    autocorr = report.get('rhythm_metrics', {}).get('autocorrelation', {}).get('lag_1_autocorrelation', 0)
    if verbosity in ['standard', 'detailed']:
        rhythm_insights.append(f"Autocorrelation (lag-1): {autocorr:.3f} ({'structured' if abs(autocorr) > 0.3 else 'moderate'} patterns)")
    
    if autocorr > 0.3:
        rhythm_insights.append(f"Positive rhythm correlation - structured patterns")
    elif autocorr < -0.3:
        rhythm_insights.append(f"Negative rhythm correlation - alternating patterns")
    
    # Add burstiness info
    burstiness = report.get('rhythm_metrics', {}).get('burstiness_patterns', {}).get('sentence_burstiness', 0)
    if abs(burstiness) > 0.3:
        rhythm_insights.append(f"Burstiness: {burstiness:.2f} ({'clustered' if burstiness > 0 else 'dispersed'} patterns)")
    
    # Paragraph structure analysis
    sentence_stats = report.get('sentence_stats', {})
    paragraph_cv_words = sentence_stats.get('paragraph_length_coefficient_of_variation_words', 0)
    paragraph_mean_words = sentence_stats.get('paragraph_length_mean_words', 0)
    paragraph_cv_sentences = sentence_stats.get('paragraph_length_coefficient_of_variation_sentences', 0)
    
    if paragraph_mean_words > 0:  # Only if we have paragraph data
        if verbosity in ['standard', 'detailed']:
            rhythm_insights.append(f"Paragraph variation: {paragraph_cv_words:.2f} CV ({paragraph_mean_words:.0f} avg words)")
        
        if paragraph_cv_words > 0.8:
            rhythm_insights.append(f"Highly variable paragraph lengths - dynamic structure")
        elif paragraph_cv_words < 0.3:
            rhythm_insights.append(f"Consistent paragraph lengths - regular structure")
    
    insights['Rhythm & Flow'] = rhythm_insights
    
    # Emotional and perspective patterns
    sentiment = report.get('sentiment_metrics', {})
    sentiment_insights = []
    
    perspective = sentiment.get('person_perspective', {})
    first_pct = perspective.get('first_person_singular_ratio', 0) + perspective.get('first_person_plural_ratio', 0)
    second_pct = perspective.get('second_person_ratio', 0)
    third_pct = perspective.get('third_person_ratio', 0)
    
    # Person perspective breakdown
    sentiment_insights.append(f"Person perspective: {first_pct:.0%} 1st : {second_pct:.0%} 2nd : {third_pct:.0%} 3rd")
    
    emotional_diversity = sentiment.get('emotion_patterns', {}).get('emotional_diversity', 0)
    if emotional_diversity > 1.5:
        sentiment_insights.append(f"Rich emotional expression (diversity: {emotional_diversity:.2f})")
    
    polarity = sentiment.get('sentiment_distribution', {}).get('sentiment_polarity', 0)
    if polarity > 0.2:
        sentiment_insights.append(f"Generally positive tone (polarity: {polarity:.2f})")
    elif polarity < -0.2:
        sentiment_insights.append(f"Generally negative tone (polarity: {polarity:.2f})")
    else:
        sentiment_insights.append("Neutral emotional tone")
    
    insights['Voice & Tone'] = sentiment_insights
    
    # Phase 1 Enhancement: Certainty and Confidence Patterns
    confidence = sentiment.get('confidence_markers', {})
    certainty_insights = []
    
    uncertainty_ratio = confidence.get('uncertainty_ratio', 0)
    certainty_ratio = confidence.get('certainty_ratio', 0)
    hedging_ratio = confidence.get('hedging_ratio', 0)
    
    if uncertainty_ratio > 0.2:
        certainty_insights.append(f"High uncertainty markers: {uncertainty_ratio:.1%} of sentences contain tentative language")
    elif uncertainty_ratio > 0.1:
        certainty_insights.append(f"Moderate uncertainty: {uncertainty_ratio:.1%} of sentences show tentativeness")
    elif verbosity in ['standard', 'detailed'] and uncertainty_ratio > 0:
        certainty_insights.append(f"Low uncertainty: {uncertainty_ratio:.1%} of sentences contain uncertainty markers")
    
    if certainty_ratio > 0.1:
        certainty_insights.append(f"Strong certainty markers: {certainty_ratio:.1%} of sentences express high confidence")
    elif verbosity in ['standard', 'detailed'] and certainty_ratio > 0:
        certainty_insights.append(f"Certainty markers: {certainty_ratio:.1%} of sentences show confidence")
    
    if hedging_ratio > 0.05:
        certainty_insights.append(f"Frequent hedging: {hedging_ratio:.1%} of sentences use cautious language")
    elif verbosity in ['standard', 'detailed'] and hedging_ratio > 0:
        certainty_insights.append(f"Some hedging: {hedging_ratio:.1%} of sentences contain hedge words")
    
    if certainty_insights:
        insights['Confidence'] = certainty_insights
    
    # Structural patterns - SHOW ALL CONFIG PATTERNS regardless of frequency
    patterns = report.get('pattern_metrics', {})
    pattern_insights = []
    
    total_patterns = sum(
        data.get('total_occurrences', 0) 
        for data in patterns.values() 
        if isinstance(data, dict) and not data.get('_debug')
    )
    sentences = report.get('sentence_stats', {}).get('sentence_count', 1)
    pattern_density = total_patterns / sentences if sentences > 0 else 0
    
    if pattern_density > 0.3:
        pattern_insights.append(f"High rhetorical structure - frequent discourse patterns")
    elif pattern_density > 0.1:
        pattern_insights.append(f"Moderate rhetorical structure")
    else:
        pattern_insights.append(f"Minimal rhetorical patterns - simple structure")
    
    # Show ALL config patterns (including zero occurrences) - respecting user config
    all_pattern_counts = []
    for pattern_name, data in patterns.items():
        if isinstance(data, dict):
            count = data.get('total_occurrences', 0)
            clean_name = pattern_name.replace('_', ' ')
            all_pattern_counts.append((clean_name, count))
    
    if all_pattern_counts:
        all_pattern_counts.sort(key=lambda x: x[1], reverse=True)
        
        if verbosity == "detailed":
            # Show all patterns with counts
            all_patterns = []
            for pattern_name, count in all_pattern_counts:
                all_patterns.append(f"{pattern_name}: {count}")
            pattern_insights.append(f"All patterns: {', '.join(all_patterns)}")
        else:
            # Show top patterns, but mention if others exist with zero counts
            top_pattern = all_pattern_counts[0]
            pattern_insights.append(f"Most frequent: {top_pattern[0]} ({top_pattern[1]} occurrences)")
            
            # Compact view: top patterns + zero count summary
            active_patterns = [f"{name}: {count}" for name, count in all_pattern_counts[:5] if count > 0]
            zero_patterns = [name for name, count in all_pattern_counts if count == 0]
            
            if active_patterns:
                pattern_insights.append(f"Active patterns: {', '.join(active_patterns)}")
            if zero_patterns and verbosity != "compact":
                pattern_insights.append(f"Inactive patterns: {', '.join(zero_patterns[:3])}{'...' if len(zero_patterns) > 3 else ''}")
    
    insights['Structure'] = pattern_insights
    
    # Topic coherence and semantic flow
    semantic = report.get('semantic_metrics', {})
    topic_insights = []
    
    topic_data = semantic.get('topic_coherence', {})
    if topic_data:
        topic_drift = topic_data.get('topic_drift', 0)
        coherence = topic_data.get('coherence_score', 0)
        concentration = topic_data.get('topic_concentration', 0)
        
        if verbosity in ['standard', 'detailed']:
            topic_insights.append(f"Topic drift: {topic_drift:.2f} ({'high variation' if topic_drift > 0.7 else 'moderate' if topic_drift > 0.4 else 'focused'})")
        
        if coherence < 0.3:
            topic_insights.append(f"Low topic coherence ({coherence:.2f}) - scattered themes")
        elif coherence > 0.7:
            topic_insights.append(f"High topic coherence ({coherence:.2f}) - focused narrative")
        elif verbosity in ['standard', 'detailed']:
            topic_insights.append(f"Moderate topic coherence ({coherence:.2f})")
        
        if concentration > 0.1:
            topic_insights.append(f"Strong topic concentration - focused themes")
        elif verbosity == 'detailed':
            topic_insights.append(f"Topic concentration: {concentration:.3f}")
    
    if topic_insights:
        insights['Topic Flow'] = topic_insights
    
    # Punctuation style
    punct = report.get('punctuation_metrics', {})
    punct_insights = []
    
    punct_density = punct.get('punctuation_density', {}).get('punctuation_density', 0)
    punct_insights.append(f"Punctuation density: {punct_density:.3f} (writing formality indicator)")
    
    # Add specific punctuation breakdown for verbosity modes
    if verbosity in ['standard', 'detailed']:
        punct_counts = punct.get('punctuation_counts', {})
        total_tokens = report.get('lexical_metrics', {}).get('token_count', 1)
        comma_density = punct_counts.get(',', 0) / total_tokens * 1000
        period_density = punct_counts.get('.', 0) / total_tokens * 1000
        punct_insights.append(f"Comma density: {comma_density:.1f}/1000 tokens, Period density: {period_density:.1f}/1000")
        
        # Enhanced punctuation fingerprinting
        semicolon_count = punct_counts.get(';', 0)
        paren_count = punct_counts.get('(', 0) + punct_counts.get(')', 0)
        dash_count = punct_counts.get('-', 0) + punct_counts.get('—', 0) + punct_counts.get('–', 0)
        
        punct_features = []
        if semicolon_count > 0:
            punct_features.append(f"semicolons: {semicolon_count}")
        if paren_count > 5:
            punct_features.append(f"parentheses: {paren_count}")
        if dash_count > 3:
            punct_features.append(f"dashes: {dash_count}")
        
        if punct_features:
            punct_insights.append(f"Style marks: {', '.join(punct_features)}")
    
    dialogue_ratio = punct.get('quotation_analysis', {}).get('dialogue_ratio', 0)
    if dialogue_ratio > 0.1:
        punct_insights.append(f"Contains dialogue ({dialogue_ratio:.1%}) - narrative style")
    
    question_ratio = punct.get('sentence_punctuation', {}).get('question_ratio', 0)
    if question_ratio > 0.05:
        punct_insights.append(f"Frequent questions ({question_ratio:.1%}) - engaging style")
    
    insights['Style Markers'] = punct_insights
    
    # Phase 1 Enhancement: Syntactic Complexity
    syntax = report.get('syntax_metrics', {})
    syntax_insights = []
    
    # Look for complexity indicators from syntax analysis
    pos_stats = syntax.get('pos_distribution', {})
    if pos_stats:
        # Calculate complexity indicators
        subordinating_conj = pos_stats.get('SCONJ', 0) + pos_stats.get('ADP', 0)  # Subordinating conjunctions + prepositions
        total_tokens = sum(pos_stats.values()) if pos_stats.values() else 1
        subordination_ratio = subordinating_conj / total_tokens if total_tokens > 0 else 0
        
        if subordination_ratio > 0.15:
            syntax_insights.append(f"High syntactic complexity (subordination: {subordination_ratio:.1%})")
        elif subordination_ratio > 0.1:
            syntax_insights.append(f"Moderate syntactic complexity (subordination: {subordination_ratio:.1%})")
        else:
            syntax_insights.append(f"Simple syntax (subordination: {subordination_ratio:.1%})")
        
        # Enhanced syntactic analysis for detailed verbosity
        if verbosity == 'detailed':
            # Coordination vs subordination balance
            coord_conj = pos_stats.get('CCONJ', 0)  # Coordinating conjunctions
            subord_conj = pos_stats.get('SCONJ', 0)  # Subordinating conjunctions
            
            if subord_conj > coord_conj:
                syntax_insights.append(f"Subordination-heavy style ({subord_conj} sub vs {coord_conj} coord)")
            elif coord_conj > subord_conj:
                syntax_insights.append(f"Coordination-heavy style ({coord_conj} coord vs {subord_conj} sub)")
            
            # Sentence structure complexity
            complexity_metrics = syntax.get('syntactic_complexity', {})
            mean_depth = complexity_metrics.get('mean_tree_depth', 0)
            if mean_depth > 0:
                syntax_insights.append(f"Parse depth: {mean_depth:.1f} (structural complexity)")
        
        # Passive voice detection using dependency parsing
        dependency_stats = syntax.get('dependency_distribution', {})
        if dependency_stats:
            passive_subjects = dependency_stats.get('nsubjpass', 0)  # Passive nominal subjects
            passive_aux = dependency_stats.get('auxpass', 0)  # Passive auxiliary
            total_sentences = report.get('sentence_stats', {}).get('sentence_count', 1)
            
            # Use the more reliable indicator
            passive_sentences = max(passive_subjects, passive_aux)
            passive_ratio = passive_sentences / total_sentences if total_sentences > 0 else 0
            
            if verbosity in ['standard', 'detailed'] and passive_ratio > 0:
                syntax_insights.append(f"Passive voice: {passive_ratio:.1%} of sentences")
            
            if passive_ratio > 0.2:
                syntax_insights.append(f"Frequent passive voice - formal style")
            elif passive_ratio > 0.1:
                syntax_insights.append(f"Moderate passive voice usage")
        
        # Legacy passive estimation fallback
        else:
            aux_verbs = pos_stats.get('AUX', 0)
            past_part = pos_stats.get('VERB', 0) * 0.3  # Rough estimate of past participles
            passive_estimate = min(aux_verbs, past_part) / total_tokens if total_tokens > 0 else 0
            
            if passive_estimate > 0.1:
                syntax_insights.append(f"Frequent passive constructions (~{passive_estimate:.1%})")
            elif passive_estimate > 0.05:
                syntax_insights.append(f"Moderate passive voice (~{passive_estimate:.1%})")
        
        # Sentence type distribution analysis
        if verbosity == 'detailed' and dependency_stats:
            # Simple sentences: minimal coordination/subordination
            coord_conj = dependency_stats.get('cc', 0)  # Coordinating conjunctions
            subord_conj = dependency_stats.get('mark', 0) + dependency_stats.get('advcl', 0)  # Subordinate markers + adverbial clauses
            total_sentences = report.get('sentence_stats', {}).get('sentence_count', 1)
            
            # Rough classification based on coordination/subordination density
            coordination_density = coord_conj / total_sentences if total_sentences > 0 else 0
            subordination_density = subord_conj / total_sentences if total_sentences > 0 else 0
            
            if coordination_density > 0.3:
                syntax_insights.append(f"High compound sentence usage (coordination: {coordination_density:.1f}/sent)")
            if subordination_density > 0.5:
                syntax_insights.append(f"High complex sentence usage (subordination: {subordination_density:.1f}/sent)")
    
    if syntax_insights:
        insights['Syntax'] = syntax_insights
    
    # Stylistic Devices Analysis
    stylistic = report.get('stylistic_devices', {})
    if stylistic and verbosity in ['detailed']:
        stylistic_insights = []
        alliteration = stylistic.get('alliteration_instances', 0)
        repetition = stylistic.get('word_repetition_count', 0)
        metaphors = stylistic.get('metaphors_similes', 0)
        density = stylistic.get('stylistic_density', 0)
        variety = stylistic.get('sentence_variety_score', 0)
        
        if density > 2.0:
            stylistic_insights.append(f"Rich stylistic devices ({density:.1f} devices per 100 words)")
        elif density > 1.0:
            stylistic_insights.append(f"Moderate stylistic devices ({density:.1f} devices per 100 words)")
        else:
            stylistic_insights.append(f"Plain stylistic devices ({density:.1f} devices per 100 words)")
        
        if alliteration > 100:
            stylistic_insights.append(f"High alliteration usage ({alliteration} instances)")
        elif alliteration > 20:
            stylistic_insights.append(f"Moderate alliteration usage ({alliteration} instances)")
        
        if repetition > 50:
            stylistic_insights.append(f"Significant repetition patterns ({repetition} instances)")
        
        if metaphors > 5:
            stylistic_insights.append(f"Frequent metaphorical language ({metaphors} instances)")
        
        if variety > 0.7:
            stylistic_insights.append(f"High sentence variety (score: {variety:.2f})")
        elif variety < 0.4:
            stylistic_insights.append(f"Low sentence variety (score: {variety:.2f})")
        
        if stylistic_insights:
            insights['Stylistic Devices'] = stylistic_insights
    
    # Discourse Flow Analysis
    discourse = report.get('discourse_flow', {})
    if discourse and verbosity in ['detailed']:
        discourse_insights = []
        transition_density = discourse.get('transition_density', 0)
        flow_consistency = discourse.get('flow_consistency', 0)
        topic_consistency = discourse.get('topic_consistency', 0)
        
        if transition_density > 8.0:
            discourse_insights.append(f"High transition word usage ({transition_density:.1f} per 100 words)")
        elif transition_density < 3.0:
            discourse_insights.append(f"Minimal transition markers ({transition_density:.1f} per 100 words)")
        
        if flow_consistency > 0.6:
            discourse_insights.append(f"Well-connected discourse (flow: {flow_consistency:.2f})")
        elif flow_consistency < 0.2:
            discourse_insights.append(f"Fragmented discourse (flow: {flow_consistency:.2f})")
        
        if topic_consistency > 0.7:
            discourse_insights.append(f"Highly coherent topics (consistency: {topic_consistency:.2f})")
        elif topic_consistency < 0.3:
            discourse_insights.append(f"Topic fragmentation (consistency: {topic_consistency:.2f})")
        
        if discourse_insights:
            insights['Discourse Flow'] = discourse_insights
    
    # AI Detection Analysis
    ai_detection = report.get('ai_detection', {})
    if ai_detection and verbosity in ['detailed']:
        ai_insights = []
        ai_likelihood = ai_detection.get('ai_likelihood_score', 0)
        pattern_density = ai_detection.get('ai_pattern_density', 0)
        sycophancy = ai_detection.get('sycophancy_count', 0)
        
        if ai_likelihood > 0.6:
            ai_insights.append(f"High AI likelihood (score: {ai_likelihood:.2f})")
        elif ai_likelihood > 0.4:
            ai_insights.append(f"Moderate AI characteristics (score: {ai_likelihood:.2f})")
        elif ai_likelihood < 0.2:
            ai_insights.append(f"Strong human characteristics (score: {ai_likelihood:.2f})")
        
        if pattern_density > 5.0:
            ai_insights.append(f"High AI pattern density ({pattern_density:.1f}%)")
        
        if sycophancy > 5:
            ai_insights.append(f"Sycophantic language detected ({sycophancy} instances)")
        
        # Flattery detection from sentiment
        sentiment = report.get('sentiment_metrics', {})
        flattery = sentiment.get('flattery_patterns', {}) if sentiment else {}
        if flattery:
            flattery_density = flattery.get('flattery_density', 0)
            if flattery_density > 0.01:
                ai_insights.append(f"Excessive flattery detected (density: {flattery_density:.3f})")
        
        if ai_insights:
            insights['AI Detection'] = ai_insights
    
    return insights


def _extract_key_metrics(report: Dict[str, Any], verbosity: str = "standard") -> Dict[str, tuple]:
    """Extract comprehensive metrics for table format with interpretations."""
    metrics = {}
    
    # Textual Properties
    sentence_stats = report.get('sentence_stats', {})
    metadata = report.get('metadata', {})
    
    metrics['Text Length'] = (f"{metadata.get('text_length', 0):,}", "characters")
    metrics['Sentences'] = (f"{sentence_stats.get('sentence_count', 0)}", "total")
    
    # Add paragraphs (missing from current table)
    paragraph_count = sentence_stats.get('paragraph_count', 0)
    if paragraph_count > 0:
        metrics['Paragraphs'] = (f"{paragraph_count}", "total")
    
    metrics['Avg Sentence'] = (f"{sentence_stats.get('mean_sentence_length_tokens', 0):.1f}±{sentence_stats.get('std_sentence_length_tokens', 0):.1f}", "words")
    
    # Core authorship indicators
    lexical = report.get('lexical_metrics', {})
    ttr = lexical.get('type_token_ratio', 0)
    metrics['Lexical Diversity'] = (f"{ttr:.3f}", _interpret_ttr(ttr))
    
    # Add Yule's K (missing from current table)
    yules_k = lexical.get('yules_k', 0)
    if yules_k > 0:
        metrics['Yules K'] = (f"{yules_k:.1f}", _interpret_yules_k(yules_k))
    
    # Perspective breakdown
    sentiment = report.get('sentiment_metrics', {})
    perspective = sentiment.get('person_perspective', {})
    first_pct = perspective.get('first_person_singular_ratio', 0) + perspective.get('first_person_plural_ratio', 0)
    second_pct = perspective.get('second_person_ratio', 0)
    third_pct = perspective.get('third_person_ratio', 0)
    metrics['Perspective'] = (f"{first_pct:.0%}:{second_pct:.0%}:{third_pct:.0%}", "1st:2nd:3rd")
    
    # Confidence indicators
    confidence = sentiment.get('confidence_markers', {})
    uncertainty_ratio = confidence.get('uncertainty_ratio', 0)
    certainty_ratio = confidence.get('certainty_ratio', 0)
    metrics['Confidence'] = (f"C{certainty_ratio:.1%} U{uncertainty_ratio:.1%}", "certain/uncertain")
    
    # Rhythm & Variability with comprehensive data
    rhythm = report.get('rhythm_metrics', {}).get('sentence_rhythm', {})
    cv = rhythm.get('coefficient_of_variation', 0)
    burstiness = report.get('rhythm_metrics', {}).get('burstiness_patterns', {}).get('sentence_burstiness', 0)
    metrics['Rhythm Variation'] = (f"CV:{cv:.2f} B:{burstiness:.2f}", _interpret_rhythm_cv(cv))
    
    # Add autocorrelation (missing from current table)
    autocorr = report.get('rhythm_metrics', {}).get('autocorrelation', {}).get('lag_1_autocorrelation', 0)
    if autocorr != 0:
        metrics['Autocorrelation'] = (f"{autocorr:.3f}", _interpret_autocorr(autocorr))
    
    # Paragraph variation (missing from current table)
    paragraph_cv = sentence_stats.get('paragraph_length_coefficient_of_variation_words', 0)
    paragraph_mean = sentence_stats.get('paragraph_length_mean_words', 0)
    if paragraph_mean > 0:
        metrics['Paragraph Var'] = (f"CV:{paragraph_cv:.2f} ({paragraph_mean:.0f}w)", "variation (avg words)")
    
    # Syntactic complexity with more detail
    syntax = report.get('syntax_metrics', {})
    pos_stats = syntax.get('pos_distribution', {})
    if pos_stats:
        subordinating_conj = pos_stats.get('SCONJ', 0) + pos_stats.get('ADP', 0)
        total_tokens = sum(pos_stats.values()) if pos_stats.values() else 1
        subordination_ratio = subordinating_conj / total_tokens if total_tokens > 0 else 0
        metrics['Syntax Complexity'] = (f"{subordination_ratio:.1%}", "subordination ratio")
    
    # Add passive voice detection (missing from current table)
    dependency_stats = syntax.get('dependency_distribution', {})
    if dependency_stats:
        passive_subjects = dependency_stats.get('nsubjpass', 0)
        passive_aux = dependency_stats.get('auxpass', 0)
        total_sentences = sentence_stats.get('sentence_count', 1)
        passive_sentences = max(passive_subjects, passive_aux)
        passive_ratio = passive_sentences / total_sentences if total_sentences > 0 else 0
        if passive_ratio > 0:
            metrics['Passive Voice'] = (f"{passive_ratio:.1%}", "of sentences")
    
    # Pattern density
    patterns = report.get('pattern_metrics', {})
    total_patterns = sum(data.get('total_occurrences', 0) for data in patterns.values() if isinstance(data, dict))
    sentences = sentence_stats.get('sentence_count', 1)
    pattern_density = total_patterns / sentences if sentences > 0 else 0
    metrics['Rhetorical Density'] = (f"{pattern_density:.3f}", _interpret_pattern_density(pattern_density))
    
    # Function words
    function_ratio = lexical.get('total_function_word_ratio', 0)
    metrics['Function Words'] = (f"{function_ratio:.1%}", _interpret_function_words(function_ratio))
    
    # Punctuation style with more detail
    punct = report.get('punctuation_metrics', {})
    punct_density = punct.get('punctuation_density', {}).get('punctuation_density', 0)
    dialogue_ratio = punct.get('quotation_analysis', {}).get('dialogue_ratio', 0)
    metrics['Punctuation'] = (f"D{punct_density:.3f} Q{dialogue_ratio:.1%}", "density+dialogue")
    
    # Add specific punctuation details (missing from current table)
    if verbosity in ['standard', 'detailed']:
        punct_counts = punct.get('punctuation_counts', {})
        total_tokens = lexical.get('token_count', 1)
        if total_tokens > 0:
            comma_density = punct_counts.get(',', 0) / total_tokens * 1000
            period_density = punct_counts.get('.', 0) / total_tokens * 1000
            metrics['Comma/Period'] = (f"{comma_density:.1f}/{period_density:.1f}", "per 1000 words")
    
    # Add topic coherence (missing from current table)
    semantic = report.get('semantic_metrics', {})
    topic_data = semantic.get('topic_coherence', {})
    if topic_data:
        topic_drift = topic_data.get('topic_drift', 0)
        coherence = topic_data.get('coherence_score', 0)
        metrics['Topic Flow'] = (f"D{topic_drift:.2f} C{coherence:.2f}", "drift/coherence")
    
    # Emotional tone (missing from current table)
    polarity = sentiment.get('sentiment_distribution', {}).get('sentiment_polarity', 0)
    if polarity != 0:
        metrics['Emotional Tone'] = (f"{polarity:+.2f}", _interpret_polarity(polarity))
    
    # Vocabulary Sophistication (NEW)
    vocab_soph = report.get('vocabulary_sophistication', {})
    if vocab_soph:
        avg_syllables = vocab_soph.get('avg_syllables_per_word', 0)
        formality = vocab_soph.get('formality_score', 0)
        sophistication = vocab_soph.get('vocabulary_sophistication', 0)
        if avg_syllables > 0:
            metrics['Vocabulary Sophistication'] = (f"Syl:{avg_syllables:.1f} F:{formality:+.2f} S:{sophistication:.2f}", "syllables/formality/sophistication")
    
    # Stylistic Devices (NEW)
    stylistic = report.get('stylistic_devices', {})
    if stylistic:
        alliteration = stylistic.get('alliteration_instances', 0)
        repetition = stylistic.get('word_repetition_count', 0)
        metaphors = stylistic.get('metaphors_similes', 0)
        density = stylistic.get('stylistic_density', 0)
        if density > 0:
            metrics['Stylistic Devices'] = (f"A:{alliteration} R:{repetition} M:{metaphors} D:{density}", "alliteration/repetition/metaphors/density")
    
    # Discourse Flow (NEW)
    discourse = report.get('discourse_flow', {})
    if discourse:
        transition_density = discourse.get('transition_density', 0)
        flow_consistency = discourse.get('flow_consistency', 0)
        coherence = discourse.get('topic_consistency', 0)
        if transition_density > 0:
            metrics['Discourse Flow'] = (f"T:{transition_density:.1f} F:{flow_consistency:.2f} C:{coherence:.2f}", "transitions/flow/coherence")
    
    # Flattery Detection (NEW)
    flattery = sentiment.get('flattery_patterns', {})
    if flattery:
        flattery_density = flattery.get('flattery_density', 0)
        sycophancy = flattery.get('sycophancy_score', 0)
        if flattery_density > 0 or sycophancy > 0:
            metrics['Flattery/Praise'] = (f"F:{flattery_density:.3f} S:{sycophancy:.3f}", "flattery/sycophancy density")
    
    # AI Detection (NEW)
    ai_detection = report.get('ai_detection', {})
    if ai_detection:
        ai_likelihood = ai_detection.get('ai_likelihood_score', 0)
        pattern_density = ai_detection.get('ai_pattern_density', 0)
        confidence_level = ai_detection.get('confidence_level', '')
        if ai_likelihood > 0:
            metrics['AI Likelihood'] = (f"{ai_likelihood:.2f} ({pattern_density:.1f}%)", confidence_level.replace('confidence ', ''))

    # Top patterns with more comprehensive display
    pattern_counts = []
    for pattern_name, data in patterns.items():
        if isinstance(data, dict):
            count = data.get('total_occurrences', 0)
            if count > 0:
                clean_name = pattern_name.replace('_', ' ')[:12]
                pattern_counts.append((clean_name, count))
    
    if pattern_counts:
        pattern_counts.sort(key=lambda x: x[1], reverse=True)
        if verbosity == 'detailed':
            # Show all active patterns
            all_patterns = [f"{name}:{count}" for name, count in pattern_counts]
            metrics['All Patterns'] = (" + ".join(all_patterns), "name:count")
        else:
            # Show top 3 patterns
            top_3 = pattern_counts[:3]
            pattern_str = " + ".join(f"{name}:{count}" for name, count in top_3)
            metrics['Top Patterns'] = (pattern_str, "name:count")
    
    # Show zero patterns if in detailed mode (missing from current table)
    if verbosity == 'detailed':
        zero_patterns = []
        for pattern_name, data in patterns.items():
            if isinstance(data, dict) and data.get('total_occurrences', 0) == 0:
                clean_name = pattern_name.replace('_', ' ')
                zero_patterns.append(clean_name)
        
        if zero_patterns:
            metrics['Zero Patterns'] = (", ".join(zero_patterns[:5]), "inactive patterns")
    
    return metrics


def _interpret_ttr(ttr: float) -> str:
    if ttr > 0.8: return "Very diverse"
    elif ttr > 0.6: return "Diverse"
    elif ttr > 0.4: return "Moderate"
    else: return "Repetitive"


def _interpret_rhythm_cv(cv: float) -> str:
    if cv > 0.8: return "Very dynamic"
    elif cv > 0.6: return "Dynamic"
    elif cv > 0.4: return "Varied"
    else: return "Regular"


def _interpret_pattern_density(density: float) -> str:
    if density > 0.3: return "High structure"
    elif density > 0.1: return "Moderate structure"
    else: return "Simple structure"


def _interpret_function_words(ratio: float) -> str:
    if ratio > 0.45: return "High complexity"
    elif ratio > 0.35: return "Moderate complexity"
    else: return "Simple grammar"


def _interpret_subordination(ratio: float) -> str:
    if ratio > 0.15: return "Complex syntax"
    elif ratio > 0.1: return "Moderate syntax"
    else: return "Simple syntax"


def _interpret_yules_k(k: float) -> str:
    if k > 200: return "Very rich"
    elif k > 100: return "Rich"
    elif k > 50: return "Moderate"
    else: return "Limited"


def _interpret_cv(cv: float) -> str:
    if cv > 0.8: return "Very dynamic"
    elif cv > 0.6: return "Dynamic"
    elif cv > 0.4: return "Varied"
    else: return "Regular"


def _interpret_autocorr(autocorr: float) -> str:
    if autocorr > 0.4: return "Structured"
    elif autocorr > 0.1: return "Slightly structured"
    elif autocorr < -0.4: return "Alternating"
    elif autocorr < -0.1: return "Slightly alternating"
    else: return "Random"


def _interpret_polarity(polarity: float) -> str:
    if polarity > 0.3: return "Positive"
    elif polarity > 0.1: return "Slightly positive"
    elif polarity < -0.3: return "Negative"
    elif polarity < -0.1: return "Slightly negative"
    else: return "Neutral"


def _interpret_function_ratio(ratio: float) -> str:
    if ratio > 0.45: return "Complex grammar"
    elif ratio > 0.35: return "Moderate grammar"
    else: return "Simple grammar"


# Legacy functions for backward compatibility
def _format_section(name: str, data: Any, indent: int = 0) -> list[str]:
    """Legacy formatter for detailed output."""
    indent_str = "  " * indent
    formatted: list[str] = []

    if isinstance(data, dict):
        formatted.append(f"{indent_str}{name}:")
        scalar_items = {
            key: value
            for key, value in data.items()
            if not isinstance(value, (dict, list, tuple))
        }
        if scalar_items:
            scalar_line = ", ".join(
                f"{key}={_format_value(value)}" for key, value in scalar_items.items()
            )
            formatted.append(f"{indent_str}  {scalar_line}")
        for key, value in data.items():
            if isinstance(value, dict):
                formatted.extend(_format_section(key, value, indent + 1))
            elif isinstance(value, (list, tuple)):
                list_values = ", ".join(_format_value(item) for item in value)
                formatted.append(f"{indent_str}  {key}: {list_values}")
    elif isinstance(data, (list, tuple)):
        formatted.append(f"{indent_str}{name}: {', '.join(_format_value(item) for item in data)}")
    else:
        formatted.append(f"{indent_str}{name}: {_format_value(data)}")

    return formatted


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
