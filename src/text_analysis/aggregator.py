"""Advanced aggregation system for style vectors and authorship analysis."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import statistics


class StyleVector:
    """Represents a normalized style fingerprint for authorship analysis."""
    
    def __init__(self, features: Dict[str, float], metadata: Optional[Dict[str, Any]] = None):
        self.features = features
        self.metadata = metadata or {}
        self._normalize_features()
    
    def _normalize_features(self) -> None:
        """Normalize features to [0, 1] range using min-max scaling where appropriate."""
        # Features that should be bounded to [0, 1] are ratios and coefficients
        ratio_features = [k for k in self.features.keys() if 'ratio' in k or 'coefficient' in k]
        
        for feature in ratio_features:
            value = self.features[feature]
            # Clamp to [0, 1] for ratio features
            self.features[feature] = max(0.0, min(1.0, value))
    
    def similarity(self, other: 'StyleVector', weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate cosine similarity between two style vectors."""
        common_features = set(self.features.keys()) & set(other.features.keys())
        
        if not common_features:
            return 0.0
        
        # Calculate weighted cosine similarity
        dot_product = 0.0
        norm_a = 0.0
        norm_b = 0.0
        
        for feature in common_features:
            weight = weights.get(feature, 1.0) if weights else 1.0
            a_val = self.features[feature] * weight
            b_val = other.features[feature] * weight
            
            dot_product += a_val * b_val
            norm_a += a_val ** 2
            norm_b += b_val ** 2
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))
    
    def distance(self, other: 'StyleVector', weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate Euclidean distance between style vectors."""
        common_features = set(self.features.keys()) & set(other.features.keys())
        
        if not common_features:
            return float('inf')
        
        squared_diff = 0.0
        for feature in common_features:
            weight = weights.get(feature, 1.0) if weights else 1.0
            diff = (self.features[feature] - other.features[feature]) * weight
            squared_diff += diff ** 2
        
        return math.sqrt(squared_diff)
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N features by absolute value."""
        sorted_features = sorted(self.features.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_features[:n]


class FeatureSelector:
    """Intelligent feature selection for authorship analysis."""
    
    AUTHORSHIP_FEATURE_WEIGHTS = {
        # High-signal authorship indicators
        'type_token_ratio': 2.0,
        'yules_k': 2.5,
        'simpsons_d': 2.0,
        'coefficient_of_variation': 1.8,
        'function_word_ratio': 2.2,
        'lag_1_autocorrelation': 1.5,
        'burstiness_coefficient': 1.6,
        'sentiment_polarity': 1.2,
        'dominant_perspective': 1.8,
        'temporal_flow_density': 1.3,
        'pattern_regularity': 1.4,
        
        # Medium-signal indicators
        'punctuation_density': 1.0,
        'dialogue_ratio': 1.1,
        'question_ratio': 1.0,
        'exclamation_ratio': 1.0,
        'rhetorical_questions_total': 1.2,
        'discourse_connectors_total': 1.1,
        'semantic_density': 1.0,
        'topic_concentration': 1.1,
        
        # Lower-signal but still useful
        'sentence_count': 0.5,
        'paragraph_count': 0.5,
        'text_length': 0.3,
        'word_length_mean': 0.8,
        'hapax_ratio': 0.9,
        'entity_density': 0.7,
    }
    
    NOISE_PATTERNS = {
        # Features likely to be noisy or content-dependent
        'total_occurrences', 'count', '_total', 'min_value', 'max_value',
        'sample_size', 'entity_count', 'vocabulary_size'
    }
    
    @classmethod
    def is_signal_feature(cls, feature_name: str) -> bool:
        """Determine if a feature is likely to contain authorship signal."""
        # Exclude noisy patterns
        for pattern in cls.NOISE_PATTERNS:
            if pattern in feature_name:
                return False
        
        # Prefer normalized metrics (ratios, coefficients, densities)
        signal_indicators = ['ratio', 'coefficient', 'density', 'diversity', 'entropy', 
                           'polarity', 'balance', 'regularity', 'correlation']
        
        return any(indicator in feature_name for indicator in signal_indicators)
    
    @classmethod
    def get_feature_weight(cls, feature_name: str) -> float:
        """Get importance weight for a feature."""
        # Direct match
        if feature_name in cls.AUTHORSHIP_FEATURE_WEIGHTS:
            return cls.AUTHORSHIP_FEATURE_WEIGHTS[feature_name]
        
        # Pattern-based weights
        if 'ratio' in feature_name:
            return 1.2
        elif 'coefficient' in feature_name:
            return 1.1
        elif 'diversity' in feature_name or 'entropy' in feature_name:
            return 1.3
        elif 'correlation' in feature_name:
            return 1.4
        
        return 1.0


class Aggregator:
    """Advanced aggregation system for style vectors and authorship analysis."""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}
        self._feature_selector = FeatureSelector()

    def add(self, module_name: str, metrics: Dict[str, Any]) -> None:
        """Add module metrics to aggregation."""
        self._data[module_name] = metrics

    def finalize(self) -> Dict[str, Any]:
        """Return basic aggregated data."""
        return self._data
    
    def extract_feature_vector(self, include_metadata: bool = True) -> Dict[str, float]:
        """Extract flattened feature vector from all module outputs."""
        features = {}
        
        def flatten_dict(d: Dict[str, Any], prefix: str = "") -> None:
            for key, value in d.items():
                full_key = f"{prefix}_{key}" if prefix else key
                
                if isinstance(value, dict):
                    flatten_dict(value, full_key)
                elif isinstance(value, (int, float)) and not math.isnan(value):
                    features[full_key] = float(value)
        
        for module_name, module_data in self._data.items():
            if module_name == 'metadata' and not include_metadata:
                continue
            flatten_dict(module_data, module_name)
        
        return features
    
    def create_style_vector(self, signal_only: bool = True, weighted: bool = True) -> StyleVector:
        """Create a StyleVector from current analysis data."""
        all_features = self.extract_feature_vector(include_metadata=False)
        
        if signal_only:
            # Filter to high-signal features only
            signal_features = {
                name: value for name, value in all_features.items()
                if self._feature_selector.is_signal_feature(name)
            }
        else:
            signal_features = all_features
        
        # Apply feature weights if requested
        if weighted:
            weighted_features = {}
            for name, value in signal_features.items():
                weight = self._feature_selector.get_feature_weight(name)
                weighted_features[name] = value * weight
            signal_features = weighted_features
        
        metadata = {
            'feature_count': len(signal_features),
            'total_available_features': len(all_features),
            'signal_ratio': len(signal_features) / len(all_features) if all_features else 0,
            'weighted': weighted,
            'signal_only': signal_only
        }
        
        return StyleVector(signal_features, metadata)
    
    def compare_authors(self, other_aggregator: 'Aggregator', 
                       method: str = 'similarity') -> Dict[str, Any]:
        """Compare style between two texts/authors."""
        vector_a = self.create_style_vector()
        vector_b = other_aggregator.create_style_vector()
        
        if method == 'similarity':
            score = vector_a.similarity(vector_b)
            interpretation = self._interpret_similarity(score)
        elif method == 'distance':
            score = vector_a.distance(vector_b)
            interpretation = self._interpret_distance(score)
        else:
            raise ValueError(f"Unknown comparison method: {method}")
        
        # Feature-level analysis
        feature_comparison = self._compare_feature_categories(vector_a, vector_b)
        
        return {
            'comparison_method': method,
            'overall_score': score,
            'interpretation': interpretation,
            'feature_analysis': feature_comparison,
            'vector_a_metadata': vector_a.metadata,
            'vector_b_metadata': vector_b.metadata
        }
    
    def _interpret_similarity(self, similarity: float) -> str:
        """Interpret cosine similarity score for authorship analysis."""
        if similarity >= 0.9:
            return "Very strong stylistic similarity - likely same author"
        elif similarity >= 0.8:
            return "Strong similarity - possibly same author or similar style"
        elif similarity >= 0.7:
            return "Moderate similarity - some stylistic overlap"
        elif similarity >= 0.6:
            return "Weak similarity - limited stylistic overlap"
        else:
            return "Low similarity - distinct writing styles"
    
    def _interpret_distance(self, distance: float) -> str:
        """Interpret Euclidean distance for authorship analysis."""
        if distance <= 0.5:
            return "Very close styles - likely same author"
        elif distance <= 1.0:
            return "Similar styles - possibly same author"
        elif distance <= 2.0:
            return "Moderate stylistic difference"
        elif distance <= 4.0:
            return "Significant stylistic difference"
        else:
            return "Very different writing styles"
    
    def _compare_feature_categories(self, vector_a: StyleVector, 
                                  vector_b: StyleVector) -> Dict[str, Any]:
        """Compare features by category for detailed analysis."""
        categories = {
            'lexical': ['lexical_', 'type_token', 'hapax', 'yules', 'simpsons'],
            'syntactic': ['syntax_', 'pos_', 'dependency_'],
            'semantic': ['semantic_', 'topic_', 'entity_'],
            'sentiment': ['sentiment_', 'emotion_', 'person_perspective'],
            'rhythm': ['rhythm_', 'coefficient_of_variation', 'autocorrelation', 'burstiness'],
            'punctuation': ['punctuation_', 'dialogue_', 'question_', 'exclamation_'],
            'patterns': ['pattern_', 'discourse_', 'temporal_', 'enumeration_']
        }
        
        category_similarities = {}
        
        for category, patterns in categories.items():
            category_features_a = {k: v for k, v in vector_a.features.items() 
                                 if any(pattern in k for pattern in patterns)}
            category_features_b = {k: v for k, v in vector_b.features.items() 
                                 if any(pattern in k for pattern in patterns)}
            
            if category_features_a and category_features_b:
                # Create temporary vectors for this category
                temp_a = StyleVector(category_features_a)
                temp_b = StyleVector(category_features_b)
                similarity = temp_a.similarity(temp_b)
                category_similarities[category] = {
                    'similarity': similarity,
                    'feature_count': len(set(category_features_a.keys()) & set(category_features_b.keys())),
                    'interpretation': self._interpret_similarity(similarity)
                }
        
        return category_similarities
    
    def get_authorship_signature(self) -> Dict[str, Any]:
        """Generate a concise authorship signature with key distinguishing features."""
        vector = self.create_style_vector()
        top_features = vector.get_top_features(15)
        
        # Categorize top features
        signature_categories = defaultdict(list)
        category_mapping = {
            'lexical': ['lexical_', 'type_token', 'hapax', 'yules', 'simpsons'],
            'rhythm': ['rhythm_', 'coefficient', 'autocorrelation', 'burstiness'],
            'sentiment': ['sentiment_', 'emotion_', 'person_perspective', 'confidence'],
            'syntax': ['syntax_', 'pos_', 'dependency_'],
            'style': ['pattern_', 'discourse_', 'temporal_', 'punctuation_']
        }
        
        for feature_name, value in top_features:
            categorized = False
            for category, patterns in category_mapping.items():
                if any(pattern in feature_name for pattern in patterns):
                    signature_categories[category].append((feature_name, value))
                    categorized = True
                    break
            if not categorized:
                signature_categories['other'].append((feature_name, value))
        
        # Generate human-readable signature
        signature_summary = []
        for category, features in signature_categories.items():
            if features:
                top_feature = features[0]  # Most prominent in category
                signature_summary.append({
                    'category': category,
                    'top_feature': top_feature[0],
                    'value': top_feature[1],
                    'feature_count': len(features)
                })
        
        return {
            'signature_features': dict(top_features),
            'category_breakdown': dict(signature_categories),
            'signature_summary': signature_summary,
            'total_features': len(vector.features),
            'signature_strength': statistics.mean([abs(v) for _, v in top_features]) if top_features else 0.0
        }
