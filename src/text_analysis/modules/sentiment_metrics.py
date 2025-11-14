"""Sentiment and emotion analysis module with pragmatic metrics."""

from __future__ import annotations

import re
from typing import Any
from collections import Counter

from .base import AnalysisModule
from ..utils.stats import summarize_counts


class SentimentMetrics(AnalysisModule):
    """Analyzes sentiment, emotion, and pragmatic patterns in text."""

    @property  
    def name(self) -> str:
        return "sentiment_metrics"

    def compute(self, payload, context: dict[str, Any]) -> dict[str, Any]:
        """Calculate sentiment and emotion statistics."""
        return self.analyze(context)

    def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        """Calculate sentiment and emotion patterns from spaCy doc and sentences."""
        sentences = context["sentences"]
        doc = context.get("spacy_doc")
        
        if not sentences:
            return {}

        # Sentiment patterns (lexicon-based approach for now)
        sentiment_scores = []
        emotion_patterns = self._analyze_emotions(sentences)
        person_perspective = self._analyze_person_perspective(doc, sentences) if doc else {}
        confidence_markers = self._analyze_confidence(sentences)
        hate_speech_indicators = self._analyze_hate_speech(sentences)
        flattery_patterns = self._analyze_flattery(sentences)
        
        # Basic sentiment lexicon scoring
        for sentence in sentences:
            score = self._calculate_sentence_sentiment(sentence)
            sentiment_scores.append(score)
        
        sentiment_stats = summarize_counts(sentiment_scores) if sentiment_scores else {}
        
        return {
            "sentiment_analysis": sentiment_stats,
            "emotion_patterns": emotion_patterns,
            "person_perspective": person_perspective,
            "confidence_markers": confidence_markers,
            "hate_speech_indicators": hate_speech_indicators,
            "flattery_patterns": flattery_patterns,
            "sentiment_distribution": self._categorize_sentiments(sentiment_scores)
        }

    def _calculate_sentence_sentiment(self, sentence: str) -> float:
        """Basic lexicon-based sentiment scoring."""
        # Simple positive/negative word lists for demonstration
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like',
            'beautiful', 'perfect', 'best', 'awesome', 'brilliant', 'outstanding', 'superb',
            'magnificent', 'marvelous', 'terrific', 'fabulous', 'incredible', 'phenomenal'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'worst', 'disgusting',
            'appalling', 'dreadful', 'abysmal', 'atrocious', 'deplorable', 'revolting',
            'repulsive', 'vile', 'despicable', 'loathsome', 'detestable', 'abominable'
        }
        
        # Intensifiers and diminishers
        intensifiers = {'very', 'extremely', 'incredibly', 'absolutely', 'totally', 'completely'}
        diminishers = {'somewhat', 'slightly', 'barely', 'hardly', 'scarcely', 'rather'}
        
        words = sentence.lower().split()
        score = 0.0
        intensity = 1.0
        
        for i, word in enumerate(words):
            # Check for intensifiers/diminishers
            if word in intensifiers:
                intensity = 1.5
                continue
            elif word in diminishers:
                intensity = 0.5
                continue
            
            # Check for negation
            negation = False
            if i > 0 and words[i-1] in {'not', "n't", 'no', 'never', 'none', 'neither'}:
                negation = True
            
            # Score sentiment words
            if word in positive_words:
                score += (1.0 * intensity) * (-1 if negation else 1)
            elif word in negative_words:
                score += (-1.0 * intensity) * (-1 if negation else 1)
            
            # Reset intensity after applying
            intensity = 1.0
        
        # Normalize by sentence length
        return score / len(words) if words else 0.0

    def _analyze_emotions(self, sentences: list[str]) -> dict[str, Any]:
        """Analyze emotional patterns using keyword detection."""
        emotion_words = {
            'anger': ['angry', 'furious', 'rage', 'mad', 'irritated', 'annoyed', 'frustrated', 'outraged'],
            'fear': ['afraid', 'scared', 'frightened', 'terrified', 'anxious', 'worried', 'nervous', 'panic'],
            'joy': ['happy', 'delighted', 'joyful', 'cheerful', 'elated', 'euphoric', 'ecstatic', 'gleeful'],
            'sadness': ['sad', 'depressed', 'melancholy', 'grief', 'sorrow', 'misery', 'despair', 'heartbroken'],
            'surprise': ['surprised', 'shocked', 'astonished', 'amazed', 'startled', 'stunned', 'bewildered'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'nauseated', 'sickened', 'appalled']
        }
        
        emotion_counts = {emotion: 0 for emotion in emotion_words.keys()}
        total_words = 0
        
        for sentence in sentences:
            words = sentence.lower().split()
            total_words += len(words)
            
            for emotion, keywords in emotion_words.items():
                for word in words:
                    if word in keywords:
                        emotion_counts[emotion] += 1
        
        # Calculate ratios
        emotion_stats = {}
        for emotion, count in emotion_counts.items():
            emotion_stats[f"{emotion}_count"] = count
            emotion_stats[f"{emotion}_ratio"] = count / total_words if total_words > 0 else 0.0
        
        # Calculate emotional diversity
        total_emotional_words = sum(emotion_counts.values())
        if total_emotional_words > 0:
            emotion_entropy = -sum(
                (count/total_emotional_words) * __import__('math').log2(count/total_emotional_words)
                for count in emotion_counts.values() if count > 0
            )
        else:
            emotion_entropy = 0.0
        
        emotion_stats["emotional_diversity"] = emotion_entropy
        emotion_stats["total_emotional_words"] = total_emotional_words
        
        return emotion_stats

    def _analyze_person_perspective(self, doc, sentences: list[str]) -> dict[str, Any]:
        """Analyze first, second, and third person perspective usage."""
        if not doc:
            return {}
        
        person_counts = {
            'first_person_singular': 0,
            'first_person_plural': 0,
            'second_person': 0,
            'third_person': 0
        }
        
        first_singular = {'i', 'me', 'my', 'mine', 'myself'}
        first_plural = {'we', 'us', 'our', 'ours', 'ourselves'}
        second_person = {'you', 'your', 'yours', 'yourself', 'yourselves'}
        
        total_pronouns = 0
        
        for token in doc:
            if token.pos_ == "PRON" and not token.is_space:
                total_pronouns += 1
                text_lower = token.text.lower()
                
                if text_lower in first_singular:
                    person_counts['first_person_singular'] += 1
                elif text_lower in first_plural:
                    person_counts['first_person_plural'] += 1
                elif text_lower in second_person:
                    person_counts['second_person'] += 1
                elif token.pos_ == "PRON":  # Other pronouns are generally third person
                    person_counts['third_person'] += 1
        
        # Calculate ratios
        perspective_stats = {}
        for perspective, count in person_counts.items():
            perspective_stats[f"{perspective}_count"] = count
            perspective_stats[f"{perspective}_ratio"] = count / total_pronouns if total_pronouns > 0 else 0.0
        
        perspective_stats["total_pronouns"] = total_pronouns
        
        # Perspective dominance
        if total_pronouns > 0:
            dominant_perspective = max(person_counts.keys(), key=lambda x: person_counts[x])
            perspective_stats["dominant_perspective"] = dominant_perspective
            perspective_stats["perspective_balance"] = 1.0 - (max(person_counts.values()) / total_pronouns)
        
        return perspective_stats

    def _analyze_confidence(self, sentences: list[str]) -> dict[str, Any]:
        """Analyze confidence and certainty markers."""
        # Modal verbs indicating uncertainty
        uncertainty_markers = {
            'might', 'may', 'could', 'would', 'should', 'possibly', 'probably',
            'perhaps', 'maybe', 'supposedly', 'allegedly', 'apparently'
        }
        
        # Strong certainty markers
        certainty_markers = {
            'definitely', 'certainly', 'absolutely', 'undoubtedly', 'clearly',
            'obviously', 'surely', 'indeed', 'must', 'will', 'shall'
        }
        
        # Hedging patterns
        hedging_patterns = [
            r'\bit\s+seems\s+(?:that|like)\b',
            r'\bit\s+appears\s+(?:that|like)\b',
            r'\bi\s+think\s+(?:that)?\b',
            r'\bi\s+believe\s+(?:that)?\b',
            r'\bin\s+my\s+opinion\b',
            r'\bkind\s+of\b',
            r'\bsort\s+of\b'
        ]
        
        uncertainty_count = 0
        certainty_count = 0
        hedging_count = 0
        total_sentences = len(sentences)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            words = sentence_lower.split()
            
            # Count uncertainty and certainty markers
            for word in words:
                if word in uncertainty_markers:
                    uncertainty_count += 1
                elif word in certainty_markers:
                    certainty_count += 1
            
            # Count hedging patterns
            for pattern in hedging_patterns:
                if re.search(pattern, sentence_lower):
                    hedging_count += 1
        
        return {
            "uncertainty_markers": uncertainty_count,
            "certainty_markers": certainty_count,
            "hedging_patterns": hedging_count,
            "uncertainty_ratio": uncertainty_count / total_sentences if total_sentences > 0 else 0.0,
            "certainty_ratio": certainty_count / total_sentences if total_sentences > 0 else 0.0,
            "hedging_ratio": hedging_count / total_sentences if total_sentences > 0 else 0.0,
            "confidence_balance": (certainty_count - uncertainty_count) / max(certainty_count + uncertainty_count, 1)
        }

    def _analyze_hate_speech(self, sentences: list[str]) -> dict[str, Any]:
        """Analyze potential hate speech and toxic language indicators."""
        # This is a basic implementation - in production you'd use specialized models
        
        # Mild profanity and negative terms (abbreviated list for demonstration)
        profanity_words = {
            'damn', 'hell', 'crap', 'stupid', 'idiot', 'moron', 'dumb', 'fool',
            'jerk', 'loser', 'pathetic', 'worthless', 'useless', 'garbage'
        }
        
        # Aggressive language patterns
        aggressive_patterns = [
            r'\byou\s+are\s+(?:so\s+)?(?:stupid|dumb|idiotic)\b',
            r'\bshut\s+up\b',
            r'\bget\s+lost\b',
            r'\bgo\s+to\s+hell\b',
            r'\bi\s+hate\s+you\b'
        ]
        
        # Dehumanizing language
        dehumanizing_words = {
            'animal', 'beast', 'creature', 'thing', 'it'  # when referring to people
        }
        
        profanity_count = 0
        aggressive_count = 0
        dehumanizing_count = 0
        total_words = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            words = sentence_lower.split()
            total_words += len(words)
            
            # Count profanity
            for word in words:
                if word in profanity_words:
                    profanity_count += 1
                elif word in dehumanizing_words:
                    dehumanizing_count += 1
            
            # Count aggressive patterns
            for pattern in aggressive_patterns:
                if re.search(pattern, sentence_lower):
                    aggressive_count += 1
        
        # Calculate toxicity score (simple heuristic)
        toxicity_score = (profanity_count * 1.0 + aggressive_count * 2.0 + dehumanizing_count * 1.5) / max(total_words, 1)
        
        return {
            "profanity_count": profanity_count,
            "aggressive_language_count": aggressive_count,
            "dehumanizing_language_count": dehumanizing_count,
            "toxicity_score": toxicity_score,
            "profanity_ratio": profanity_count / total_words if total_words > 0 else 0.0,
            "aggressive_ratio": aggressive_count / len(sentences) if sentences else 0.0
        }

    def _categorize_sentiments(self, sentiment_scores: list[float]) -> dict[str, Any]:
        """Categorize sentences by sentiment polarity."""
        if not sentiment_scores:
            return {}
        
        positive = sum(1 for score in sentiment_scores if score > 0.1)
        negative = sum(1 for score in sentiment_scores if score < -0.1)
        neutral = len(sentiment_scores) - positive - negative
        
        total = len(sentiment_scores)
        
        return {
            "positive_sentences": positive,
            "negative_sentences": negative,
            "neutral_sentences": neutral,
            "positive_ratio": positive / total if total > 0 else 0.0,
            "negative_ratio": negative / total if total > 0 else 0.0,
            "neutral_ratio": neutral / total if total > 0 else 0.0,
            "sentiment_polarity": (positive - negative) / total if total > 0 else 0.0
        }

    def _analyze_flattery(self, sentences: list[str]) -> dict[str, Any]:
        """Analyze flattery, praise, and sycophantic patterns."""
        # Flattery and praise indicators
        flattery_words = {
            'direct_praise': ['amazing', 'brilliant', 'excellent', 'outstanding', 'exceptional', 'superb',
                             'magnificent', 'incredible', 'wonderful', 'fantastic', 'marvelous', 'spectacular'],
            'excessive_positives': ['absolutely', 'totally', 'completely', 'perfectly', 'extremely',
                                   'incredibly', 'utterly', 'tremendously', 'extraordinarily'],
            'superlatives': ['best', 'greatest', 'finest', 'most', 'ultimate', 'supreme', 'perfect',
                           'unmatched', 'unparalleled', 'incomparable', 'flawless'],
            'sycophantic': ['honored', 'privilege', 'grateful', 'thankful', 'blessed', 'humbled',
                          'impressed', 'inspired', 'admire', 'respect']
        }
        
        # Flattery patterns
        flattery_patterns = [
            r'(?:you\s+are|you\'re)\s+(?:so|very|really|absolutely|incredibly|amazingly)\s+\w+',
            r'i\s+(?:love|adore|admire|respect)\s+(?:how|the\s+way|that)',
            r'(?:thank\s+you|thanks)\s+(?:so\s+much|very\s+much|a\s+lot)',
            r'you\s+(?:always|never\s+fail\s+to|consistently)\s+\w+',
            r'(?:what\s+a|such\s+a)\s+\w+\s+\w+',  # "what a great idea"
            r'i\s+(?:couldn\'t|can\'t)\s+agree\s+more',
            r'(?:absolutely|totally|completely)\s+(?:right|correct|true|agree)',
        ]
        
        # AI assistant sycophancy patterns
        ai_sycophancy = [
            r'(?:of\s+course|certainly|absolutely),?\s+i\'?d?\s+be\s+(?:happy|glad|delighted)\s+to',
            r'i\'?m?\s+(?:here\s+to\s+help|at\s+your\s+service)',
            r'(?:great|excellent|wonderful)\s+(?:question|point|idea|suggestion)',
            r'that\'?s\s+(?:a\s+)?(great|excellent|wonderful|fantastic)\s+\w+',
            r'i\s+(?:completely|totally|absolutely)\s+understand',
            r'you\'?re?\s+(?:absolutely|totally|completely)\s+(?:right|correct)',
        ]
        
        flattery_counts = {category: 0 for category in flattery_words.keys()}
        pattern_matches = 0
        ai_sycophancy_count = 0
        total_words = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            words = sentence_lower.split()
            total_words += len(words)
            
            # Count flattery words by category
            for category, word_list in flattery_words.items():
                for word in words:
                    if word in word_list:
                        flattery_counts[category] += 1
            
            # Count flattery patterns
            for pattern in flattery_patterns:
                if re.search(pattern, sentence_lower):
                    pattern_matches += 1
            
            # Count AI sycophancy patterns
            for pattern in ai_sycophancy:
                if re.search(pattern, sentence_lower):
                    ai_sycophancy_count += 1
        
        # Calculate total flattery score
        total_flattery_words = sum(flattery_counts.values())
        flattery_density = (total_flattery_words + pattern_matches) / total_words if total_words > 0 else 0.0
        
        # Calculate sycophancy score (specific for AI-generated text)
        sycophancy_score = ai_sycophancy_count / len(sentences) if sentences else 0.0
        
        return {
            'flattery_word_counts': flattery_counts,
            'total_flattery_words': total_flattery_words,
            'flattery_patterns': pattern_matches,
            'ai_sycophancy_patterns': ai_sycophancy_count,
            'flattery_density': round(flattery_density, 4),
            'sycophancy_score': round(sycophancy_score, 4),
            'praise_intensity': round((flattery_counts.get('excessive_positives', 0) + 
                                     flattery_counts.get('superlatives', 0)) / total_words if total_words > 0 else 0.0, 4)
        }