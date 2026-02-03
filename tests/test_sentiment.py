"""
Unit tests for the sentiment analysis module.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.sentiment import SentimentAnalyzer


class TestSentimentAnalyzer:
    """Tests for SentimentAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    def test_analyze_positive_sentiment(self):
        """Test positive sentiment detection."""
        text = "I love this! It's amazing and wonderful!"
        result = self.analyzer.analyze(text)
        assert result['label'] == 'positive'
        assert result['compound'] > 0
    
    def test_analyze_negative_sentiment(self):
        """Test negative sentiment detection."""
        text = "This is terrible and awful. I hate it!"
        result = self.analyzer.analyze(text)
        assert result['label'] == 'negative'
        assert result['compound'] < 0
    
    def test_analyze_neutral_sentiment(self):
        """Test neutral sentiment detection."""
        text = "The meeting is at 3 PM."
        result = self.analyzer.analyze(text)
        # Check that compound score is close to neutral
        assert abs(result['compound']) < 0.3
    
    def test_analyze_returns_all_scores(self):
        """Test that analyze returns all required scores."""
        text = "Test text"
        result = self.analyzer.analyze(text)
        assert 'positive' in result
        assert 'negative' in result
        assert 'neutral' in result
        assert 'compound' in result
        assert 'label' in result
        assert 'score' in result
    
    def test_analyze_batch(self):
        """Test batch sentiment analysis."""
        texts = [
            "I love this!",
            "This is terrible!",
            "This is okay."
        ]
        results = self.analyzer.analyze_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
    
    def test_get_sentiment_distribution(self):
        """Test sentiment distribution calculation."""
        texts = [
            "I love this!",
            "This is terrible!",
            "This is okay.",
            "Amazing!",
            "Awful!"
        ]
        distribution = self.analyzer.get_sentiment_distribution(texts)
        assert 'positive' in distribution
        assert 'negative' in distribution
        assert 'neutral' in distribution
        assert sum(distribution.values()) == len(texts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
