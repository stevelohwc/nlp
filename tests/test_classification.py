"""
Unit tests for the text classification module.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.classification import TextClassifier


class TestTextClassifier:
    """Tests for TextClassifier class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.texts = [
            "This is positive text, very good!",
            "This is negative text, very bad!",
            "Great product, highly recommend!",
            "Terrible product, not recommended!",
            "Excellent quality and service!",
            "Poor quality and service!",
        ]
        self.labels = ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
    
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        classifier = TextClassifier(classifier_type='naive_bayes')
        assert classifier.classifier_type == 'naive_bayes'
        assert not classifier.is_trained
    
    def test_train_classifier(self):
        """Test training the classifier."""
        classifier = TextClassifier(classifier_type='naive_bayes')
        results = classifier.train(self.texts, self.labels, test_size=0.33)
        
        assert classifier.is_trained
        assert 'accuracy' in results
        assert 'classification_report' in results
        assert results['accuracy'] >= 0  # Accuracy should be non-negative
    
    def test_predict_without_training(self):
        """Test that prediction fails without training."""
        classifier = TextClassifier()
        with pytest.raises(ValueError):
            classifier.predict(["Test text"])
    
    def test_predict_after_training(self):
        """Test prediction after training."""
        classifier = TextClassifier(classifier_type='naive_bayes')
        classifier.train(self.texts, self.labels, test_size=0.33)
        
        predictions = classifier.predict(["This is great!"])
        assert len(predictions) == 1
        assert predictions[0] in ['positive', 'negative']
    
    def test_predict_proba(self):
        """Test probability prediction."""
        classifier = TextClassifier(classifier_type='naive_bayes')
        classifier.train(self.texts, self.labels, test_size=0.33)
        
        probabilities = classifier.predict_proba(["This is great!"])
        assert probabilities.shape[0] == 1
        assert probabilities.shape[1] == 2  # Two classes
        # Probabilities should sum to 1
        assert abs(probabilities[0].sum() - 1.0) < 0.01
    
    def test_different_vectorizers(self):
        """Test different vectorizer types."""
        for vectorizer_type in ['tfidf', 'count']:
            classifier = TextClassifier(
                classifier_type='naive_bayes',
                vectorizer_type=vectorizer_type
            )
            results = classifier.train(self.texts, self.labels, test_size=0.33)
            assert classifier.is_trained


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
