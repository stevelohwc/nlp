"""
Unit tests for the preprocessing module.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import TextPreprocessor


class TestTextPreprocessor:
    """Tests for TextPreprocessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
    
    def test_clean_text_lowercase(self):
        """Test text lowercasing."""
        text = "Hello World!"
        result = self.preprocessor.clean_text(text, lowercase=True)
        assert result.islower()
    
    def test_clean_text_remove_punctuation(self):
        """Test punctuation removal."""
        text = "Hello, World!"
        result = self.preprocessor.clean_text(text, remove_punctuation=True)
        assert "," not in result
        assert "!" not in result
    
    def test_clean_text_remove_urls(self):
        """Test URL removal."""
        text = "Check out https://example.com for more info"
        result = self.preprocessor.clean_text(text)
        assert "https://example.com" not in result
    
    def test_clean_text_remove_emails(self):
        """Test email removal."""
        text = "Contact us at info@example.com"
        result = self.preprocessor.clean_text(text)
        assert "info@example.com" not in result
    
    def test_tokenize(self):
        """Test tokenization."""
        text = "This is a test"
        tokens = self.preprocessor.tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) == 4
        assert tokens == ["This", "is", "a", "test"]
    
    def test_remove_stopwords(self):
        """Test stopword removal."""
        tokens = ["this", "is", "a", "test", "sentence"]
        result = self.preprocessor.remove_stopwords(tokens)
        assert "test" in result
        assert "sentence" in result
        assert "is" not in result
        assert "a" not in result
    
    def test_stem_tokens(self):
        """Test stemming."""
        tokens = ["running", "runs", "runner"]
        result = self.preprocessor.stem_tokens(tokens)
        # All should have the same stem
        assert len(set(result)) == 1
    
    def test_lemmatize_tokens(self):
        """Test lemmatization."""
        tokens = ["running", "runs"]
        result = self.preprocessor.lemmatize_tokens(tokens)
        assert "running" in result or "run" in result
    
    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline."""
        text = "This is a TEST sentence with PUNCTUATION!"
        result = self.preprocessor.preprocess(text)
        assert isinstance(result, list)
        assert len(result) > 0
        # Should be lowercase and no punctuation
        for token in result:
            assert token.islower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
