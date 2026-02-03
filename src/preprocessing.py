"""
Text Preprocessing Module
Provides utilities for cleaning and preprocessing text data for NLP tasks.
"""

import re
import string
import nltk
from typing import List, Optional

# Download required NLTK data (if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


class TextPreprocessor:
    """
    A class for preprocessing text data with various NLP techniques.
    """

    def __init__(self, language: str = 'english'):
        """
        Initialize the TextPreprocessor.

        Args:
            language (str): Language for stopwords. Default is 'english'.
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str, remove_punctuation: bool = True,
                   lowercase: bool = True) -> str:
        """
        Clean text by removing unwanted characters and normalizing.

        Args:
            text (str): Input text to clean.
            remove_punctuation (bool): Whether to remove punctuation.
            lowercase (bool): Whether to convert to lowercase.

        Returns:
            str: Cleaned text.
        """
        if lowercase:
            text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text (str): Input text to tokenize.

        Returns:
            List[str]: List of tokens.
        """
        return word_tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from a list of tokens.

        Args:
            tokens (List[str]): List of word tokens.

        Returns:
            List[str]: List of tokens without stopwords.
        """
        return [word for word in tokens if word.lower() not in self.stop_words]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming to a list of tokens.

        Args:
            tokens (List[str]): List of word tokens.

        Returns:
            List[str]: List of stemmed tokens.
        """
        return [self.stemmer.stem(word) for word in tokens]

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to a list of tokens.

        Args:
            tokens (List[str]): List of word tokens.

        Returns:
            List[str]: List of lemmatized tokens.
        """
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def preprocess(self, text: str, remove_stopwords: bool = True,
                   use_lemmatization: bool = True) -> List[str]:
        """
        Complete preprocessing pipeline.

        Args:
            text (str): Input text to preprocess.
            remove_stopwords (bool): Whether to remove stopwords.
            use_lemmatization (bool): Whether to use lemmatization (else stemming).

        Returns:
            List[str]: List of preprocessed tokens.
        """
        # Clean text
        cleaned = self.clean_text(text)

        # Tokenize
        tokens = self.tokenize(cleaned)

        # Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)

        # Lemmatize or stem
        if use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        else:
            tokens = self.stem_tokens(tokens)

        return tokens


def preprocess_text(text: str, **kwargs) -> List[str]:
    """
    Convenience function for text preprocessing.

    Args:
        text (str): Input text to preprocess.
        **kwargs: Additional arguments passed to TextPreprocessor.preprocess()

    Returns:
        List[str]: List of preprocessed tokens.
    """
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess(text, **kwargs)


if __name__ == "__main__":
    # Example usage
    sample_text = "Hello! This is a sample text for NLP preprocessing. Visit https://example.com for more info."
    
    preprocessor = TextPreprocessor()
    
    print("Original text:")
    print(sample_text)
    print("\nCleaned text:")
    print(preprocessor.clean_text(sample_text))
    print("\nTokens:")
    tokens = preprocessor.tokenize(sample_text.lower())
    print(tokens)
    print("\nWithout stopwords:")
    print(preprocessor.remove_stopwords(tokens))
    print("\nPreprocessed:")
    print(preprocessor.preprocess(sample_text))
