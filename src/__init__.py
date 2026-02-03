"""
NLP Package
A Natural Language Processing toolkit for text analysis and classification.
"""

__version__ = "1.0.0"
__author__ = "NLP Group"

from .preprocessing import TextPreprocessor, preprocess_text
from .sentiment import SentimentAnalyzer, analyze_sentiment
from .classification import TextClassifier
from .utils import (
    load_text_file,
    save_text_file,
    load_json,
    save_json,
    calculate_statistics,
    print_statistics
)

__all__ = [
    'TextPreprocessor',
    'preprocess_text',
    'SentimentAnalyzer',
    'analyze_sentiment',
    'TextClassifier',
    'load_text_file',
    'save_text_file',
    'load_json',
    'save_json',
    'calculate_statistics',
    'print_statistics'
]
