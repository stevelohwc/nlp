"""
Sentiment Analysis Module
Provides sentiment analysis capabilities for text data.
"""

import nltk
from typing import Dict, List, Any
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK data
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')


class SentimentAnalyzer:
    """
    A class for performing sentiment analysis on text data.
    """

    def __init__(self):
        """
        Initialize the SentimentAnalyzer with VADER sentiment analyzer.
        """
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of the given text.

        Args:
            text (str): Input text to analyze.

        Returns:
            Dict: Dictionary containing sentiment scores and label.
                - 'positive': Positive sentiment score
                - 'negative': Negative sentiment score
                - 'neutral': Neutral sentiment score
                - 'compound': Compound score (-1 to 1)
                - 'label': Overall sentiment label (positive/negative/neutral)
        """
        scores = self.analyzer.polarity_scores(text)
        
        # Determine overall sentiment label
        compound = scores['compound']
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': compound,
            'label': label,
            'score': abs(compound)
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for multiple texts.

        Args:
            texts (List[str]): List of texts to analyze.

        Returns:
            List[Dict]: List of sentiment analysis results.
        """
        return [self.analyze(text) for text in texts]

    def get_sentiment_distribution(self, texts: List[str]) -> Dict[str, int]:
        """
        Get distribution of sentiments across multiple texts.

        Args:
            texts (List[str]): List of texts to analyze.

        Returns:
            Dict[str, int]: Count of each sentiment label.
        """
        results = self.analyze_batch(texts)
        distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for result in results:
            distribution[result['label']] += 1
        
        return distribution


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Convenience function for sentiment analysis.

    Args:
        text (str): Input text to analyze.

    Returns:
        Dict: Sentiment analysis results.
    """
    analyzer = SentimentAnalyzer()
    return analyzer.analyze(text)


if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer()
    
    sample_texts = [
        "I love this project! It's amazing and wonderful.",
        "This is terrible and I hate it.",
        "This is okay, nothing special.",
        "The weather is nice today.",
    ]
    
    print("Sentiment Analysis Examples:\n")
    for text in sample_texts:
        result = analyzer.analyze(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['label'].upper()}")
        print(f"Compound Score: {result['compound']:.3f}")
        print(f"Positive: {result['positive']:.3f}, "
              f"Negative: {result['negative']:.3f}, "
              f"Neutral: {result['neutral']:.3f}")
        print("-" * 60)
    
    print("\nSentiment Distribution:")
    distribution = analyzer.get_sentiment_distribution(sample_texts)
    for sentiment, count in distribution.items():
        print(f"{sentiment.capitalize()}: {count}")
