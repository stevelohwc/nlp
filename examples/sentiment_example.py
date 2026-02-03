"""
Example: Sentiment Analysis
Demonstrates the sentiment analysis capabilities.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.sentiment import SentimentAnalyzer


def main():
    """Main function demonstrating sentiment analysis."""
    
    print("=" * 70)
    print("Sentiment Analysis Example")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Sample texts with different sentiments
    texts = [
        "I absolutely love this product! It's fantastic and exceeded all my expectations!",
        "This is the worst experience I've ever had. Completely disappointed.",
        "The service was okay. Nothing special, but not terrible either.",
        "Amazing quality! Highly recommend to everyone. Five stars!",
        "Terrible quality, waste of money. Would not recommend.",
        "It's fine, does what it's supposed to do.",
    ]
    
    print("\nAnalyzing individual texts:\n")
    
    results = []
    for i, text in enumerate(texts, 1):
        result = analyzer.analyze(text)
        results.append(result)
        
        print(f"Text {i}: {text}")
        print(f"Sentiment: {result['label'].upper()}")
        print(f"Compound Score: {result['compound']:.3f}")
        print(f"Positive: {result['positive']:.3f} | "
              f"Negative: {result['negative']:.3f} | "
              f"Neutral: {result['neutral']:.3f}")
        print("-" * 70)
    
    # Get sentiment distribution
    print("\nSentiment Distribution:")
    print("=" * 70)
    distribution = analyzer.get_sentiment_distribution(texts)
    
    total = sum(distribution.values())
    for sentiment, count in distribution.items():
        percentage = (count / total) * 100
        print(f"{sentiment.capitalize()}: {count} ({percentage:.1f}%)")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
