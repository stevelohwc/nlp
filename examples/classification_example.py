"""
Example: Text Classification
Demonstrates the text classification capabilities.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.classification import TextClassifier


def main():
    """Main function demonstrating text classification."""
    
    print("=" * 70)
    print("Text Classification Example")
    print("=" * 70)
    
    # Sample training data - movie reviews
    train_texts = [
        "This movie is fantastic! Great acting and amazing plot.",
        "Terrible film. Waste of time and money.",
        "I loved every minute of this movie. Highly recommended!",
        "Boring and predictable. Not worth watching.",
        "Excellent cinematography and brilliant performances.",
        "Poor script and bad direction. Very disappointed.",
        "One of the best movies I've seen this year!",
        "Awful movie. I walked out halfway through.",
        "Masterpiece! The director is a genius.",
        "Complete disaster. Nothing worked in this film.",
        "Beautiful story with great character development.",
        "Waste of talent. Even good actors couldn't save it.",
        "Incredible movie experience. Must watch!",
        "Disappointing and frustrating to watch.",
        "Amazing from start to finish. Loved it!",
        "Terrible acting and boring storyline.",
    ]
    
    train_labels = [
        'positive', 'negative', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative'
    ]
    
    # Create and train classifier
    print("\nTraining text classifier...")
    print("-" * 70)
    
    classifier = TextClassifier(classifier_type='naive_bayes', vectorizer_type='tfidf')
    results = classifier.train(train_texts, train_labels, test_size=0.25)
    
    print(f"Training completed!")
    print(f"Test set size: {results['test_size']} samples")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"\nClassification Report:")
    print(results['classification_report'])
    
    # Test with new examples
    print("\n" + "=" * 70)
    print("Testing with new examples:")
    print("=" * 70)
    
    test_texts = [
        "This is a wonderful film with great performances!",
        "I didn't enjoy this movie at all. Very boring.",
        "Absolutely brilliant! One of my favorites.",
        "Not recommended. Poor quality throughout.",
    ]
    
    predictions = classifier.predict(test_texts)
    probabilities = classifier.predict_proba(test_texts)
    
    for i, (text, pred, proba) in enumerate(zip(test_texts, predictions, probabilities), 1):
        print(f"\nTest {i}:")
        print(f"Text: {text}")
        print(f"Prediction: {pred.upper()}")
        print(f"Confidence: {max(proba):.3f}")
        print(f"Probabilities: Negative={proba[0]:.3f}, Positive={proba[1]:.3f}")
    
    # Show top features
    print("\n" + "=" * 70)
    print("Top features for each class:")
    print("=" * 70)
    
    for label in ['positive', 'negative']:
        print(f"\nTop words for '{label}' class:")
        top_features = classifier.get_top_features(label=label, n=5)
        for word, score in top_features:
            print(f"  {word}: {score:.3f}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
