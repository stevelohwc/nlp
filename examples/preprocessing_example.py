"""
Example: Text Preprocessing
Demonstrates the text preprocessing capabilities.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import TextPreprocessor


def main():
    """Main function demonstrating text preprocessing."""
    
    print("=" * 70)
    print("Text Preprocessing Example")
    print("=" * 70)
    
    # Sample texts
    texts = [
        "Hello! Check out https://example.com for more info. Contact us at info@example.com",
        "The Natural Language Processing course is AMAZING!!! #NLP #AI",
        "I can't believe how easy it is to preprocess text data!!!",
    ]
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    for i, text in enumerate(texts, 1):
        print(f"\n{'='*70}")
        print(f"Example {i}:")
        print(f"{'='*70}")
        print(f"Original text:\n{text}")
        
        # Clean text
        cleaned = preprocessor.clean_text(text)
        print(f"\nCleaned text:\n{cleaned}")
        
        # Tokenize
        tokens = preprocessor.tokenize(cleaned)
        print(f"\nTokens:\n{tokens}")
        
        # Remove stopwords
        no_stopwords = preprocessor.remove_stopwords(tokens)
        print(f"\nWithout stopwords:\n{no_stopwords}")
        
        # Lemmatize
        lemmatized = preprocessor.lemmatize_tokens(no_stopwords)
        print(f"\nLemmatized:\n{lemmatized}")
        
        # Complete preprocessing pipeline
        preprocessed = preprocessor.preprocess(text)
        print(f"\nFully preprocessed:\n{preprocessed}")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
