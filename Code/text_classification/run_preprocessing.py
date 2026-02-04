"""
Run preprocessing pipeline on IMDB dataset
Generates cleaned text and TF-IDF/BoW features
"""

import pandas as pd
import numpy as np
import pickle
import re
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from text_classification.data_loader import IMDBDataLoader
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

# Download NLTK data
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

# Text cleaning functions
def remove_html_tags(text):
    """Remove HTML tags from text"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_urls(text):
    """Remove URLs from text"""
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def remove_special_chars(text, keep_apostrophe=False):
    """Remove special characters, keep only letters and spaces"""
    if keep_apostrophe:
        pattern = r'[^a-zA-Z\s\']'
    else:
        pattern = r'[^a-zA-Z\s]'
    return re.sub(pattern, '', text)

def remove_extra_spaces(text):
    """Remove extra whitespace"""
    return ' '.join(text.split())

def expand_contractions(text):
    """Expand common contractions"""
    contractions = {
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'ve": " have",
        "'m": " am"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    return text

class TextPreprocessor:
    """Comprehensive text preprocessing pipeline"""

    def __init__(self, lowercase=True, remove_stopwords=False):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()

    def clean_text(self, text):
        """Apply all cleaning steps to text"""
        # Remove HTML tags
        text = remove_html_tags(text)

        # Remove URLs
        text = remove_urls(text)

        # Expand contractions
        text = expand_contractions(text)

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove special characters
        text = remove_special_chars(text, keep_apostrophe=False)

        # Remove extra spaces
        text = remove_extra_spaces(text)

        return text

    def tokenize_and_process(self, text):
        """Tokenize and apply stopword removal"""
        # Tokenize
        tokens = text.split()

        # Remove stopwords
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]

        return ' '.join(tokens)

    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.tokenize_and_process(text)
        return text

    def preprocess_dataframe(self, df, text_column='text'):
        """Preprocess all texts in a DataFrame"""
        df = df.copy()
        print(f"  Processing {len(df)} texts...")
        # Use regular apply with progress tracking
        cleaned_texts = []
        for i, text in enumerate(df[text_column]):
            cleaned_texts.append(self.preprocess(text))
            if (i + 1) % 5000 == 0:
                print(f"    Processed {i + 1}/{len(df)} texts...")
        df['cleaned_text'] = cleaned_texts
        return df

def main():
    print("="*80)
    print("TEXT PREPROCESSING PIPELINE")
    print("="*80)

    # Load dataset
    print("\n[1/6] Loading dataset...")
    loader = IMDBDataLoader()
    train_df, test_df = loader.load_dataset()
    print(f"  Train: {train_df.shape}, Test: {test_df.shape}")

    # Initialize preprocessor
    print("\n[2/6] Initializing preprocessor...")
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_stopwords=False  # Keep stopwords for initial models
    )
    print("  Configuration: lowercase=True, remove_stopwords=False")

    # Preprocess training data
    print("\n[3/6] Preprocessing training data...")
    train_processed = preprocessor.preprocess_dataframe(train_df)

    # Preprocess test data
    print("\n[4/6] Preprocessing test data...")
    test_processed = preprocessor.preprocess_dataframe(test_df)

    # Display sample
    print("\n  Sample preprocessed review:")
    print(f"  Original: {train_df['text'].iloc[0][:150]}...")
    print(f"  Cleaned: {train_processed['cleaned_text'].iloc[0][:150]}...")

    # Create train/validation split
    print("\n[5/6] Creating train/validation split...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_processed['cleaned_text'],
        train_processed['label'],
        test_size=0.2,
        random_state=42,
        stratify=train_processed['label']
    )

    test_texts = test_processed['cleaned_text']
    test_labels = test_processed['label']

    print(f"  Training: {len(train_texts):,} samples")
    print(f"  Validation: {len(val_texts):,} samples")
    print(f"  Test: {len(test_texts):,} samples")

    # Feature extraction - TF-IDF
    print("\n[6/6] Extracting features...")
    print("  TF-IDF vectorization...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=5,
        max_df=0.7,
        ngram_range=(1, 2)
    )

    X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
    X_val_tfidf = tfidf_vectorizer.transform(val_texts)
    X_test_tfidf = tfidf_vectorizer.transform(test_texts)

    print(f"    Train: {X_train_tfidf.shape}")
    print(f"    Validation: {X_val_tfidf.shape}")
    print(f"    Test: {X_test_tfidf.shape}")
    print(f"    Vocabulary size: {len(tfidf_vectorizer.vocabulary_):,}")

    # Feature extraction - Bag of Words
    print("  Bag-of-Words vectorization...")
    count_vectorizer = CountVectorizer(
        max_features=10000,
        min_df=5,
        max_df=0.7,
        ngram_range=(1, 2)
    )

    X_train_bow = count_vectorizer.fit_transform(train_texts)
    X_val_bow = count_vectorizer.transform(val_texts)
    X_test_bow = count_vectorizer.transform(test_texts)

    print(f"    Train: {X_train_bow.shape}")
    print(f"    Validation: {X_val_bow.shape}")
    print(f"    Test: {X_test_bow.shape}")

    # Save all preprocessed data
    print("\n[SAVING] Saving preprocessed data...")
    processed_dir = Path(__file__).parent.parent / 'datasets' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save preprocessed CSVs
    train_processed.to_csv(processed_dir / 'train_preprocessed.csv', index=False)
    test_processed.to_csv(processed_dir / 'test_preprocessed.csv', index=False)
    print(f"  ✓ Saved preprocessed CSVs")

    # Save data splits
    splits_data = {
        'train_texts': train_texts,
        'val_texts': val_texts,
        'test_texts': test_texts,
        'train_labels': train_labels.values,
        'val_labels': val_labels.values,
        'test_labels': test_labels.values
    }
    with open(processed_dir / 'data_splits.pkl', 'wb') as f:
        pickle.dump(splits_data, f)
    print(f"  ✓ Saved data splits")

    # Save TF-IDF features
    tfidf_data = {
        'vectorizer': tfidf_vectorizer,
        'X_train': X_train_tfidf,
        'X_val': X_val_tfidf,
        'X_test': X_test_tfidf,
        'y_train': train_labels.values,
        'y_val': val_labels.values,
        'y_test': test_labels.values
    }
    with open(processed_dir / 'tfidf_features.pkl', 'wb') as f:
        pickle.dump(tfidf_data, f)
    print(f"  ✓ Saved TF-IDF features")

    # Save BoW features
    bow_data = {
        'vectorizer': count_vectorizer,
        'X_train': X_train_bow,
        'X_val': X_val_bow,
        'X_test': X_test_bow,
        'y_train': train_labels.values,
        'y_val': val_labels.values,
        'y_test': test_labels.values
    }
    with open(processed_dir / 'bow_features.pkl', 'wb') as f:
        pickle.dump(bow_data, f)
    print(f"  ✓ Saved BoW features")

    # Save preprocessor
    with open(processed_dir / 'preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"  ✓ Saved preprocessor")

    print(f"\n  All files saved to: {processed_dir}")

    # Summary
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"\n✓ Preprocessed {len(train_processed) + len(test_processed):,} reviews")
    print(f"✓ Created train/val/test splits: {len(train_texts):,} / {len(val_texts):,} / {len(test_texts):,}")
    print(f"✓ Generated TF-IDF features: {X_train_tfidf.shape[1]:,} dimensions")
    print(f"✓ Generated BoW features: {X_train_bow.shape[1]:,} dimensions")
    print(f"✓ Ready for model training!")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
