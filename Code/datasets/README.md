# IMDB Movie Reviews Dataset

## Overview

This directory contains the IMDB sentiment analysis dataset used for the text classification component of the NLP assignment (CT052-3-M-NLP).

**Dataset**: IMDB Movie Reviews
**Task**: Binary sentiment classification (positive/negative)
**Total Samples**: 50,000 movie reviews
**Classes**: 2 (positive, negative)
**Source**: http://ai.stanford.edu/~amaas/data/sentiment/

## Directory Structure

```
datasets/
├── raw/                      # Original dataset files
│   ├── imdb_train.csv        # 25,000 training reviews
│   ├── imdb_test.csv         # 25,000 test reviews
│   └── datasets/             # Hugging Face cache (auto-generated)
├── processed/                # Preprocessed data
│   ├── train_preprocessed.csv    # Cleaned training data
│   ├── test_preprocessed.csv     # Cleaned test data
│   ├── data_splits.pkl           # Train/val/test text splits
│   ├── tfidf_features.pkl        # TF-IDF vectorized features
│   ├── bow_features.pkl          # Bag-of-Words features
│   └── preprocessor.pkl          # Saved preprocessor object
├── README.md                 # This file
└── DATASET_INFO.md           # Detailed dataset documentation
```

## Dataset Statistics

- **Training set**: 25,000 reviews (12,500 positive, 12,500 negative)
- **Test set**: 25,000 reviews (12,500 positive, 12,500 negative)
- **Class balance**: Perfect 50/50 split
- **Average review length**: ~230 words
- **Vocabulary size**: ~88,000 unique words (before preprocessing)
- **Data quality**: No missing values, minimal duplicates

## Citation

If you use this dataset in your work, please cite:

```bibtex
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
```

## Usage

### Loading Raw Data

```python
from text_classification.data_loader import IMDBDataLoader

# Initialize loader
loader = IMDBDataLoader()

# Load dataset (downloads if not cached)
train_df, test_df = loader.load_dataset()

# Get train/validation split
train_df, val_df = loader.get_train_validation_split(validation_size=0.2)

# Get dataset information
info = loader.get_dataset_info()
```

### Loading Preprocessed Data

```python
import pickle
from pathlib import Path

# Load preprocessed features
with open('processed/tfidf_features.pkl', 'rb') as f:
    tfidf_data = pickle.load(f)

X_train = tfidf_data['X_train']
y_train = tfidf_data['y_train']
X_val = tfidf_data['X_val']
y_val = tfidf_data['y_val']
X_test = tfidf_data['X_test']
y_test = tfidf_data['y_test']
vectorizer = tfidf_data['vectorizer']
```

## Data Preprocessing

The preprocessed data has undergone the following transformations:

1. **HTML tag removal** - Removed `<br />` and other HTML elements
2. **URL removal** - Removed web links
3. **Contraction expansion** - Expanded contractions (e.g., "isn't" → "is not")
4. **Lowercase conversion** - Converted all text to lowercase
5. **Special character removal** - Kept only letters and spaces
6. **Whitespace normalization** - Removed extra spaces

### Feature Extraction Methods

Two feature extraction approaches are provided:

1. **TF-IDF** (Term Frequency-Inverse Document Frequency)
   - Max features: 10,000
   - Min document frequency: 5
   - Max document frequency: 0.7
   - N-grams: (1, 2) - unigrams and bigrams

2. **Bag-of-Words** (Count Vectorizer)
   - Same parameters as TF-IDF
   - Raw word counts instead of TF-IDF weights

## Data Splits

The original training set (25,000) was split into:
- **Training**: 20,000 samples (64% of total)
- **Validation**: 5,000 samples (16% of total)
- **Test**: 25,000 samples (20% of total) - kept separate

All splits maintain stratified class distribution (50/50).

## Benchmark Performance

Published results on IMDB dataset:

| Model | Accuracy | Year | Reference |
|-------|----------|------|-----------|
| Logistic Regression (Baseline) | ~88% | 2011 | Maas et al. |
| Random Forest | ~85% | - | - |
| SVM with TF-IDF | ~89% | - | - |
| LSTM | ~87% | - | - |
| CNN | ~90% | - | - |
| BERT (fine-tuned) | ~94-95% | 2019+ | Various |

**Target**: Achieve 88-92% accuracy with classical ML models, 93-95% with deep learning.

## Data Quality Notes

- **Missing values**: None detected
- **Duplicates**: Minimal (< 0.1%)
- **Imbalanced classes**: No (perfect 50/50 split)
- **HTML artifacts**: Present in raw data, removed in preprocessing
- **Review length**: Varies from ~10 to ~2,500 words
- **Outliers**: Some very short reviews exist but are kept for completeness

## License

The IMDB dataset is freely available for academic research purposes.

## Contact

For questions about the dataset or preprocessing:
- See `Code/text_classification/data_loader.py` for loading utilities
- See `Code/text_classification/preprocessing.ipynb` for preprocessing pipeline
- Refer to `DATASET_INFO.md` for detailed documentation
