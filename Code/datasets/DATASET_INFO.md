# IMDB Dataset Selection and Documentation

## Assignment Context

This document provides detailed justification and documentation for the IMDB Movie Reviews dataset selection for the text classification component of the NLP assignment (CT052-3-M-NLP, Part A - 30 marks).

## 1. Dataset Selection Rationale

### Available Options
The assignment allowed five dataset types:
1. Sentiment Analysis
2. Spam Detection
3. Authorship Attribution
4. Fake Reviews Detection
5. Cyberbullying Detection

### Why IMDB Sentiment Analysis?

**Selected Dataset**: IMDB Movie Reviews (50,000 reviews)

**Justification**:

1. **Academic Suitability**
   - Well-established benchmark dataset in NLP research
   - Extensive prior work available for comparison (required by assignment)
   - Widely used in academic literature since 2011
   - Provides credible baseline for model evaluation

2. **Dataset Quality**
   - Large scale: 50,000 labeled reviews (meets assignment requirements)
   - Clean labels: Binary classification (positive/negative) with high agreement
   - Balanced classes: Perfect 50/50 split eliminates class imbalance issues
   - No missing data or quality concerns
   - Pre-split into train/test sets by original authors

3. **Technical Advantages**
   - Appropriate complexity: Not too simple, not too complex
   - Supports multiple model types: Classical ML and deep learning approaches
   - Rich text content: Average 230 words per review allows meaningful analysis
   - Large vocabulary: Demonstrates NLP preprocessing skills
   - Real-world data: Authentic user-generated content with natural language

4. **Assignment Requirements Alignment**
   - Enables multiple model comparison (Naive Bayes, SVM, Random Forest, Neural Networks)
   - Sufficient size for hyperparameter tuning with grid/random search
   - Well-documented benchmarks for comparison (88-95% accuracy range)
   - Straightforward deployment as webpage (movie review sentiment predictor)
   - Clear evaluation metrics (accuracy, precision, recall, F1)

5. **Pedagogical Value**
   - Demonstrates fundamental NLP concepts (tokenization, vectorization)
   - Shows impact of different preprocessing techniques
   - Illustrates feature engineering (TF-IDF, word embeddings)
   - Provides intuitive classification task (sentiment is interpretable)

6. **Deployment Feasibility**
   - Easy to create user-friendly web interface
   - Users can input movie reviews and get instant predictions
   - Results are interpretable and verifiable
   - Practical application with clear use case

### Alternatives Considered

**Spam Detection** (Not Selected)
- Pros: Simple binary classification, high accuracy achievable
- Cons: May be too simple for demonstration purposes, limited vocabulary
- Benchmarks often exceed 97%, leaving little room for model comparison

**Fake Reviews Detection** (Strong Alternative)
- Pros: More unique/challenging, relevant to e-commerce
- Cons: Dataset quality varies, fewer students likely choose this
- Could be considered for future work

**Cyberbullying Detection** (Not Selected)
- Pros: Important social problem, multi-class options
- Cons: Sensitive content, class imbalance issues, ethical considerations
- Requires careful handling of offensive language

**Authorship Attribution** (Not Selected)
- Pros: Interesting linguistic problem, multi-class classification
- Cons: Requires more feature engineering, less intuitive for deployment
- May be too specialized for general NLP assignment

## 2. Dataset Description

### Source
- **Origin**: Stanford AI Lab, Andrew Maas et al. (2011)
- **URL**: http://ai.stanford.edu/~amaas/data/sentiment/
- **Access**: Freely available via Hugging Face Datasets library
- **License**: Academic research purposes

### Data Collection Methodology
- Reviews scraped from IMDB website
- Only reviews with rating ≥7 (positive) or ≤4 (negative) included
- Neutral reviews (ratings 5-6) excluded to ensure clear sentiment labels
- Maximum 30 reviews per movie to ensure diversity
- Balanced sampling to achieve 50/50 positive/negative distribution

### Dataset Size
- **Total samples**: 50,000 movie reviews
- **Training set**: 25,000 reviews (12,500 positive, 12,500 negative)
- **Test set**: 25,000 reviews (12,500 positive, 12,500 negative)
- **Classes**: 2 (binary classification)
  - Label 0: Negative sentiment
  - Label 1: Positive sentiment

### Text Characteristics
- **Average length**: 230 words per review
- **Character range**: 100 to 13,000+ characters
- **Word range**: ~10 to 2,500 words
- **Vocabulary**: ~88,000 unique words (raw), ~10,000-20,000 (after preprocessing)
- **Language**: English
- **Content**: User-written movie reviews with natural language patterns

## 3. Exploratory Data Analysis Findings

### Class Distribution
- **Perfect balance**: Exactly 12,500 samples per class in both train and test
- **No resampling needed**: Balanced classes eliminate need for oversampling/undersampling
- **Stratification**: All splits maintain 50/50 ratio

### Text Length Analysis
- **Mean length**: ~1,300 characters, ~230 words
- **Median length**: ~175 words
- **Standard deviation**: High variance in review lengths
- **Distribution**: Right-skewed (some very long reviews)
- **Sentiment comparison**: Positive and negative reviews have similar length distributions

### Vocabulary Analysis
- **Total unique words**: ~88,000 (before preprocessing)
- **After preprocessing**: ~30,000-40,000 (depends on min_df parameter)
- **Most common words**: Stop words dominate (the, and, a, is, to, etc.)
- **After stopword removal**: Content words emerge (film, movie, good, bad, great, etc.)

### Data Quality
- **Missing values**: 0 (None detected)
- **Duplicates**: < 50 across 50,000 reviews (< 0.1%)
- **HTML artifacts**: Present (`<br />` tags) - removed in preprocessing
- **Special characters**: Minimal issues, handled in preprocessing
- **Encoding**: UTF-8, no encoding issues detected

### Sentiment Indicators
- **Positive keywords**: great, excellent, amazing, wonderful, love, best, perfect
- **Negative keywords**: bad, terrible, worst, awful, waste, boring, poor
- **Sentiment-neutral words**: movie, film, scene, character, story, plot

## 4. Preprocessing Decisions

### Text Cleaning Pipeline
1. **HTML Tag Removal**
   - Justification: Reviews contain `<br />` tags for line breaks
   - Implementation: Regex pattern to remove all HTML tags
   - Impact: Cleaner text, removes non-semantic content

2. **Lowercase Conversion**
   - Justification: "Great" and "great" should be treated identically
   - Implementation: Python `.lower()` method
   - Impact: Reduces vocabulary size, improves feature matching

3. **Special Character Removal**
   - Justification: Punctuation rarely carries sentiment in this dataset
   - Implementation: Keep only letters and spaces
   - Impact: Simplifies vocabulary, focuses on word content

4. **Contraction Expansion**
   - Justification: "isn't" → "is not" preserves negation semantics
   - Implementation: Dictionary-based replacement
   - Impact: Better captures negation patterns

5. **Whitespace Normalization**
   - Justification: Multiple spaces can interfere with tokenization
   - Implementation: Replace multiple spaces with single space
   - Impact: Consistent tokenization

### Optional Preprocessing (Experimentation)

**Stop Word Removal** (Not Used in Final Pipeline)
- Pros: Reduces vocabulary, focuses on content words
- Cons: May remove sentiment-bearing words ("not", "no")
- Decision: Kept stop words for initial models, can experiment later

**Stemming** (Not Used in Final Pipeline)
- Pros: "running", "runs", "ran" → "run" (reduces vocabulary)
- Cons: May lose semantic meaning ("universe" → "univers")
- Decision: Avoided for initial models, lemmatization preferred if needed

**Lemmatization** (Not Used in Final Pipeline)
- Pros: More accurate than stemming ("better" → "good")
- Cons: Slower processing, requires POS tagging
- Decision: Reserved for advanced models if baseline performance is insufficient

### Feature Extraction Methods

**TF-IDF (Term Frequency-Inverse Document Frequency)**
- **Configuration**:
  - max_features: 10,000 (top 10k most important words)
  - min_df: 5 (word must appear in at least 5 documents)
  - max_df: 0.7 (word must appear in less than 70% of documents)
  - ngram_range: (1, 2) (unigrams and bigrams)

- **Justification**:
  - Weights words by importance (rare words get higher weight)
  - Reduces impact of frequent but uninformative words
  - Bigrams capture phrases like "not good", "very bad"
  - Standard baseline for text classification

- **Output**: Sparse matrix (20,000 × 10,000) for training set

**Bag-of-Words (Count Vectorizer)**
- **Configuration**: Same as TF-IDF
- **Justification**:
  - Simple baseline for comparison
  - Works well with Naive Bayes
  - Interpretable feature values (raw counts)
- **Output**: Sparse matrix (20,000 × 10,000) for training set

**Future: Word Embeddings** (Not Implemented Yet)
- Word2Vec, GloVe, or pre-trained embeddings
- For deep learning models (LSTM, CNN)
- Capture semantic similarity between words

**Future: Contextual Embeddings** (Advanced)
- BERT, RoBERTa, or domain-specific models
- For state-of-the-art performance (94-95% accuracy)
- Requires GPU for fine-tuning

## 5. Train/Validation/Test Split Strategy

### Splitting Approach
- **Original split**: 25k train / 25k test (provided by dataset)
- **Our split**: 20k train / 5k validation / 25k test

### Rationale
1. **Validation set purpose**:
   - Hyperparameter tuning without touching test set
   - Early stopping for deep learning models
   - Model selection among multiple candidates

2. **Stratification**:
   - All splits maintain 50/50 positive/negative ratio
   - Prevents sampling bias
   - Ensures representative performance estimation

3. **Test set integrity**:
   - Original test set kept completely separate
   - Never used for training or validation
   - Final evaluation only after all tuning is complete

### Split Sizes
- **Training**: 20,000 samples (64% of train+val)
  - Positive: 10,000
  - Negative: 10,000

- **Validation**: 5,000 samples (16% of train+val)
  - Positive: 2,500
  - Negative: 2,500

- **Test**: 25,000 samples (provided by dataset)
  - Positive: 12,500
  - Negative: 12,500

### Random Seed
- **Fixed seed**: 42 (for reproducibility)
- All random operations use same seed
- Ensures consistent results across runs

## 6. Prior Work and Benchmarks

### Original Paper (Maas et al., 2011)
- **Model**: Logistic Regression with word vectors
- **Accuracy**: 88.89%
- **Key contribution**: Learned word vectors for sentiment

### Classical Machine Learning Benchmarks
- **Naive Bayes (Multinomial)**: ~85-87%
- **Logistic Regression**: ~88-89%
- **SVM (Linear kernel)**: ~88-90%
- **Random Forest**: ~85-87%
- **XGBoost**: ~88-90%

### Deep Learning Benchmarks
- **LSTM (single layer)**: ~86-88%
- **LSTM (bidirectional, stacked)**: ~88-90%
- **CNN (Kim, 2014)**: ~89-91%
- **Attention-based models**: ~90-92%

### Transformer-Based Benchmarks (2019+)
- **BERT (base, fine-tuned)**: ~93-94%
- **RoBERTa**: ~94-95%
- **DistilBERT**: ~92-93%
- **ALBERT**: ~94%

### Our Performance Targets
- **Baseline (Logistic Regression)**: ≥ 88%
- **Advanced ML (SVM, Ensemble)**: ≥ 89%
- **Deep Learning (LSTM, CNN)**: ≥ 90%
- **Transformer (BERT)**: ≥ 93% (optional, if time permits)

## 7. Evaluation Strategy

### Metrics
1. **Accuracy**: Overall correctness (appropriate for balanced dataset)
2. **Precision**: Positive predictive value (important for both classes)
3. **Recall**: Sensitivity (important for both classes)
4. **F1-Score**: Harmonic mean of precision and recall
5. **Confusion Matrix**: Detailed error analysis

### Cross-Validation
- **Not primary method**: Dataset is large enough for single train/val/test split
- **Optional**: 5-fold CV on training set for robust estimation
- **Use case**: Hyperparameter tuning with grid search

### Comparison with Benchmarks
- Compare our results with published benchmarks
- Analyze performance gap (if any) and reasons
- Discuss strengths and limitations in final report

## 8. Deployment Plan

### Web Application Design
- **Framework**: Flask or Streamlit
- **Input**: Text area for movie review (500-1000 character limit)
- **Processing**: Real-time preprocessing and prediction
- **Output**: Sentiment label (positive/negative) with confidence score
- **Features**:
  - Clean, intuitive interface
  - Example reviews for users to test
  - Visualization of prediction confidence
  - Optional: Word importance highlighting

### Model Selection for Deployment
- **Best performing model** from evaluation phase
- **Considerations**:
  - Inference speed (should be < 1 second)
  - Model size (lightweight models preferred)
  - Accuracy vs. speed tradeoff
- **Likely choice**: Logistic Regression or SVM (fast, accurate, small size)

## 9. Limitations and Future Work

### Dataset Limitations
1. **Binary classification**: Only positive/negative, no neutral sentiment
2. **Domain-specific**: Movie reviews may not generalize to other domains
3. **Temporal bias**: Reviews from 2011 or earlier, language may evolve
4. **English-only**: Not multilingual
5. **Formal reviews**: May not capture informal sentiment (e.g., tweets)

### Potential Improvements
1. **Data augmentation**: Back-translation, synonym replacement
2. **Ensemble methods**: Combine multiple models for better performance
3. **Transfer learning**: Fine-tune pre-trained language models (BERT)
4. **Explainability**: LIME or SHAP for model interpretation
5. **Multi-domain**: Test on other sentiment datasets (Amazon, Yelp)

## 10. References

### Primary Citation
```bibtex
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L. and Daly, Raymond E. and Pham, Peter T. and Huang, Dan and Ng, Andrew Y. and Potts, Christopher},
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

### Additional References
- Dataset URL: http://ai.stanford.edu/~amaas/data/sentiment/
- Hugging Face: https://huggingface.co/datasets/imdb
- Benchmark results: Papers with Code - IMDB Movie Reviews

## 11. Conclusion

The IMDB Movie Reviews dataset is an excellent choice for this NLP assignment because it:
1. Meets all assignment requirements (size, quality, benchmarks)
2. Allows demonstration of multiple NLP techniques
3. Supports various model architectures (classical ML and deep learning)
4. Has clear evaluation metrics and benchmarks
5. Is straightforward to deploy as a web application
6. Provides educational value with interpretable results

This dataset will enable comprehensive model comparison, hyperparameter tuning, and meaningful evaluation against established benchmarks, fulfilling the assignment objectives.
