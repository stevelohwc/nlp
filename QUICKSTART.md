# Quick Start Guide - NLP Assignment Text Classification

## What Has Been Done ✅

The dataset selection and preparation phase is complete:

1. **IMDB Dataset Selected**: 50,000 movie reviews for sentiment analysis
2. **Dataset Downloaded**: 25k training + 25k test samples (63MB total)
3. **Data Loader Created**: Python utility for easy dataset access
4. **EDA Notebook Ready**: Comprehensive exploratory data analysis
5. **Preprocessing Notebook Ready**: Full text preprocessing pipeline
6. **Documentation Complete**: Detailed rationale and usage guides

## Project Structure

```
nlp/
├── requirements.txt                    # Python dependencies
├── QUICKSTART.md                       # This file
├── CLAUDE.md                          # Project instructions
├── Code/
│   ├── IMPLEMENTATION_STATUS.md       # Detailed progress tracker
│   ├── datasets/
│   │   ├── README.md                  # Dataset usage guide
│   │   ├── DATASET_INFO.md            # 7000+ word documentation
│   │   ├── raw/
│   │   │   ├── imdb_train.csv         # 25k training reviews (32MB)
│   │   │   ├── imdb_test.csv          # 25k test reviews (31MB)
│   │   │   └── imdb/                  # Hugging Face cache
│   │   └── processed/                 # Will contain preprocessed data
│   └── text_classification/
│       ├── data_loader.py             # Dataset loading utilities ✅
│       ├── eda.ipynb                  # EDA notebook (ready) ✅
│       └── preprocessing.ipynb         # Preprocessing (ready) ✅
└── Documentations/                    # Assignment specs
```

## Next Steps

### Step 1: Install Dependencies

```bash
cd /Users/weichn/Applications/smartgit/nlp
pip install -r requirements.txt
```

This will install:
- pandas, numpy (data manipulation)
- scikit-learn (ML models)
- matplotlib, seaborn, wordcloud (visualization)
- nltk (NLP preprocessing)
- datasets (Hugging Face)
- jupyter (notebooks)

### Step 2: Run Exploratory Data Analysis

```bash
jupyter notebook Code/text_classification/eda.ipynb
```

**What this does**:
- Loads the 50k IMDB reviews
- Analyzes class distribution (perfectly balanced)
- Examines text length patterns
- Performs vocabulary analysis
- Generates word clouds for positive/negative reviews
- Creates summary statistics

**Output**: Visualizations and insights for your report

**Time**: 5-10 minutes to run all cells

### Step 3: Run Preprocessing Pipeline

```bash
jupyter notebook Code/text_classification/preprocessing.ipynb
```

**What this does**:
- Cleans text (HTML removal, normalization)
- Splits data (20k train / 5k validation / 25k test)
- Creates TF-IDF features (10k features, unigrams+bigrams)
- Creates Bag-of-Words features
- Saves preprocessed data to `Code/datasets/processed/`

**Output**:
- `train_preprocessed.csv`, `test_preprocessed.csv`
- `data_splits.pkl` (train/val/test splits)
- `tfidf_features.pkl` (ready for model training)
- `bow_features.pkl` (alternative features)
- `preprocessor.pkl` (reusable preprocessor)

**Time**: 10-15 minutes to run all cells

### Step 4: Train Baseline Models (Safe Script)

```bash
python Code/text_classification/train_baseline_models.py
```

Optional flags:
- `--sample-size 2000` (quick, low-memory run)
- `--n-jobs 2` (limit CPU usage)
- `--solver saga` (default, sparse-safe)
- `--max-iter 200`
- `--memory-guard-ratio 0.6`
- `--memory-guard-multiplier 3.0`

Example quick run:
```bash
python Code/text_classification/train_baseline_models.py --sample-size 2000 --n-jobs 2
```

This script includes a memory guard to prevent system hangs and saves models to `Code/models/`.

**Target**: ≥88% accuracy

### Step 5: Train Advanced Models (TODO)

- Support Vector Machine (SVM)
- Random Forest
- XGBoost
- Hyperparameter tuning with GridSearchCV

### Step 6: Deploy as Web App (TODO)

Create Flask or Streamlit app for real-time sentiment prediction.

## Key Files to Reference

### For Report Writing:
1. **`Code/datasets/DATASET_INFO.md`**:
   - Complete dataset selection justification
   - Preprocessing decisions explained
   - Benchmark comparisons
   - Use sections directly in your report

2. **`Code/IMPLEMENTATION_STATUS.md`**:
   - Track your progress
   - Update as you complete each phase
   - Use as outline for methodology section

3. **EDA Notebook Output**:
   - Include visualizations in report
   - Reference statistics in data description
   - Use word clouds to show sentiment patterns

### For Coding:
1. **`Code/text_classification/data_loader.py`**:
   - Use `IMDBDataLoader()` to load dataset
   - Access train/test splits easily
   - Get dataset statistics

2. **`Code/datasets/README.md`**:
   - Quick reference for dataset usage
   - Example code snippets
   - File format descriptions

## Testing the Setup

Run this Python script to verify everything works:

```python
# test_setup.py
import sys
sys.path.append('Code')

from text_classification.data_loader import IMDBDataLoader

# Load dataset
print("Loading IMDB dataset...")
loader = IMDBDataLoader()
train_df, test_df = loader.load_dataset()

# Display info
info = loader.get_dataset_info()
print("\nDataset loaded successfully!")
print(f"Training samples: {info['train_size']:,}")
print(f"Test samples: {info['test_size']:,}")
print(f"Classes: {info['class_names']}")

# Show sample
samples = loader.get_sample_reviews(n=1, sentiment='positive')
print(f"\nSample positive review:")
print(samples[0]['text'][:200] + "...")

print("\n✅ Setup verified! Ready to proceed with EDA and preprocessing.")
```

Save as `test_setup.py` and run:
```bash
python test_setup.py
```

## Common Issues and Solutions

### Issue: Import errors
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Dataset not found
**Solution**: Re-run data loader
```bash
cd Code/text_classification
python data_loader.py
```

### Issue: Jupyter kernel not found
**Solution**: Install ipykernel
```bash
pip install ipykernel
python -m ipykernel install --user --name nlp --display-name "Python (NLP)"
```

### Issue: Out of memory
**Solution**: Use smaller max_features in TF-IDF
```python
TfidfVectorizer(max_features=5000)  # Instead of 10000
```

## Performance Benchmarks

Target accuracies for your models:

| Model | Target Accuracy |
|-------|----------------|
| Naive Bayes | ≥85% |
| Logistic Regression | ≥88% |
| SVM | ≥89% |
| Random Forest | ≥87% |
| XGBoost | ≥89% |
| LSTM | ≥90% |
| BERT (optional) | ≥93% |

## Timeline Suggestion

- **Week 1**: EDA, preprocessing, baseline models (Done: dataset prep ✅)
- **Week 2**: Advanced models, hyperparameter tuning
- **Week 3**: Deep learning (optional), web deployment
- **Week 4**: Spelling correction system
- **Week 5**: Report writing
- **Week 6**: Demonstration video, final review

**Deadline**: 27.03.2026

## Questions?

- Check `Code/datasets/README.md` for dataset usage
- Check `Code/datasets/DATASET_INFO.md` for detailed documentation
- Check `Code/IMPLEMENTATION_STATUS.md` for progress tracking
- Review `CLAUDE.md` for overall project requirements

## Summary

You now have:
1. ✅ Complete dataset (50k IMDB reviews)
2. ✅ Data loading utilities
3. ✅ EDA notebook ready to run
4. ✅ Preprocessing pipeline ready to run
5. ✅ Comprehensive documentation

**Next action**: Install dependencies and run the EDA notebook!

```bash
pip install -r requirements.txt
jupyter notebook Code/text_classification/eda.ipynb
```

Good luck with your NLP assignment! 🚀
