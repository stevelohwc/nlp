# News-Classification-NLP

Standalone fake-news classification project (independent from IMDB sentiment files).

## What this project includes

- News-only dataset pipeline (`fake.csv`, `true.csv`)
- Hyperparameter tuning with 4 models:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Linear SVM (`LinearSVC`)
  - Random Forest
- Best model bundle with metadata/version contract
- Streamlit app with:
  - Single prediction
  - Benchmark validation on labeled holdout split
  - Live URL extraction + prediction (inference-only disclaimer)

## Directory layout

```text
Code/News-Classification-NLP/
├── app.py
├── frontend_page.py
├── app_core.py
├── data_loader.py
├── text_preprocessing.py
├── run_preprocessing.py
├── train_baseline_models.py
├── train_advanced_models.py
├── evaluate.py
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── evaluation_plots/
└── docs/
```

## Setup

```bash
cd /Users/weichn/Applications/smartgit/nlp/Code/News-Classification-NLP
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

This project expects exactly these input files:
- `data/raw/fake.csv`
- `data/raw/true.csv`

Primary dataset source:
- Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

### Option A: Auto-download dataset (Kaggle API)

1. Create a Kaggle API token in your Kaggle account (`kaggle.json`).
2. Put the file at `~/.kaggle/kaggle.json`.
3. Set permission:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

4. Validate and auto-download if files are missing:

```bash
python data_loader.py --data-dir data/raw --download-if-missing
```

### Option B: Manual dataset copy

Download `fake.csv` and `true.csv`, then place them in:
- `data/raw/fake.csv`
- `data/raw/true.csv`

## How to train the model

Run from `Code/News-Classification-NLP/` after setup:

### 1) Preprocess and build TF-IDF features

```bash
python run_preprocessing.py --data-dir data/raw --out-dir data/processed
```

Main outputs:
- `data/processed/tfidf_features.pkl`
- `data/processed/preprocessor.pkl`
- `data/processed/vectorizer.pkl`
- `data/processed/train_preprocessed.csv`
- `data/processed/val_preprocessed.csv`
- `data/processed/test_preprocessed.csv`

### 2) Train baseline models (NB + Logistic Regression)

```bash
python train_baseline_models.py --processed-dir data/processed
```

Main outputs:
- `models/naive_bayes_model.pkl`
- `models/logistic_regression_model.pkl`
- `reports/baseline_results.json`

### 3) Train advanced models and build best bundle

```bash
python train_advanced_models.py --processed-dir data/processed
```

Main outputs:
- `models/linear_svc_model.pkl`
- `models/random_forest_model.pkl`
- `models/best_model_bundle.pkl`
- `reports/advanced_results.json`

### 4) Evaluate all trained models (optional but recommended)

```bash
python evaluate.py --models-dir models --processed-dir data/processed
```

Main outputs:
- `reports/evaluation_report.json`
- `evaluation_plots/confusion_matrices.png`
- `evaluation_plots/roc_curves.png`
- `evaluation_plots/model_comparison.png`

## How to run `app.py`

The app needs `models/best_model_bundle.pkl` and `data/processed/tfidf_features.pkl`.
If missing, run the training pipeline above first.

```bash
cd /Users/weichn/Applications/smartgit/nlp/Code/News-Classification-NLP
source .venv/bin/activate
streamlit run app.py
```

Then open the local Streamlit URL shown in terminal (usually `http://localhost:8501`).

## Related docs

- [Hyperparameter tuning details](docs/HYPERPARAMETER_TUNING.md)
- [Real-case validation workflow](docs/REAL_CASE_VALIDATION.md)
- [UI robustness and safety checks](docs/UI_ROBUSTNESS.md)
- [Training notebook](news_classification_model_training.ipynb)

## Model bundle contract

`models/best_model_bundle.pkl` contains:
- `model`
- `vectorizer`
- `preprocessor`
- `label_map`
- `metrics`
- `training_config`
- `library_versions`

## Notes

- `Code/text_classification` is reference-only and untouched by this standalone project.
- Live URL mode is inference only; it is not a factual verification system.
