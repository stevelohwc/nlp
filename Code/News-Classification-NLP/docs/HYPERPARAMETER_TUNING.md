# Hyperparameter Tuning

## Models
- Baseline: `MultinomialNB`, `LogisticRegression`
- Advanced: `LinearSVC`, `RandomForestClassifier`

## Selection policy
Model selection uses:
1. Validation F1 (primary)
2. Validation Recall (secondary)
3. Validation Accuracy (tie-break)

## Search configuration
- Cross-validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Baseline script: `GridSearchCV`
- Advanced script:
  - `GridSearchCV` for `LinearSVC`
  - `RandomizedSearchCV` for `RandomForestClassifier`

## Artifacts
- Baseline report: `reports/baseline_results.json`
- Advanced report: `reports/advanced_results.json`
- Best model bundle: `models/best_model_bundle.pkl`

## Reproducibility notes
- Fixed random state defaults are used in all scripts.
- Preprocessing/vectorization artifacts are persisted in `data/processed/`.
- Best model bundle stores training config and library version metadata.
