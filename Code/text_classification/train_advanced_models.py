"""
Train advanced models for sentiment classification.
Models: LinearSVC (with GridSearchCV) and Random Forest (with RandomizedSearchCV).
Mirrors the style and memory-management patterns of train_baseline_models.py.
"""

import pickle
import sys
import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import json
import time
import gc
import psutil
import os
from scipy import sparse
from scipy.stats import uniform


# ---------------------------------------------------------------------------
# Utility functions (mirrors baseline script)
# ---------------------------------------------------------------------------
def compute_safe_cores():
    total_cores = os.cpu_count() or 1
    safe_cores = max(1, total_cores - 2) if total_cores > 2 else 1
    return total_cores, safe_cores


def format_bytes(num_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def sparse_nbytes(matrix):
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes


def ensure_csr(name, matrix):
    if sparse.issparse(matrix) and not sparse.isspmatrix_csr(matrix):
        print(f"  Converting {name} to CSR format …")
        return matrix.tocsr()
    return matrix


def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"  Memory: {memory_info.rss / 1024 / 1024:.1f} MB ({process.memory_percent():.1f}%)")


def guard_memory_usage(matrix, ratio=0.6, multiplier=3.0):
    if not sparse.issparse(matrix):
        print("\nERROR: Features are dense. Aborting.")
        sys.exit(1)
    available = psutil.virtual_memory().available
    matrix_bytes = sparse_nbytes(matrix)
    estimated = matrix_bytes * multiplier
    budget = available * ratio
    print(f"  Memory Guard: matrix={format_bytes(matrix_bytes)} "
          f"est={format_bytes(estimated)} budget={format_bytes(budget)}")
    if estimated > budget:
        print("\nERROR: Estimated memory exceeds budget. Reduce data or close apps.")
        sys.exit(1)


def print_metrics(y_true, y_pred, model_name, split_name="Validation"):
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall    = recall_score(y_true, y_pred, average='binary')
    f1        = f1_score(y_true, y_pred, average='binary')

    print(f"\n  {model_name} — {split_name} Set:")
    print(f"    Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    [[TN={cm[0, 0]:5d}  FP={cm[0, 1]:5d}]")
    print(f"     [FN={cm[1, 0]:5d}  TP={cm[1, 1]:5d}]]")

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist()
    }


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def main():
    total_cores, safe_cores = compute_safe_cores()

    print("=" * 80)
    print("ADVANCED MODEL TRAINING — IMDB Sentiment Classification")
    print("=" * 80)
    print(f"\nSystem: {total_cores} CPU cores available, using {safe_cores} for training")
    print_memory_usage()

    # ------------------------------------------------------------------
    # [1] Load TF-IDF features (shared by both advanced models)
    # ------------------------------------------------------------------
    print("\n[1/5] Loading TF-IDF features …")
    processed_dir = Path(__file__).parent.parent / 'datasets' / 'processed'

    with open(processed_dir / 'tfidf_features.pkl', 'rb') as f:
        tfidf_data = pickle.load(f)

    tfidf_vectorizer = tfidf_data.get('vectorizer')
    X_train = tfidf_data['X_train']
    X_val   = tfidf_data['X_val']
    X_test  = tfidf_data['X_test']
    y_train = np.asarray(tfidf_data['y_train'])
    y_val   = np.asarray(tfidf_data['y_val'])
    y_test  = np.asarray(tfidf_data['y_test'])
    del tfidf_data
    gc.collect()

    X_train = ensure_csr("X_train", X_train)
    X_val   = ensure_csr("X_val",   X_val)
    X_test  = ensure_csr("X_test",  X_test)

    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print_memory_usage()
    guard_memory_usage(X_train)

    results = {}

    # ------------------------------------------------------------------
    # [2] LinearSVC + GridSearchCV
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[2/5] Training LinearSVC with GridSearchCV")
    print("-" * 80)

    param_grid = {
        'C': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    }
    svm_base = LinearSVC(max_iter=5000, random_state=42, dual=True)

    print(f"  GridSearchCV: {len(param_grid['C'])} C values × 3-fold CV = "
          f"{len(param_grid['C']) * 3} fits")
    start_time = time.time()
    grid_search = GridSearchCV(
        svm_base, param_grid,
        cv=3, scoring='accuracy', n_jobs=safe_cores, verbose=1
    )
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start_time

    svm_model = grid_search.best_estimator_
    print(f"\n  Best C = {grid_search.best_params_['C']}  "
          f"(CV accuracy = {grid_search.best_score_:.4f})")
    print(f"  Training completed in {train_time:.2f} seconds")
    print_memory_usage()

    # Evaluate
    svm_val_metrics  = print_metrics(y_val, svm_model.predict(X_val),  "LinearSVC", "Validation")
    svm_test_metrics = print_metrics(y_test, svm_model.predict(X_test), "LinearSVC", "Test")

    results['svm'] = {
        'model': 'LinearSVC',
        'features': 'TF-IDF',
        'hyperparameters': {'C': grid_search.best_params_['C'], 'max_iter': 5000},
        'cv_best_score': float(grid_search.best_score_),
        'training_time': train_time,
        'validation_metrics': svm_val_metrics,
        'test_metrics': svm_test_metrics
    }

    # ------------------------------------------------------------------
    # [3] Random Forest + RandomizedSearchCV
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[3/5] Training Random Forest with RandomizedSearchCV")
    print("-" * 80)

    param_distributions = {
        'n_estimators':      [100, 200, 300, 500],
        'max_depth':         [10, 20, 30, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf':  [1, 2, 4]
    }

    rf_base = RandomForestClassifier(random_state=42, n_jobs=safe_cores)

    print(f"  RandomizedSearchCV: n_iter=20, 3-fold CV = 60 fits")
    start_time = time.time()
    rand_search = RandomizedSearchCV(
        rf_base, param_distributions,
        n_iter=20, cv=3, scoring='accuracy',
        n_jobs=1,          # each RF already uses safe_cores internally
        random_state=42, verbose=1
    )
    rand_search.fit(X_train, y_train)
    train_time = time.time() - start_time

    rf_model = rand_search.best_estimator_
    print(f"\n  Best params: {rand_search.best_params_}")
    print(f"  CV accuracy = {rand_search.best_score_:.4f}")
    print(f"  Training completed in {train_time:.2f} seconds")
    print_memory_usage()

    # Evaluate
    rf_val_metrics  = print_metrics(y_val, rf_model.predict(X_val),  "Random Forest", "Validation")
    rf_test_metrics = print_metrics(y_test, rf_model.predict(X_test), "Random Forest", "Test")

    results['random_forest'] = {
        'model': 'RandomForestClassifier',
        'features': 'TF-IDF',
        'hyperparameters': {k: (v if v is not None else 'None')
                            for k, v in rand_search.best_params_.items()},
        'cv_best_score': float(rand_search.best_score_),
        'training_time': train_time,
        'validation_metrics': rf_val_metrics,
        'test_metrics': rf_test_metrics
    }

    # ------------------------------------------------------------------
    # [4] Determine overall best model (across all 4)
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[4/5] Model Comparison (all 4 models)")
    print("-" * 80)

    # Load baseline results for comparison
    models_dir = Path(__file__).parent.parent / 'models'
    with open(models_dir / 'baseline_results.json', 'r') as f:
        baseline = json.load(f)

    all_models = {
        'Naive Bayes':          baseline['naive_bayes']['validation_metrics'],
        'Logistic Regression':  baseline['logistic_regression']['validation_metrics'],
        'LinearSVC':            svm_val_metrics,
        'Random Forest':        rf_val_metrics,
    }

    print(f"\n  {'Model':<25} {'Val Acc':<12} {'Val Prec':<12} {'Val Rec':<12} {'Val F1':<12}")
    print(f"  {'-' * 73}")
    for name, m in all_models.items():
        print(f"  {name:<25} {m['accuracy']:.4f}       {m['precision']:.4f}       "
              f"{m['recall']:.4f}       {m['f1']:.4f}")

    # Pick the best by validation accuracy
    best_name = max(all_models, key=lambda k: all_models[k]['accuracy'])
    print(f"\n  Overall best (validation): {best_name} "
          f"({all_models[best_name]['accuracy'] * 100:.2f}%)")

    # ------------------------------------------------------------------
    # [5] Save models & results
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[5/5] Saving models and results")
    print("-" * 80)

    models_dir.mkdir(parents=True, exist_ok=True)

    # SVM
    with open(models_dir / 'svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    print("  Saved svm_model.pkl")

    # Random Forest
    with open(models_dir / 'random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("  Saved random_forest_model.pkl")

    # Advanced results JSON
    with open(models_dir / 'advanced_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("  Saved advanced_results.json")

    # --- Update best_model.pkl (overall best across all 4) ---
    # Map best_name back to the actual model object + vectorizer
    best_model_map = {
        'Naive Bayes': {
            'model_file': 'naive_bayes_model.pkl',
            'features': 'bow'
        },
        'Logistic Regression': {
            'model_file': 'logistic_regression_model.pkl',
            'features': 'tfidf'
        },
        'LinearSVC': {
            'model_obj': svm_model,
            'features': 'tfidf'
        },
        'Random Forest': {
            'model_obj': rf_model,
            'features': 'tfidf'
        },
    }

    info = best_model_map[best_name]
    if 'model_obj' in info:
        best_model_obj = info['model_obj']
    else:
        with open(models_dir / info['model_file'], 'rb') as f:
            best_model_obj = pickle.load(f)

    # Load the appropriate vectorizer
    feat_key = 'tfidf_features.pkl' if info['features'] == 'tfidf' else 'bow_features.pkl'
    with open(processed_dir / feat_key, 'rb') as f:
        feat_data = pickle.load(f)
    best_vectorizer = feat_data.get('vectorizer')
    del feat_data
    gc.collect()

    with open(models_dir / 'best_model.pkl', 'wb') as f:
        pickle.dump({
            'model':      best_model_obj,
            'model_name': best_name,
            'features':   info['features'],
            'vectorizer': best_vectorizer
        }, f)
    print(f"  Saved best_model.pkl  →  {best_name}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ADVANCED MODEL TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n  Trained 2 advanced models (LinearSVC, Random Forest)")
    print(f"  Overall best model: {best_name}")
    print(f"  Target benchmark: >=88% (Maas et al., 2011)")

    best_test_acc = max(
        baseline['naive_bayes']['test_metrics']['accuracy'],
        baseline['logistic_regression']['test_metrics']['accuracy'],
        svm_test_metrics['accuracy'],
        rf_test_metrics['accuracy']
    )
    if best_test_acc >= 0.88:
        print(f"  SUCCESS — best test accuracy {best_test_acc * 100:.2f}% meets benchmark")
    else:
        print(f"  Best test accuracy {best_test_acc * 100:.2f}% — below benchmark")

    print_memory_usage()
    print("=" * 80)


if __name__ == "__main__":
    main()
