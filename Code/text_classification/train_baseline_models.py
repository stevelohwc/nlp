"""
Train baseline models for sentiment classification
Models: Naive Bayes and Logistic Regression
"""

import argparse
import pickle
import sys
import numpy as np
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import json
import time
import gc
import psutil
import os
from scipy import sparse

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
        print(f"  Converting {name} to CSR format for efficient indexing...")
        return matrix.tocsr()
    return matrix

def parse_args():
    total_cores, safe_cores = compute_safe_cores()
    parser = argparse.ArgumentParser(
        description="Train baseline models for IMDB sentiment classification"
    )
    parser.add_argument("--solver", default="saga",
                        choices=["saga", "sag", "liblinear", "lbfgs", "newton-cg"],
                        help="Logistic Regression solver (default: saga)")
    parser.add_argument("--n-jobs", type=int, default=safe_cores,
                        help=f"Number of CPU cores to use (default: {safe_cores})")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="Maximum iterations for Logistic Regression (default: 200)")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Optional downsample size for training split")
    parser.add_argument("--memory-guard-ratio", type=float, default=0.6,
                        help="Fraction of available RAM allowed for estimated training memory (default: 0.6)")
    parser.add_argument("--memory-guard-multiplier", type=float, default=3.0,
                        help="Multiplier for estimating training memory use (default: 3.0)")
    return parser.parse_args()

def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    memory_percent = process.memory_percent()
    print(f"  Memory: {memory_mb:.1f} MB ({memory_percent:.1f}%)")

def print_metrics(y_true, y_pred, model_name, split_name="Validation"):
    """Print comprehensive metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    print(f"\n  {model_name} - {split_name} Set Performance:")
    print(f"    Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    [[TN={cm[0,0]:5d}  FP={cm[0,1]:5d}]")
    print(f"     [FN={cm[1,0]:5d}  TP={cm[1,1]:5d}]]")

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist()
    }

def guard_memory_usage(matrix, ratio, multiplier):
    if not sparse.issparse(matrix):
        print("\nERROR: TF-IDF features are dense in memory.")
        print("This can cause extreme RAM usage and system instability.")
        print("Please re-run preprocessing to ensure sparse feature output,")
        print("or reduce the feature size and try again.")
        sys.exit(1)

    available = psutil.virtual_memory().available
    matrix_bytes = sparse_nbytes(matrix)
    estimated = matrix_bytes * multiplier
    budget = available * ratio

    print("\n  Memory Guard:")
    print(f"    Sparse matrix size: {format_bytes(matrix_bytes)}")
    print(f"    Estimated training memory: {format_bytes(estimated)}")
    print(f"    Allowed budget (ratio {ratio:.2f}): {format_bytes(budget)}")

    if estimated > budget:
        print("\nERROR: Estimated training memory exceeds allowed budget.")
        print("Suggestions:")
        print("  - Use --sample-size to reduce training rows")
        print("  - Re-run preprocessing with a smaller max_features")
        print("  - Reduce n_jobs or close other memory-heavy applications")
        sys.exit(1)

def main():
    args = parse_args()

    if args.n_jobs < 1:
        print("WARNING: --n-jobs must be >= 1. Using 1.")
        args.n_jobs = 1
    if not (0 < args.memory_guard_ratio <= 1.0):
        print("WARNING: --memory-guard-ratio must be in (0, 1]. Using 0.6.")
        args.memory_guard_ratio = 0.6
    if args.memory_guard_multiplier <= 0:
        print("WARNING: --memory-guard-multiplier must be > 0. Using 3.0.")
        args.memory_guard_multiplier = 3.0

    print("="*80)
    print("BASELINE MODEL TRAINING - IMDB Sentiment Classification")
    print("="*80)

    # Determine safe number of CPU cores (leave some for OS)
    total_cores, safe_cores = compute_safe_cores()
    print(f"\nSystem: {total_cores} CPU cores available, using {args.n_jobs} for training")
    print_memory_usage()

    # Load preprocessed features
    print("\n[1/5] Loading preprocessed features...")
    processed_dir = Path(__file__).parent.parent / 'datasets' / 'processed'

    # Load BoW features FIRST (for Naive Bayes) - more memory efficient
    print("  Loading BoW features for Naive Bayes...")
    with open(processed_dir / 'bow_features.pkl', 'rb') as f:
        bow_data = pickle.load(f)

    bow_vectorizer = bow_data.get('vectorizer')
    X_train_bow = bow_data['X_train']
    X_val_bow = bow_data['X_val']
    X_test_bow = bow_data['X_test']
    y_train = np.asarray(bow_data['y_train'])
    y_val = np.asarray(bow_data['y_val'])
    y_test = np.asarray(bow_data['y_test'])

    # Free dict reference early to avoid retaining large matrices
    del bow_data
    gc.collect()

    # Optional downsample for training split only
    train_sample_idx = None
    if args.sample_size is not None:
        if args.sample_size < 1:
            print("WARNING: --sample-size must be >= 1. Ignoring.")
        elif args.sample_size < len(y_train):
            rng = np.random.default_rng(42)
            train_sample_idx = rng.choice(len(y_train), size=args.sample_size, replace=False)
            train_sample_idx.sort()
            X_train_bow = X_train_bow[train_sample_idx]
            y_train = y_train[train_sample_idx]
            print(f"  Downsampled training set to {len(y_train):,} samples")
        else:
            print("  --sample-size >= training size; using full training set")

    print(f"  Train: {X_train_bow.shape}, Val: {X_val_bow.shape}, Test: {X_test_bow.shape}")
    print_memory_usage()

    # Store results
    results = {}

    # ========================================
    # Model 1: Naive Bayes (with BoW features)
    # ========================================
    print("\n" + "-"*80)
    print("[2/5] Training Naive Bayes (Multinomial)")
    print("-"*80)

    print("\n  Initializing MultinomialNB...")
    nb_model = MultinomialNB(alpha=1.0)

    print(f"  Training on {len(y_train):,} samples...")
    start_time = time.time()
    nb_model.fit(X_train_bow, y_train)
    train_time = time.time() - start_time
    print(f"  Training completed in {train_time:.2f} seconds")
    print_memory_usage()

    # Validation set predictions
    print("\n  Evaluating on validation set...")
    y_val_pred_nb = nb_model.predict(X_val_bow)
    nb_val_metrics = print_metrics(y_val, y_val_pred_nb, "Naive Bayes", "Validation")

    # Test set predictions
    print("\n  Evaluating on test set...")
    y_test_pred_nb = nb_model.predict(X_test_bow)
    nb_test_metrics = print_metrics(y_test, y_test_pred_nb, "Naive Bayes", "Test")

    results['naive_bayes'] = {
        'model': 'MultinomialNB',
        'features': 'Bag-of-Words',
        'hyperparameters': {'alpha': 1.0},
        'training_time': train_time,
        'validation_metrics': nb_val_metrics,
        'test_metrics': nb_test_metrics
    }

    # Clean up BoW features to free memory
    print("\n  Cleaning up BoW features from memory...")
    del X_train_bow, X_val_bow, X_test_bow
    gc.collect()
    print_memory_usage()

    # ========================================
    # Model 2: Logistic Regression (with TF-IDF)
    # ========================================
    print("\n" + "-"*80)
    print("[3/5] Training Logistic Regression")
    print("-"*80)

    # Now load TF-IDF features (after BoW cleanup)
    print("\n  Loading TF-IDF features...")
    with open(processed_dir / 'tfidf_features.pkl', 'rb') as f:
        tfidf_data = pickle.load(f)

    tfidf_vectorizer = tfidf_data.get('vectorizer')
    X_train = tfidf_data['X_train']
    X_val = tfidf_data['X_val']
    X_test = tfidf_data['X_test']

    # Free dict reference early to avoid retaining large matrices
    del tfidf_data
    gc.collect()

    # Ensure sparse CSR format for safe memory usage
    X_train = ensure_csr("X_train", X_train)
    X_val = ensure_csr("X_val", X_val)
    X_test = ensure_csr("X_test", X_test)

    # Apply same downsample indices used for BoW training
    if train_sample_idx is not None:
        X_train = X_train[train_sample_idx]

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print_memory_usage()

    # Memory guard to prevent system instability
    guard_memory_usage(X_train, args.memory_guard_ratio, args.memory_guard_multiplier)

    print("\n  Initializing LogisticRegression...")
    print(f"  Hyperparameters: C=1.0, max_iter={args.max_iter}, solver='{args.solver}', n_jobs={args.n_jobs}")
    lr_model = LogisticRegression(
        C=1.0,
        max_iter=args.max_iter,
        solver=args.solver,
        random_state=42,
        n_jobs=args.n_jobs,  # Use safe number of cores, not all
        verbose=1  # Show convergence progress
    )

    print(f"  Training on {len(y_train):,} samples...")
    print("  (This may take 1-3 minutes. Progress updates will appear below)")
    start_time = time.time()
    lr_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"\n  Training completed in {train_time:.2f} seconds")
    print(f"  Converged: {lr_model.n_iter_} iterations")
    print_memory_usage()

    # Validation set predictions
    print("\n  Evaluating on validation set...")
    y_val_pred_lr = lr_model.predict(X_val)
    lr_val_metrics = print_metrics(y_val, y_val_pred_lr, "Logistic Regression", "Validation")

    # Test set predictions
    print("\n  Evaluating on test set...")
    y_test_pred_lr = lr_model.predict(X_test)
    lr_test_metrics = print_metrics(y_test, y_test_pred_lr, "Logistic Regression", "Test")

    results['logistic_regression'] = {
        'model': 'LogisticRegression',
        'features': 'TF-IDF',
        'hyperparameters': {'C': 1.0, 'max_iter': args.max_iter, 'solver': args.solver, 'n_jobs': args.n_jobs},
        'training_time': train_time,
        'validation_metrics': lr_val_metrics,
        'test_metrics': lr_test_metrics
    }

    # Clean up TF-IDF features
    print("\n  Cleaning up TF-IDF features from memory...")
    del X_train, X_val, X_test
    gc.collect()
    print_memory_usage()

    # ========================================
    # Model comparison
    # ========================================
    print("\n" + "-"*80)
    print("[4/5] Model Comparison")
    print("-"*80)

    print("\n  VALIDATION SET RESULTS:")
    print(f"  {'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"  {'-'*73}")
    print(f"  {'Naive Bayes':<25} {nb_val_metrics['accuracy']:.4f}       {nb_val_metrics['precision']:.4f}       {nb_val_metrics['recall']:.4f}       {nb_val_metrics['f1']:.4f}")
    print(f"  {'Logistic Regression':<25} {lr_val_metrics['accuracy']:.4f}       {lr_val_metrics['precision']:.4f}       {lr_val_metrics['recall']:.4f}       {lr_val_metrics['f1']:.4f}")

    print("\n  TEST SET RESULTS:")
    print(f"  {'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"  {'-'*73}")
    print(f"  {'Naive Bayes':<25} {nb_test_metrics['accuracy']:.4f}       {nb_test_metrics['precision']:.4f}       {nb_test_metrics['recall']:.4f}       {nb_test_metrics['f1']:.4f}")
    print(f"  {'Logistic Regression':<25} {lr_test_metrics['accuracy']:.4f}       {lr_test_metrics['precision']:.4f}       {lr_test_metrics['recall']:.4f}       {lr_test_metrics['f1']:.4f}")

    # Determine best model
    if lr_val_metrics['accuracy'] > nb_val_metrics['accuracy']:
        best_model_name = 'Logistic Regression'
        best_model = lr_model
        best_features = 'tfidf'
    else:
        best_model_name = 'Naive Bayes'
        best_model = nb_model
        best_features = 'bow'

    print(f"\n  Best Model: {best_model_name} (Validation Accuracy: {max(nb_val_metrics['accuracy'], lr_val_metrics['accuracy']):.4f})")

    # ========================================
    # Save models and results
    # ========================================
    print("\n" + "-"*80)
    print("[5/5] Saving models and results")
    print("-"*80)

    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save Naive Bayes
    with open(models_dir / 'naive_bayes_model.pkl', 'wb') as f:
        pickle.dump(nb_model, f)
    print(f"  ✓ Saved Naive Bayes model")

    # Save Logistic Regression
    with open(models_dir / 'logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    print(f"  ✓ Saved Logistic Regression model")

    # Save best model separately with vectorizer
    print(f"\n  Selecting vectorizer for best model ({best_features})...")
    if best_features == 'tfidf':
        vectorizer = tfidf_vectorizer
    else:
        vectorizer = bow_vectorizer

    with open(models_dir / 'best_baseline_model.pkl', 'wb') as f:
        pickle.dump({
            'model': best_model,
            'model_name': best_model_name,
            'features': best_features,
            'vectorizer': vectorizer
        }, f)
    print(f"  ✓ Saved best model: {best_model_name}")

    # Save results
    with open(models_dir / 'baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved results to baseline_results.json")

    print(f"\n  All files saved to: {models_dir}")

    # ========================================
    # Final Summary
    # ========================================
    print("\n" + "="*80)
    print("BASELINE MODEL TRAINING COMPLETE")
    print("="*80)

    print(f"\n✓ Trained 2 baseline models (Naive Bayes, Logistic Regression)")
    print(f"✓ Best validation accuracy: {max(nb_val_metrics['accuracy'], lr_val_metrics['accuracy']):.4f} ({max(nb_val_metrics['accuracy'], lr_val_metrics['accuracy'])*100:.2f}%)")
    print(f"✓ Best test accuracy: {max(nb_test_metrics['accuracy'], lr_test_metrics['accuracy']):.4f} ({max(nb_test_metrics['accuracy'], lr_test_metrics['accuracy'])*100:.2f}%)")
    print(f"✓ Target benchmark: ≥88% (Maas et al., 2011)")

    if max(lr_test_metrics['accuracy'], nb_test_metrics['accuracy']) >= 0.88:
        print(f"\nSUCCESS! Achieved benchmark performance!")
    else:
        print(f"\nBelow benchmark. Consider advanced models or hyperparameter tuning.")

    print(f"\nFinal memory usage:")
    print_memory_usage()

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
