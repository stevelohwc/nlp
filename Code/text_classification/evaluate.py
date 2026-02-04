"""
Comprehensive evaluation and comparison across all 4 trained models.
Generates confusion-matrix heatmaps, ROC curves, a model-comparison bar
chart, and a summary evaluation_report.json.

Run AFTER both train_baseline_models.py and train_advanced_models.py.
"""

import pickle
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')                       # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODELS_DIR    = Path(__file__).parent.parent / 'models'
_PROCESSED_DIR = Path(__file__).parent.parent / 'datasets' / 'processed'
_PLOTS_DIR     = Path(__file__).resolve().parent / 'evaluation_plots'


def _load_pkl(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


# ===========================================================================
# Load everything
# ===========================================================================
def load_data():
    """Return models dict and test labels / features."""
    # --- Feature sets ---
    tfidf = _load_pkl(_PROCESSED_DIR / 'tfidf_features.pkl')
    bow   = _load_pkl(_PROCESSED_DIR / 'bow_features.pkl')

    X_test_tfidf = tfidf['X_test']
    X_test_bow   = bow['X_test']
    y_test       = np.asarray(tfidf['y_test'])   # same labels in both

    # --- Models ---
    nb  = _load_pkl(_MODELS_DIR / 'naive_bayes_model.pkl')
    lr  = _load_pkl(_MODELS_DIR / 'logistic_regression_model.pkl')
    svm = _load_pkl(_MODELS_DIR / 'svm_model.pkl')
    rf  = _load_pkl(_MODELS_DIR / 'random_forest_model.pkl')

    models = {
        'Naive Bayes':         {'model': nb,  'X_test': X_test_bow,   'type': 'proba'},
        'Logistic Regression': {'model': lr,  'X_test': X_test_tfidf, 'type': 'proba'},
        'LinearSVC':           {'model': svm, 'X_test': X_test_tfidf, 'type': 'decision'},
        'Random Forest':       {'model': rf,  'X_test': X_test_tfidf, 'type': 'proba'},
    }

    return models, y_test


# ===========================================================================
# Prediction helpers
# ===========================================================================
def get_predictions(models, y_test):
    """Return dict of {name: (y_pred, y_scores)}."""
    out = {}
    for name, info in models.items():
        model  = info['model']
        X_test = info['X_test']
        y_pred = model.predict(X_test)

        if info['type'] == 'proba':
            y_scores = model.predict_proba(X_test)[:, 1]
        else:                                           # LinearSVC
            y_scores = model.decision_function(X_test)  # works directly with roc_curve

        out[name] = (y_pred, y_scores)
    return out


# ===========================================================================
# Plot 1 — Confusion matrices (2 × 2 subplot grid)
# ===========================================================================
def plot_confusion_matrices(predictions, y_test, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Confusion Matrices — All Models", fontsize=16, fontweight='bold', y=1.02)

    names = list(predictions.keys())
    for ax, name in zip(axes.flat, names):
        y_pred = predictions[name][0]
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'],
                    cbar=False, linewidths=0.5)
        acc = accuracy_score(y_test, y_pred)
        ax.set_title(f"{name}  (acc {acc * 100:.2f}%)", fontsize=12, fontweight='bold')
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===========================================================================
# Plot 2 — ROC curves (all 4 on one figure)
# ===========================================================================
def plot_roc_curves(predictions, y_test, out_path):
    fig, ax = plt.subplots(figsize=(9, 7))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for (name, (_, y_scores)), color in zip(predictions.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name}  (AUC = {roc_auc:.3f})")

    # Diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===========================================================================
# Plot 3 — Model comparison bar chart
# ===========================================================================
def plot_model_comparison(metrics_dict, out_path):
    """
    metrics_dict: {model_name: {'accuracy', 'precision', 'recall', 'f1'}}
    """
    names   = list(metrics_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    labels  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    x       = np.arange(len(metrics))
    width   = 0.18
    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (name, color) in enumerate(zip(names, colors)):
        values = [metrics_dict[name][m] for m in metrics]
        ax.bar(x + i * width, values, width, label=name, color=color, edgecolor='white')

    # Benchmark line at 88.89 %  (Maas et al.)
    ax.axhline(y=0.8889, color='black', linestyle='--', linewidth=1.5,
               label='Maas et al. benchmark (88.89%)')

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — IMDB Sentiment Classification", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim([0.70, 1.02])
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 80)
    print("MODEL EVALUATION — IMDB Sentiment Classification")
    print("=" * 80)

    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    print("\n[1/4] Loading models and test data …")
    models, y_test = load_data()
    print(f"  Test set size: {len(y_test):,}")

    # --- Predictions ---
    print("\n[2/4] Generating predictions …")
    predictions = get_predictions(models, y_test)

    # Compute full metrics for each model
    metrics_dict = {}
    report_models = {}
    for name, (y_pred, y_scores) in predictions.items():
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary')
        rec  = recall_score(y_test, y_pred, average='binary')
        f1   = f1_score(y_test, y_pred, average='binary')
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        metrics_dict[name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1)
        }
        report_models[name] = {
            'accuracy':  float(acc),
            'precision': float(prec),
            'recall':    float(rec),
            'f1':        float(f1),
            'auc':       float(roc_auc),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        print(f"    {name:<25} acc={acc:.4f}  prec={prec:.4f}  "
              f"rec={rec:.4f}  f1={f1:.4f}  auc={roc_auc:.3f}")

    # --- Plots ---
    print("\n[3/4] Generating plots …")
    plot_confusion_matrices(predictions, y_test, _PLOTS_DIR / 'confusion_matrices.png')
    plot_roc_curves(predictions, y_test, _PLOTS_DIR / 'roc_curves.png')
    plot_model_comparison(metrics_dict, _PLOTS_DIR / 'model_comparison.png')

    # --- Report JSON ---
    print("\n[4/4] Writing evaluation report …")
    best_model = max(report_models, key=lambda k: report_models[k]['accuracy'])
    report = {
        'benchmark': {
            'source': 'Maas et al., 2011',
            'accuracy': 0.8889
        },
        'models': report_models,
        'best_model': {
            'name': best_model,
            'metrics': report_models[best_model]
        },
        'plots': [
            'evaluation_plots/confusion_matrices.png',
            'evaluation_plots/roc_curves.png',
            'evaluation_plots/model_comparison.png'
        ]
    }
    report_path = Path(__file__).resolve().parent / 'evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved {report_path}")

    # --- Summary ---
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\n  Best model:  {best_model}  "
          f"(accuracy = {report_models[best_model]['accuracy'] * 100:.2f}%)")
    print(f"  Benchmark:   Maas et al. 88.89%")
    if report_models[best_model]['accuracy'] >= 0.8889:
        print(f"  Result:      MEETS benchmark")
    else:
        print(f"  Result:      Below benchmark")
    print(f"\n  3 plots saved to: {_PLOTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
