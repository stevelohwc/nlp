"""Evaluate all trained fake-news models and generate visual reports.

CLI contract:
    python evaluate.py --models-dir models --processed-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def resolve_path(path_value: str, project_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (project_dir / path).resolve()


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate fake-news models")
    parser.add_argument("--models-dir", default=str(here / "models"))
    parser.add_argument("--processed-dir", default=str(here / "data" / "processed"))
    parser.add_argument("--plots-dir", default=str(here / "evaluation_plots"))
    parser.add_argument("--reports-dir", default=str(here / "reports"))
    return parser.parse_args()


def load_models(models_dir: Path) -> dict:
    model_files = {
        "Naive Bayes": "naive_bayes_model.pkl",
        "Logistic Regression": "logistic_regression_model.pkl",
        "LinearSVC": "linear_svc_model.pkl",
        "Random Forest": "random_forest_model.pkl",
    }

    loaded = {}
    for name, filename in model_files.items():
        path = models_dir / filename
        if not path.exists():
            continue
        with open(path, "rb") as fh:
            loaded[name] = pickle.load(fh)
    return loaded


def get_probability_like_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))
    pred = model.predict(X)
    return pred.astype(float)


def compute_metrics(model, X, y_true) -> dict:
    y_pred = model.predict(X)
    y_score = get_probability_like_score(model, X)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    fpr, tpr, _ = roc_curve(y_true, y_score)
    metrics["roc_auc"] = float(auc(fpr, tpr))
    metrics["fpr"] = fpr
    metrics["tpr"] = tpr
    return metrics


def plot_confusion_matrices(metrics_by_model: dict, out_path: Path) -> None:
    names = list(metrics_by_model.keys())
    count = len(names)
    cols = 2
    rows = math.ceil(count / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(names):
        ax = axes[i]
        cm = np.array(metrics_by_model[name]["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Fake", "Real"],
            yticklabels=["Fake", "Real"],
            ax=ax,
        )
        ax.set_title(f"{name} (F1={metrics_by_model[name]['f1']:.4f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    for j in range(len(names), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(metrics_by_model: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, metrics in metrics_by_model.items():
        ax.plot(metrics["fpr"], metrics["tpr"], label=f"{name} (AUC={metrics['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Fake News Models")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(metrics_by_model: dict, out_path: Path) -> None:
    names = list(metrics_by_model.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]

    values = np.array([[metrics_by_model[name][m] for m in metrics] for name in names])
    x = np.arange(len(metrics))
    width = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, name in enumerate(names):
        ax.bar(x + idx * width, values[idx], width=width, label=name)

    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison - Fake News")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    models_dir = resolve_path(args.models_dir, project_dir)
    processed_dir = resolve_path(args.processed_dir, project_dir)
    plots_dir = resolve_path(args.plots_dir, project_dir)
    reports_dir = resolve_path(args.reports_dir, project_dir)

    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(processed_dir / "tfidf_features.pkl", "rb") as fh:
        tfidf_data = pickle.load(fh)

    X_test = tfidf_data["X_test"]
    y_test = np.asarray(tfidf_data["y_test"])

    models = load_models(models_dir)
    if not models:
        raise FileNotFoundError(
            f"No model files found in {models_dir}. Run training scripts first."
        )

    metrics_by_model = {}
    for name, model in models.items():
        metrics_by_model[name] = compute_metrics(model, X_test, y_test)

    best_model_name = max(
        metrics_by_model,
        key=lambda name: (
            metrics_by_model[name]["f1"],
            metrics_by_model[name]["recall"],
            metrics_by_model[name]["accuracy"],
        ),
    )

    plot_confusion_matrices(metrics_by_model, plots_dir / "confusion_matrices.png")
    plot_roc_curves(metrics_by_model, plots_dir / "roc_curves.png")
    plot_model_comparison(metrics_by_model, plots_dir / "model_comparison.png")

    report = {
        "selection_metric": "f1_then_recall_then_accuracy",
        "best_model": {
            "name": best_model_name,
            "metrics": {
                k: v
                for k, v in metrics_by_model[best_model_name].items()
                if k not in {"fpr", "tpr"}
            },
        },
        "models": {
            model_name: {
                key: value
                for key, value in model_metrics.items()
                if key not in {"fpr", "tpr"}
            }
            for model_name, model_metrics in metrics_by_model.items()
        },
        "plots": {
            "confusion_matrices": str((plots_dir / "confusion_matrices.png").name),
            "roc_curves": str((plots_dir / "roc_curves.png").name),
            "model_comparison": str((plots_dir / "model_comparison.png").name),
        },
    }

    with open(reports_dir / "evaluation_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Best model: {best_model_name}")
    print(f"Report: {reports_dir / 'evaluation_report.json'}")
    print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()
