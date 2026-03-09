"""Train baseline fake-news models with hyperparameter tuning.

CLI contract:
    python train_baseline_models.py --processed-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB


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
    parser = argparse.ArgumentParser(description="Train baseline models (NB, LR)")
    parser.add_argument("--processed-dir", default=str(here / "data" / "processed"))
    parser.add_argument("--models-dir", default=str(here / "models"))
    parser.add_argument("--reports-dir", default=str(here / "reports"))
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def select_best_index(cv_results: dict) -> int:
    f1_values = cv_results["mean_test_f1"]
    recall_values = cv_results["mean_test_recall"]
    accuracy_values = cv_results["mean_test_accuracy"]
    return max(
        range(len(f1_values)),
        key=lambda idx: (f1_values[idx], recall_values[idx], accuracy_values[idx]),
    )


def run_search(
    model,
    param_grid,
    X_train,
    y_train,
    cv,
    n_jobs: int,
    random_state: int,
):
    scoring = {"f1": "f1", "recall": "recall", "accuracy": "accuracy"}
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit=False,
        n_jobs=n_jobs,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_idx = select_best_index(search.cv_results_)
    best_params = search.cv_results_["params"][best_idx]

    model_cls = model.__class__
    final_model = model_cls(**{**model.get_params(), **best_params})

    if hasattr(final_model, "random_state"):
        final_model.set_params(random_state=random_state)

    final_model.fit(X_train, y_train)

    cv_metrics = {
        "f1": float(search.cv_results_["mean_test_f1"][best_idx]),
        "recall": float(search.cv_results_["mean_test_recall"][best_idx]),
        "accuracy": float(search.cv_results_["mean_test_accuracy"][best_idx]),
    }
    return final_model, best_params, cv_metrics


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    processed_dir = resolve_path(args.processed_dir, project_dir)
    models_dir = resolve_path(args.models_dir, project_dir)
    reports_dir = resolve_path(args.reports_dir, project_dir)

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(processed_dir / "tfidf_features.pkl", "rb") as fh:
        tfidf_data = pickle.load(fh)

    X_train = tfidf_data["X_train"]
    X_val = tfidf_data["X_val"]
    X_test = tfidf_data["X_test"]
    y_train = np.asarray(tfidf_data["y_train"])
    y_val = np.asarray(tfidf_data["y_val"])
    y_test = np.asarray(tfidf_data["y_test"])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)

    print("=" * 80)
    print("BASELINE TRAINING - FAKE NEWS")
    print("=" * 80)

    results = {}

    nb_model, nb_params, nb_cv = run_search(
        model=MultinomialNB(),
        param_grid={"alpha": [0.1, 0.3, 0.5, 1.0, 2.0]},
        X_train=X_train,
        y_train=y_train,
        cv=cv,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )
    nb_val = evaluate(y_val, nb_model.predict(X_val))
    nb_test = evaluate(y_test, nb_model.predict(X_test))

    with open(models_dir / "naive_bayes_model.pkl", "wb") as fh:
        pickle.dump(nb_model, fh)

    results["naive_bayes"] = {
        "model": "MultinomialNB",
        "hyperparameters": nb_params,
        "cv_metrics": nb_cv,
        "validation_metrics": nb_val,
        "test_metrics": nb_test,
    }

    lr_model, lr_params, lr_cv = run_search(
        model=LogisticRegression(),
        param_grid={
            "C": [0.1, 1.0, 3.0, 10.0],
            "solver": ["liblinear", "saga"],
            "class_weight": [None, "balanced"],
            "max_iter": [1000, 2000],
        },
        X_train=X_train,
        y_train=y_train,
        cv=cv,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )
    lr_val = evaluate(y_val, lr_model.predict(X_val))
    lr_test = evaluate(y_test, lr_model.predict(X_test))

    with open(models_dir / "logistic_regression_model.pkl", "wb") as fh:
        pickle.dump(lr_model, fh)

    results["logistic_regression"] = {
        "model": "LogisticRegression",
        "hyperparameters": lr_params,
        "cv_metrics": lr_cv,
        "validation_metrics": lr_val,
        "test_metrics": lr_test,
    }

    baseline_best = max(
        results.items(),
        key=lambda kv: (
            kv[1]["validation_metrics"]["f1"],
            kv[1]["validation_metrics"]["recall"],
            kv[1]["validation_metrics"]["accuracy"],
        ),
    )[0]

    summary = {
        "models": results,
        "best_baseline_model": baseline_best,
        "selection_metric": "f1_then_recall_then_accuracy",
    }

    with open(reports_dir / "baseline_results.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Saved models to: {models_dir}")
    print(f"Saved baseline report: {reports_dir / 'baseline_results.json'}")
    print(f"Best baseline model: {baseline_best}")
    print("=" * 80)


if __name__ == "__main__":
    main()
