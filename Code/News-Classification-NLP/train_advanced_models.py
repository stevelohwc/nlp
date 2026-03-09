"""Train advanced fake-news models and publish the best model bundle.

CLI contract:
    python train_advanced_models.py --processed-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC


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
    parser = argparse.ArgumentParser(description="Train advanced models (LinearSVC, RandomForest)")
    parser.add_argument("--processed-dir", default=str(here / "data" / "processed"))
    parser.add_argument("--models-dir", default=str(here / "models"))
    parser.add_argument("--reports-dir", default=str(here / "reports"))
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--rf-n-iter", type=int, default=4)
    parser.add_argument(
        "--rf-model-n-jobs",
        type=int,
        default=1,
        help="RandomForestClassifier n_jobs. Keep 1 to avoid nested parallel slowdown.",
    )
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


def run_grid_search(model, param_grid, X_train, y_train, cv, n_jobs: int):
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
    final_model.fit(X_train, y_train)

    cv_metrics = {
        "f1": float(search.cv_results_["mean_test_f1"][best_idx]),
        "recall": float(search.cv_results_["mean_test_recall"][best_idx]),
        "accuracy": float(search.cv_results_["mean_test_accuracy"][best_idx]),
    }
    return final_model, best_params, cv_metrics


def run_random_search(model, param_distributions, X_train, y_train, cv, n_jobs: int, n_iter: int, random_state: int):
    scoring = {"f1": "f1", "recall": "recall", "accuracy": "accuracy"}
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        refit=False,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best_idx = select_best_index(search.cv_results_)
    best_params = search.cv_results_["params"][best_idx]

    model_cls = model.__class__
    final_model = model_cls(**{**model.get_params(), **best_params})
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

    with open(processed_dir / "preprocessor.pkl", "rb") as fh:
        preprocessor = pickle.load(fh)

    X_train = tfidf_data["X_train"]
    X_val = tfidf_data["X_val"]
    X_test = tfidf_data["X_test"]
    y_train = np.asarray(tfidf_data["y_train"])
    y_val = np.asarray(tfidf_data["y_val"])
    y_test = np.asarray(tfidf_data["y_test"])
    vectorizer = tfidf_data["vectorizer"]

    baseline_report_path = reports_dir / "baseline_results.json"
    if not baseline_report_path.exists():
        raise FileNotFoundError(
            f"Missing baseline report at {baseline_report_path}. "
            "Run train_baseline_models.py first."
        )

    with open(baseline_report_path, "r") as fh:
        baseline_report = json.load(fh)

    with open(models_dir / "naive_bayes_model.pkl", "rb") as fh:
        naive_bayes_model = pickle.load(fh)
    with open(models_dir / "logistic_regression_model.pkl", "rb") as fh:
        logistic_model = pickle.load(fh)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)

    print("=" * 80)
    print("ADVANCED TRAINING - FAKE NEWS")
    print("=" * 80)

    svm_model, svm_params, svm_cv = run_grid_search(
        model=LinearSVC(random_state=args.random_state),
        param_grid={
            "C": [0.1, 1.0],
            "class_weight": [None, "balanced"],
            "max_iter": [5000],
        },
        X_train=X_train,
        y_train=y_train,
        cv=cv,
        n_jobs=args.n_jobs,
    )
    svm_val = evaluate(y_val, svm_model.predict(X_val))
    svm_test = evaluate(y_test, svm_model.predict(X_test))

    with open(models_dir / "linear_svc_model.pkl", "wb") as fh:
        pickle.dump(svm_model, fh)

    rf_model, rf_params, rf_cv = run_random_search(
        model=RandomForestClassifier(random_state=args.random_state, n_jobs=args.rf_model_n_jobs),
        param_distributions={
            "n_estimators": [80, 120],
            "max_depth": [None, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt"],
            "class_weight": [None, "balanced"],
        },
        X_train=X_train,
        y_train=y_train,
        cv=cv,
        n_jobs=args.n_jobs,
        n_iter=args.rf_n_iter,
        random_state=args.random_state,
    )
    rf_val = evaluate(y_val, rf_model.predict(X_val))
    rf_test = evaluate(y_test, rf_model.predict(X_test))

    with open(models_dir / "random_forest_model.pkl", "wb") as fh:
        pickle.dump(rf_model, fh)

    advanced_results = {
        "linear_svc": {
            "model": "LinearSVC",
            "hyperparameters": svm_params,
            "cv_metrics": svm_cv,
            "validation_metrics": svm_val,
            "test_metrics": svm_test,
        },
        "random_forest": {
            "model": "RandomForestClassifier",
            "hyperparameters": rf_params,
            "cv_metrics": rf_cv,
            "validation_metrics": rf_val,
            "test_metrics": rf_test,
        },
    }

    combined = {
        "naive_bayes": baseline_report["models"]["naive_bayes"],
        "logistic_regression": baseline_report["models"]["logistic_regression"],
        "linear_svc": advanced_results["linear_svc"],
        "random_forest": advanced_results["random_forest"],
    }

    best_key = max(
        combined,
        key=lambda key: (
            combined[key]["validation_metrics"]["f1"],
            combined[key]["validation_metrics"]["recall"],
            combined[key]["validation_metrics"]["accuracy"],
        ),
    )

    model_lookup = {
        "naive_bayes": naive_bayes_model,
        "logistic_regression": logistic_model,
        "linear_svc": svm_model,
        "random_forest": rf_model,
    }

    best_bundle = {
        "model_name": best_key,
        "model": model_lookup[best_key],
        "vectorizer": vectorizer,
        "preprocessor": preprocessor,
        "label_map": {0: "Fake News", 1: "Real News"},
        "metrics": {
            "selection_metric": "f1_then_recall_then_accuracy",
            "all_models": combined,
            "best_model_validation_metrics": combined[best_key]["validation_metrics"],
            "best_model_test_metrics": combined[best_key]["test_metrics"],
        },
        "training_config": {
            "cv_folds": 5,
            "rf_n_iter": args.rf_n_iter,
            "random_state": args.random_state,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "library_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }

    with open(models_dir / "best_model_bundle.pkl", "wb") as fh:
        pickle.dump(best_bundle, fh)

    advanced_report = {
        "models": advanced_results,
        "overall_best_model": best_key,
        "selection_metric": "f1_then_recall_then_accuracy",
        "all_models": combined,
    }

    with open(reports_dir / "advanced_results.json", "w") as fh:
        json.dump(advanced_report, fh, indent=2)

    print(f"Saved advanced models to: {models_dir}")
    print(f"Saved best model bundle: {models_dir / 'best_model_bundle.pkl'}")
    print(f"Saved advanced report: {reports_dir / 'advanced_results.json'}")
    print(f"Overall best model: {best_key}")
    print("=" * 80)


if __name__ == "__main__":
    main()
