"""Preprocess fake-news data and build TF-IDF features.

CLI contract:
    python run_preprocessing.py --data-dir data/raw --out-dir data/processed
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from data_loader import NewsDatasetLoader
from text_preprocessing import NewsTextPreprocessor


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
    parser = argparse.ArgumentParser(description="Run preprocessing for fake-news project")
    parser.add_argument("--data-dir", default=str(here / "data" / "raw"), help="Input directory containing fake.csv and true.csv")
    parser.add_argument("--out-dir", default=str(here / "data" / "processed"), help="Output directory for processed artifacts")
    parser.add_argument(
        "--kaggle-dataset",
        default=None,
        help="Optional Kaggle dataset slug override (owner/dataset)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not auto-download from Kaggle when files are missing",
    )
    parser.add_argument("--max-features", type=int, default=10000, help="TF-IDF max feature size")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    data_dir = resolve_path(args.data_dir, project_dir)
    out_dir = resolve_path(args.out_dir, project_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PREPROCESSING - FAKE NEWS")
    print("=" * 80)

    loader = NewsDatasetLoader(data_dir)
    dataset_paths = loader.ensure_dataset(
        download_if_missing=not args.no_download,
        kaggle_dataset=args.kaggle_dataset,
    )

    df = loader.load_news_dataframe(dataset_paths)
    print(f"Loaded dataset rows: {len(df):,}")

    preprocessor = NewsTextPreprocessor()
    df["clean_text"] = df["input_text"].map(preprocessor.preprocess)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    print(f"Rows after cleaning: {len(df):,}")

    X = df["clean_text"].to_numpy()
    y = df["label"].to_numpy(dtype=np.int64)
    raw_text = df["input_text"].to_numpy()

    # 70/15/15 split with stratification.
    X_train, X_temp, y_train, y_temp, raw_train, raw_temp = train_test_split(
        X,
        y,
        raw_text,
        test_size=0.30,
        random_state=args.random_state,
        stratify=y,
    )
    X_val, X_test, y_val, y_test, raw_val, raw_test = train_test_split(
        X_temp,
        y_temp,
        raw_temp,
        test_size=0.50,
        random_state=args.random_state,
        stratify=y_temp,
    )

    print(f"Train size: {len(X_train):,}")
    print(f"Validation size: {len(X_val):,}")
    print(f"Test size: {len(X_test):,}")

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF feature matrix (train): {X_train_tfidf.shape}")

    with open(out_dir / "data_splits.pkl", "wb") as fh:
        pickle.dump(
            {
                "X_train_text": X_train,
                "X_val_text": X_val,
                "X_test_text": X_test,
                "X_train_raw": raw_train,
                "X_val_raw": raw_val,
                "X_test_raw": raw_test,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
            },
            fh,
        )

    with open(out_dir / "tfidf_features.pkl", "wb") as fh:
        pickle.dump(
            {
                "X_train": X_train_tfidf,
                "X_val": X_val_tfidf,
                "X_test": X_test_tfidf,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
                "vectorizer": vectorizer,
            },
            fh,
        )

    with open(out_dir / "vectorizer.pkl", "wb") as fh:
        pickle.dump(vectorizer, fh)

    with open(out_dir / "preprocessor.pkl", "wb") as fh:
        pickle.dump(preprocessor, fh)

    pd.DataFrame({"text": raw_train, "clean_text": X_train, "label": y_train}).to_csv(
        out_dir / "train_preprocessed.csv", index=False
    )
    pd.DataFrame({"text": raw_val, "clean_text": X_val, "label": y_val}).to_csv(
        out_dir / "val_preprocessed.csv", index=False
    )
    pd.DataFrame({"text": raw_test, "clean_text": X_test, "label": y_test}).to_csv(
        out_dir / "test_preprocessed.csv", index=False
    )

    print("Saved:")
    print(f"  - {out_dir / 'data_splits.pkl'}")
    print(f"  - {out_dir / 'tfidf_features.pkl'}")
    print(f"  - {out_dir / 'vectorizer.pkl'}")
    print(f"  - {out_dir / 'preprocessor.pkl'}")
    print(f"  - {out_dir / 'train_preprocessed.csv'}")
    print(f"  - {out_dir / 'val_preprocessed.csv'}")
    print(f"  - {out_dir / 'test_preprocessed.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
