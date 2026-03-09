"""News dataset loading utilities for the standalone fake-news project.

This module enforces project isolation:
- Only fake-news CSV files (fake.csv, true.csv) are valid inputs.
- IMDB sentiment dataset paths are explicitly blocked.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_KAGGLE_DATASETS = [
    "clmentbisaillon/fake-and-real-news-dataset",
    "fake-news-detection",
]
IMDB_FILENAMES = {"imdb_train.csv", "imdb_test.csv"}


class DatasetIsolationError(RuntimeError):
    """Raised when a non-news dataset path is used by mistake."""


class DatasetValidationError(RuntimeError):
    """Raised when required dataset files are missing/corrupt."""


@dataclass(frozen=True)
class DatasetPaths:
    data_dir: Path
    fake_csv: Path
    true_csv: Path


class NewsDatasetLoader:
    """Load and validate fake/real news data."""

    def __init__(self, data_dir: str | Path):
        data_dir_path = Path(data_dir).expanduser().resolve()
        self._assert_news_only_path(data_dir_path)
        self.data_dir = data_dir_path
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _assert_news_only_path(data_dir: Path) -> None:
        raw = str(data_dir).replace("\\", "/")
        if "/Code/datasets" in raw:
            raise DatasetIsolationError(
                "Blocked path inside Code/datasets. This project is standalone and must "
                "use Code/News-Classification-NLP/data/raw."
            )

        for filename in IMDB_FILENAMES:
            if (data_dir / filename).exists():
                raise DatasetIsolationError(
                    f"Detected IMDB file {filename!r} in dataset directory. "
                    "Use fake.csv and true.csv only."
                )

    def _canonicalize_dataset_files(self) -> DatasetPaths:
        fake_candidate = None
        true_candidate = None

        for csv_path in self.data_dir.rglob("*.csv"):
            name = csv_path.name.lower()
            if name == "fake.csv" and fake_candidate is None:
                fake_candidate = csv_path
            elif name == "true.csv" and true_candidate is None:
                true_candidate = csv_path

        if fake_candidate is None or true_candidate is None:
            raise DatasetValidationError(
                "Could not find both fake.csv and true.csv in data/raw. "
                "Run with --download-if-missing or add files manually."
            )

        fake_csv = self.data_dir / "fake.csv"
        true_csv = self.data_dir / "true.csv"

        if fake_candidate.resolve() != fake_csv:
            fake_csv.write_bytes(fake_candidate.read_bytes())
        if true_candidate.resolve() != true_csv:
            true_csv.write_bytes(true_candidate.read_bytes())

        return DatasetPaths(data_dir=self.data_dir, fake_csv=fake_csv, true_csv=true_csv)

    def _download_from_kaggle(self, datasets: Iterable[str]) -> None:
        errors: list[str] = []
        command_prefixes = self._kaggle_command_prefixes()

        for dataset in datasets:
            success = False
            for prefix in command_prefixes:
                cmd = [
                    *prefix,
                    "datasets",
                    "download",
                    "-d",
                    dataset,
                    "-p",
                    str(self.data_dir),
                    "--force",
                    "--unzip",
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    success = True
                    break

                error_text = (result.stderr or result.stdout or "Unknown kaggle error").strip()
                errors.append(
                    f"{dataset} using `{' '.join(prefix)}`: {error_text}"
                )

            if success:
                return

        joined_errors = "\n".join(errors)
        raise DatasetValidationError(
            "Auto-download from Kaggle failed for all dataset candidates. "
            "Set up Kaggle credentials (kaggle.json) or copy fake.csv and true.csv into "
            f"{self.data_dir}.\n\nKaggle errors:\n{joined_errors}"
        )

    @staticmethod
    def _kaggle_command_prefixes() -> list[list[str]]:
        prefixes: list[list[str]] = []

        venv_kaggle = Path(sys.executable).with_name("kaggle")
        if venv_kaggle.exists():
            prefixes.append([str(venv_kaggle)])

        path_kaggle = shutil.which("kaggle")
        if path_kaggle:
            candidate = [path_kaggle]
            if candidate not in prefixes:
                prefixes.append(candidate)

        prefixes.append([sys.executable, "-m", "kaggle.cli"])
        prefixes.append([sys.executable, "-m", "kaggle"])
        return prefixes

    def ensure_dataset(
        self,
        download_if_missing: bool = True,
        kaggle_dataset: str | None = None,
    ) -> DatasetPaths:
        """Ensure fake.csv/true.csv exist and pass basic validation."""
        try:
            dataset_paths = self._canonicalize_dataset_files()
        except DatasetValidationError:
            if not download_if_missing:
                raise

            candidates = [kaggle_dataset] if kaggle_dataset else list(DEFAULT_KAGGLE_DATASETS)
            self._download_from_kaggle(candidates)
            dataset_paths = self._canonicalize_dataset_files()

        self._validate_csv(dataset_paths.fake_csv, required_columns={"text"})
        self._validate_csv(dataset_paths.true_csv, required_columns={"text"})
        return dataset_paths

    @staticmethod
    def _read_csv(csv_path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(csv_path)
        except UnicodeDecodeError:
            return pd.read_csv(csv_path, encoding="latin-1")

    def _validate_csv(self, csv_path: Path, required_columns: set[str]) -> None:
        if not csv_path.exists():
            raise DatasetValidationError(f"Missing dataset file: {csv_path}")

        df = self._read_csv(csv_path)
        missing = required_columns - set(df.columns)
        if missing:
            raise DatasetValidationError(
                f"Dataset file {csv_path.name} missing required columns: {sorted(missing)}"
            )
        if df.empty:
            raise DatasetValidationError(f"Dataset file {csv_path.name} is empty.")

    def load_news_dataframe(self, dataset_paths: DatasetPaths) -> pd.DataFrame:
        """Return a unified dataframe with columns: title, text, label, input_text."""
        fake_df = self._read_csv(dataset_paths.fake_csv).copy()
        true_df = self._read_csv(dataset_paths.true_csv).copy()

        for frame in (fake_df, true_df):
            if "title" not in frame.columns:
                frame["title"] = ""
            frame["title"] = frame["title"].fillna("").astype(str)
            frame["text"] = frame["text"].fillna("").astype(str)

        fake_df["label"] = 0
        true_df["label"] = 1

        merged = pd.concat([fake_df, true_df], ignore_index=True)
        merged = merged[["title", "text", "label"] + [c for c in merged.columns if c not in {"title", "text", "label"}]]
        merged["input_text"] = (merged["title"] + " " + merged["text"]).str.strip()
        merged = merged[merged["input_text"].str.len() > 0].reset_index(drop=True)

        if merged.empty:
            raise DatasetValidationError("No non-empty records found after combining title/text.")

        return merged


def resolve_path(path_value: str, project_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (project_dir / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and inspect fake-news dataset files")
    parser.add_argument("--data-dir", default="data/raw", help="Directory for fake.csv and true.csv")
    parser.add_argument(
        "--kaggle-dataset",
        default=None,
        help="Optional Kaggle dataset slug override (owner/dataset)",
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Auto-download from Kaggle if fake.csv/true.csv are missing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    data_dir = resolve_path(args.data_dir, project_dir)
    loader = NewsDatasetLoader(data_dir)
    dataset_paths = loader.ensure_dataset(
        download_if_missing=args.download_if_missing,
        kaggle_dataset=args.kaggle_dataset,
    )
    df = loader.load_news_dataframe(dataset_paths)

    print("=" * 72)
    print("NEWS DATASET READY")
    print("=" * 72)
    print(f"Data directory: {dataset_paths.data_dir}")
    print(f"fake.csv rows: {len(loader._read_csv(dataset_paths.fake_csv)):,}")
    print(f"true.csv rows: {len(loader._read_csv(dataset_paths.true_csv)):,}")
    print(f"Combined usable rows: {len(df):,}")
    label_counts = df["label"].value_counts().to_dict()
    print(f"Label distribution: {label_counts}")


if __name__ == "__main__":
    main()
