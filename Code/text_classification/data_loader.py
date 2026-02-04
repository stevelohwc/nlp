"""
Data Loader for IMDB Movie Reviews Dataset

This module provides utilities for loading and accessing the IMDB sentiment analysis dataset.
The dataset contains 50,000 movie reviews (25,000 training, 25,000 test) labeled as positive or negative.

Citation:
Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011).
Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting
of the association for computational linguistics: Human language technologies (pp. 142-150).
"""

import os
import pickle
from pathlib import Path
from typing import Tuple, Dict, List
import pandas as pd
from datasets import load_dataset


class IMDBDataLoader:
    """Loader for IMDB movie reviews dataset"""

    def __init__(self, data_dir: str = None):
        """
        Initialize IMDB data loader

        Args:
            data_dir: Directory to store/load dataset (default: Code/datasets/raw)
        """
        if data_dir is None:
            # Get project root (assuming this file is in Code/text_classification/)
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "Code" / "datasets" / "raw"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = None
        self.train_data = None
        self.test_data = None

    def download_dataset(self, force_reload: bool = False) -> None:
        """
        Download IMDB dataset using Hugging Face datasets library

        Args:
            force_reload: If True, re-download even if cached
        """
        print("Downloading IMDB dataset from Hugging Face...")

        # Load dataset (will cache automatically)
        self.dataset = load_dataset("imdb", cache_dir=str(self.data_dir))

        print(f"Dataset downloaded successfully!")
        print(f"Train samples: {len(self.dataset['train'])}")
        print(f"Test samples: {len(self.dataset['test'])}")

        # Save to local files for easy access
        self._save_to_files()

    def _save_to_files(self) -> None:
        """Save dataset splits to CSV files"""
        train_file = self.data_dir / "imdb_train.csv"
        test_file = self.data_dir / "imdb_test.csv"

        # Convert to pandas DataFrames
        train_df = pd.DataFrame(self.dataset['train'])
        test_df = pd.DataFrame(self.dataset['test'])

        # Save to CSV
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        print(f"Saved train data to: {train_file}")
        print(f"Saved test data to: {test_file}")

    def load_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load IMDB dataset from local files or download if not available

        Returns:
            Tuple of (train_df, test_df) pandas DataFrames
        """
        train_file = self.data_dir / "imdb_train.csv"
        test_file = self.data_dir / "imdb_test.csv"

        # Check if files exist
        if not train_file.exists() or not test_file.exists():
            print("Dataset files not found. Downloading...")
            self.download_dataset()

        # Load from CSV
        self.train_data = pd.read_csv(train_file)
        self.test_data = pd.read_csv(test_file)

        print(f"Loaded {len(self.train_data)} training samples")
        print(f"Loaded {len(self.test_data)} test samples")

        return self.train_data, self.test_data

    def get_train_validation_split(self, validation_size: float = 0.2,
                                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split training data into train and validation sets

        Args:
            validation_size: Fraction of training data to use for validation
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_df, validation_df)
        """
        from sklearn.model_selection import train_test_split

        if self.train_data is None:
            self.load_dataset()

        train_df, val_df = train_test_split(
            self.train_data,
            test_size=validation_size,
            random_state=random_state,
            stratify=self.train_data['label']  # Ensure balanced splits
        )

        print(f"Train set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")

        return train_df, val_df

    def get_dataset_info(self) -> Dict:
        """
        Get comprehensive dataset statistics

        Returns:
            Dictionary containing dataset information
        """
        if self.train_data is None or self.test_data is None:
            self.load_dataset()

        info = {
            'train_size': len(self.train_data),
            'test_size': len(self.test_data),
            'total_size': len(self.train_data) + len(self.test_data),
            'num_classes': 2,
            'class_names': ['negative', 'positive'],
            'train_class_distribution': self.train_data['label'].value_counts().to_dict(),
            'test_class_distribution': self.test_data['label'].value_counts().to_dict(),
            'features': list(self.train_data.columns),
        }

        return info

    def get_sample_reviews(self, n: int = 5, sentiment: str = None) -> List[Dict]:
        """
        Get sample reviews from the dataset

        Args:
            n: Number of samples to retrieve
            sentiment: Filter by sentiment ('positive', 'negative', or None for both)

        Returns:
            List of dictionaries containing review text and label
        """
        if self.train_data is None:
            self.load_dataset()

        if sentiment == 'positive':
            samples = self.train_data[self.train_data['label'] == 1].head(n)
        elif sentiment == 'negative':
            samples = self.train_data[self.train_data['label'] == 0].head(n)
        else:
            samples = self.train_data.head(n)

        return samples.to_dict('records')


def main():
    """Example usage of IMDBDataLoader"""

    # Initialize loader
    loader = IMDBDataLoader()

    # Download and load dataset
    train_df, test_df = loader.load_dataset()

    # Get dataset info
    info = loader.get_dataset_info()
    print("\nDataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Get sample reviews
    print("\nSample Positive Reviews:")
    positive_samples = loader.get_sample_reviews(n=2, sentiment='positive')
    for i, sample in enumerate(positive_samples, 1):
        print(f"\n{i}. {sample['text'][:200]}...")

    print("\nSample Negative Reviews:")
    negative_samples = loader.get_sample_reviews(n=2, sentiment='negative')
    for i, sample in enumerate(negative_samples, 1):
        print(f"\n{i}. {sample['text'][:200]}...")


if __name__ == "__main__":
    main()
