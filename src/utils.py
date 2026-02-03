"""
Utility Functions
Common utility functions for the NLP project.
"""

import os
import json
from typing import List, Dict, Any
import pandas as pd


def load_text_file(filepath: str, encoding: str = 'utf-8') -> str:
    """
    Load text from a file.

    Args:
        filepath (str): Path to the text file.
        encoding (str): File encoding. Default is 'utf-8'.

    Returns:
        str: File contents as a string.
    """
    with open(filepath, 'r', encoding=encoding) as f:
        return f.read()


def save_text_file(text: str, filepath: str, encoding: str = 'utf-8'):
    """
    Save text to a file.

    Args:
        text (str): Text to save.
        filepath (str): Path to save the file.
        encoding (str): File encoding. Default is 'utf-8'.
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w', encoding=encoding) as f:
        f.write(text)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        Dict: Loaded JSON data.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
    """
    Save data to a JSON file.

    Args:
        data (Dict): Data to save.
        filepath (str): Path to save the file.
        indent (int): JSON indentation level.
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)


def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Args:
        filepath (str): Path to the CSV file.
        **kwargs: Additional arguments passed to pd.read_csv()

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(filepath, **kwargs)


def save_csv(df: pd.DataFrame, filepath: str, **kwargs):
    """
    Save a pandas DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        filepath (str): Path to save the file.
        **kwargs: Additional arguments passed to df.to_csv()
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    df.to_csv(filepath, **kwargs)


def calculate_statistics(texts: List[str]) -> Dict[str, Any]:
    """
    Calculate basic statistics for a list of texts.

    Args:
        texts (List[str]): List of text documents.

    Returns:
        Dict: Statistics including word count, character count, etc.
    """
    word_counts = [len(text.split()) for text in texts]
    char_counts = [len(text) for text in texts]
    
    return {
        'num_documents': len(texts),
        'total_words': sum(word_counts),
        'total_characters': sum(char_counts),
        'avg_words_per_doc': sum(word_counts) / len(texts) if texts else 0,
        'avg_chars_per_doc': sum(char_counts) / len(texts) if texts else 0,
        'min_words': min(word_counts) if word_counts else 0,
        'max_words': max(word_counts) if word_counts else 0,
    }


def print_statistics(stats: Dict[str, Any]):
    """
    Print statistics in a formatted way.

    Args:
        stats (Dict): Statistics dictionary.
    """
    print("=" * 60)
    print("Text Statistics")
    print("=" * 60)
    for key, value in stats.items():
        formatted_key = key.replace('_', ' ').title()
        if isinstance(value, float):
            print(f"{formatted_key}: {value:.2f}")
        else:
            print(f"{formatted_key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "This is a sample text.",
        "Another example with more words in it.",
        "Short text.",
        "This is a longer sample text with many more words to demonstrate the statistics calculation."
    ]
    
    stats = calculate_statistics(sample_texts)
    print_statistics(stats)
