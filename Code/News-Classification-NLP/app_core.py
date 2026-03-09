"""Core services for the news-classification Streamlit app.

This module intentionally contains non-UI logic only.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import requests
import sklearn
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

try:
    import trafilatura
except Exception:  # pragma: no cover - optional dependency
    trafilatura = None


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
SAMPLE_FILE = PROJECT_DIR / "sample.txt"
MODEL_BUNDLE_PATH = MODELS_DIR / "best_model_bundle.pkl"

MALAYSIA_SAMPLE_URLS = [
    (
        "Malay Mail (sample 1)",
        "https://www.malaymail.com/news/singapore/2025/09/10/yes-i-am-muslimchineseindian-but-im-also-singaporean-lee-hsien-loong-says-national-identity-matters-but-may-not-be-most-important/190617",
    ),
    (
        "Malay Mail (sample 2)",
        "https://www.malaymail.com/news/world/2025/09/11/nepal-gen-z-backs-ex-chief-justice-sushila-karki-to-lead-after-pm-ousted-in-deadly-protests/190762",
    ),
    (
        "Malay Mail (sample 3)",
        "https://www.malaymail.com/news/world/2025/09/11/not-a-friendly-place-what-life-looks-like-for-thaksin-in-bangkoks-notorious-klong-prem-prison/190768",
    ),
]

SECTION_PATH_PARTS = {
    "news",
    "nation",
    "world",
    "business",
    "money",
    "opinion",
    "lifestyle",
    "life",
    "sports",
    "entertainment",
    "tech",
    "technology",
    "travel",
    "property",
    "metro",
    "focus",
    "asean",
    "malaysia",
    "singapore",
    "home",
    "latest",
    "category",
}

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_logger() -> logging.Logger:
    logger = logging.getLogger("news_classification_app")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(REPORTS_DIR / "app_runtime.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


LOGGER = get_logger()


def parse_major_minor(version: str) -> str:
    pieces = version.split(".")
    return ".".join(pieces[:2])


@st.cache_data
def read_samples() -> dict[str, str]:
    samples = {
        "real": "French lawmakers voted to oust Prime Minister François Bayrou Monday, plunging the country into a new political crisis.",
        "fake": "BREAKING: Government secretly installing mind control chips in vaccines activated by 5G towers.",
    }

    if not SAMPLE_FILE.exists():
        return samples

    raw = SAMPLE_FILE.read_text(encoding="utf-8", errors="ignore")
    real_marker = "REAL SAMPLE:"
    fake_marker = "FAKE SAMPLE:"
    links_marker = "REAL LINKS"

    if real_marker in raw and fake_marker in raw:
        real_block = raw.split(real_marker, 1)[1].split(fake_marker, 1)[0].strip()
        fake_block = raw.split(fake_marker, 1)[1]
        if links_marker in fake_block:
            fake_block = fake_block.split(links_marker, 1)[0]
        fake_block = fake_block.strip()

        if real_block:
            samples["real"] = real_block
        if fake_block:
            samples["fake"] = fake_block

    return samples


@st.cache_resource
def load_model_bundle() -> dict:
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            "Missing models/best_model_bundle.pkl. "
            "Run preprocessing + training scripts first."
        )

    bundle = joblib.load(MODEL_BUNDLE_PATH)

    required_keys = {
        "model",
        "vectorizer",
        "preprocessor",
        "label_map",
        "metrics",
        "training_config",
        "library_versions",
    }
    missing = required_keys - set(bundle.keys())
    if missing:
        raise RuntimeError(f"Model bundle missing keys: {sorted(missing)}")

    sklearn_bundle = bundle["library_versions"].get("scikit_learn")
    if sklearn_bundle:
        current = parse_major_minor(sklearn.__version__)
        expected = parse_major_minor(str(sklearn_bundle))
        if current != expected:
            raise RuntimeError(
                "scikit-learn version mismatch. "
                f"Bundle expects {sklearn_bundle}, current env is {sklearn.__version__}."
            )

    return bundle


@st.cache_data
def load_holdout_features() -> tuple:
    features_path = DATA_PROCESSED_DIR / "tfidf_features.pkl"
    if not features_path.exists():
        raise FileNotFoundError(
            "Missing data/processed/tfidf_features.pkl. "
            "Run run_preprocessing.py first."
        )

    data = joblib.load(features_path)
    return data["X_test"], np.asarray(data["y_test"])


def model_confidence(model, X_vec, prediction: int) -> tuple[float, float | None]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_vec)[0]
        confidence = float(proba[prediction])
        real_score = float(proba[1])
        return confidence, real_score

    if hasattr(model, "decision_function"):
        raw = float(model.decision_function(X_vec)[0])
        real_score = float(1.0 / (1.0 + np.exp(-raw)))
        confidence = real_score if prediction == 1 else (1.0 - real_score)
        return confidence, real_score

    return 0.0, None


def validate_text_input(text: str, min_chars: int = 30, max_chars: int = 20000) -> tuple[bool, str]:
    clean = text.strip()
    if not clean:
        return False, "Please enter article text before running prediction."
    if len(clean) < min_chars:
        return False, f"Article text is too short. Minimum length is {min_chars} characters."
    if len(clean) > max_chars:
        return False, f"Article text is too long. Maximum length is {max_chars} characters."
    return True, ""


def predict_article(bundle: dict, text: str, threshold: float = 0.5) -> dict:
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    preprocessor = bundle["preprocessor"]

    cleaned = preprocessor.preprocess(text)
    if not cleaned:
        raise ValueError("Text becomes empty after preprocessing. Please provide more content.")

    X_vec = vectorizer.transform([cleaned])
    pred = int(model.predict(X_vec)[0])
    confidence, real_score = model_confidence(model, X_vec, pred)

    if real_score is not None:
        pred = 1 if real_score >= threshold else 0
        confidence = real_score if pred == 1 else (1.0 - real_score)

    label_map = bundle["label_map"]
    label = label_map.get(pred, "Real News" if pred == 1 else "Fake News")

    return {
        "label": label,
        "prediction": pred,
        "confidence": confidence,
        "real_score": real_score,
    }


def is_valid_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def looks_like_section_or_home_url(url: str) -> tuple[bool, str]:
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        return True, "This looks like a homepage URL."

    parts = [segment.lower() for segment in path.split("/") if segment]
    if len(parts) <= 2 and all(part in SECTION_PATH_PARTS for part in parts):
        return True, (
            "This looks like a section page URL, not one specific article "
            f"(`/{'/'.join(parts)}`)."
        )

    if len(parts) == 1 and parts[0] in SECTION_PATH_PARTS:
        return True, "This looks like a category/section URL."

    return False, ""


def validate_extracted_article_text(text: str) -> tuple[bool, str]:
    clean = text.strip()
    if len(clean) < 300:
        return False, "Extracted text is too short to be a full article."

    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    if not lines:
        return False, "No usable text was extracted from the page."

    sentence_count = len(re.findall(r"[.!?](?:\s|$)", clean))
    word_counts = [len(re.findall(r"\b\w+\b", line)) for line in lines]
    headline_like_lines = [
        line
        for line, count in zip(lines, word_counts)
        if 4 <= count <= 18 and not re.search(r"[.!?]$", line)
    ]

    if len(lines) >= 6:
        headline_ratio = len(headline_like_lines) / len(lines)
        if headline_ratio >= 0.55 and sentence_count < max(4, len(lines) // 4):
            return (
                False,
                "The extracted text looks like a headline list from a section page, "
                "not a single news article.",
            )

    return True, ""


def extract_text_from_url(url: str, strategy: str = "trafilatura") -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        )
    }

    if strategy == "trafilatura" and trafilatura is not None:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(
                downloaded,
                url=url,
                include_comments=False,
                include_tables=False,
                include_links=False,
                include_images=False,
                favor_precision=True,
            )
            if extracted and extracted.strip():
                return extracted.strip()

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    article_paragraphs: list[str] = []
    for node in soup.find_all("article"):
        article_paragraphs.extend(p.get_text(" ", strip=True) for p in node.find_all("p"))

    main_paragraphs: list[str] = []
    for node in soup.find_all("main"):
        main_paragraphs.extend(p.get_text(" ", strip=True) for p in node.find_all("p"))

    paragraphs = article_paragraphs or main_paragraphs
    if not paragraphs:
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]

    text = "\n".join(p for p in paragraphs if p)
    return text.strip()


def benchmark_evaluation(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
