"""
Flask web application — Sentiment classification deployment.

Auto-installs Flask if missing (Werkzeug is already present).
Serves on 0.0.0.0:8000.

Run:
    python Code/text_classification/deploy.py
"""

import sys
import subprocess
import pickle
import re
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Auto-install Flask if missing
# ---------------------------------------------------------------------------
try:
    import flask
except ImportError:
    print("Flask not found — installing …")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'flask'])
    import flask

from flask import Flask, request, jsonify, render_template

# ---------------------------------------------------------------------------
# TextPreprocessor — duplicated here so that pickle can unpickle the
# preprocessor object (stored as __main__.TextPreprocessor).  This avoids
# importing run_preprocessing.py which pulls in the broken data_loader chain.
# ---------------------------------------------------------------------------
def _remove_html_tags(text):
    return re.sub(re.compile('<.*?>'), '', text)

def _remove_urls(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def _expand_contractions(text):
    for c, e in [("n't", " not"), ("'re", " are"), ("'s", " is"),
                 ("'d", " would"), ("'ll", " will"), ("'ve", " have"),
                 ("'m", " am")]:
        text = text.replace(c, e)
    return text

class TextPreprocessor:
    """Mirrors the class pickled by run_preprocessing.py."""
    def __init__(self, lowercase=True, remove_stopwords=False):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.stop_words = set()

    def clean_text(self, text):
        text = _remove_html_tags(text)
        text = _remove_urls(text)
        text = _expand_contractions(text)
        if self.lowercase:
            text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text

    def preprocess(self, text):
        text = self.clean_text(text)
        if self.remove_stopwords and self.stop_words:
            tokens = [w for w in text.split() if w not in self.stop_words]
            text = ' '.join(tokens)
        return text


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE        = Path(__file__).resolve().parent
_MODELS_DIR  = _HERE.parent / 'models'
_PROCESSED_DIR = _HERE.parent / 'datasets' / 'processed'

# ---------------------------------------------------------------------------
# App & global state
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Loaded once at startup
_model      = None
_model_name = None
_vectorizer = None
_preprocessor = None


def _load_pkl(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def _startup_load():
    """Load best model bundle + preprocessor into globals."""
    global _model, _model_name, _vectorizer, _preprocessor

    # Try best_model.pkl first (written by train_advanced_models.py),
    # fall back to best_baseline_model.pkl
    bundle_path = _MODELS_DIR / 'best_model.pkl'
    if not bundle_path.exists():
        bundle_path = _MODELS_DIR / 'best_baseline_model.pkl'

    bundle = _load_pkl(bundle_path)
    _model      = bundle['model']
    _model_name = bundle['model_name']
    _vectorizer = bundle['vectorizer']

    _preprocessor = _load_pkl(_PROCESSED_DIR / 'preprocessor.pkl')
    print(f"  Loaded model: {_model_name}  (from {bundle_path.name})")


# ---------------------------------------------------------------------------
# Confidence helper
# ---------------------------------------------------------------------------
def _get_confidence(model, X_vec):
    """Return confidence score in [0, 1] for the predicted class.

    - predict_proba  → max of the two class probabilities
    - decision_function (LinearSVC) → sigmoid of the raw score
    """
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_vec)[0]
        return float(np.max(probs))
    else:                                               # LinearSVC
        raw = model.decision_function(X_vec)[0]
        # Sigmoid
        return float(1.0 / (1.0 + np.exp(-raw)))


# ===========================================================================
# Routes
# ===========================================================================
@app.route('/')
def index():
    return render_template('index.html', model_name=_model_name)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Preprocess
    cleaned = _preprocessor.preprocess(text)

    # Vectorize
    X_vec = _vectorizer.transform([cleaned])

    # Predict
    label_idx  = _model.predict(X_vec)[0]
    label      = 'Positive' if label_idx == 1 else 'Negative'
    confidence = _get_confidence(_model, X_vec)

    return jsonify({
        'label':      label,
        'confidence': round(confidence, 4),
        'model_name': _model_name
    })


@app.route('/fetch_reviews', methods=['POST'])
def fetch_reviews():
    """Scrape user reviews from an IMDb reviews page URL.

    Expects JSON: { "url": "https://www.imdb.com/title/ttXXXXXXXX/reviews/…" }
    Returns JSON: { "reviews": [ {"summary", "rating", "text"}, … ] }
    """
    import requests as _req
    from bs4 import BeautifulSoup

    data = request.get_json(force=True)
    url  = (data.get('url') or '').strip()

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    # Basic validation — must be an IMDb title reviews page
    if 'imdb.com/title/' not in url:
        return jsonify({'error': 'URL must be an IMDb title page (imdb.com/title/…)'}), 400

    # Normalise: ensure the path ends at /reviews/
    # e.g. https://www.imdb.com/title/tt0371746/reviews/?ref_=…
    # Strip query string for the fetch, keep /reviews/
    from urllib.parse import urlparse, urlencode
    parsed = urlparse(url)
    path   = parsed.path.rstrip('/')
    if not path.endswith('/reviews'):
        # Try appending /reviews
        path = path + '/reviews'
    fetch_url = f"https://www.imdb.com{path}/"

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/131.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    try:
        resp = _req.get(fetch_url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        return jsonify({'error': f'Failed to fetch page: {exc}'}), 502

    soup = BeautifulSoup(resp.text, 'lxml')
    cards = soup.find_all(attrs={'data-testid': 'review-card-parent'})

    if not cards:
        return jsonify({'error': 'No reviews found on this page. '
                                 'Make sure the URL points to a reviews page.'}), 404

    reviews = []
    for card in cards:
        # Summary / title
        summary_el = card.find(attrs={'data-testid': 'review-summary'})
        summary = (summary_el.find('h3').get_text(strip=True)
                   if summary_el and summary_el.find('h3') else '')

        # Rating (e.g. "7/10")
        rating_el = card.find(class_='review-rating')
        rating = rating_el.get_text(strip=True) if rating_el else None

        # Body text (only present for longer reviews)
        overflow_el = card.find(attrs={'data-testid': 'review-overflow'})
        if overflow_el:
            content_div = overflow_el.find('div', recursive=False)
            body = (content_div.get_text(strip=True)
                    if content_div else overflow_el.get_text(strip=True))
        else:
            body = None

        reviews.append({
            'summary': summary,
            'rating':  rating,
            'text':    body if body else summary   # short reviews: summary IS the text
        })

    return jsonify({'reviews': reviews})


@app.route('/examples')
def examples():
    return jsonify({
        'positive': (
            "This movie was absolutely fantastic! The acting was superb, "
            "the storyline was compelling and kept me on the edge of my seat "
            "the entire time. I would highly recommend it to everyone."
        ),
        'negative': (
            "Terrible film. The plot made no sense, the actors were wooden, "
            "and the dialogue was painfully awkward. I nearly fell asleep "
            "halfway through. A complete waste of time and money."
        )
    })


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == '__main__':
    print("=" * 60)
    print(" Sentiment Classification — Web Deployment")
    print("=" * 60)
    _startup_load()
    print(f"  Starting server on http://0.0.0.0:8000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8000, debug=False)
