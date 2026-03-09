"""Shared text preprocessing utilities for fake-news classification."""

from __future__ import annotations

import re
from dataclasses import dataclass

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


for resource in ("corpora/stopwords",):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1], quiet=True)


@dataclass
class NewsTextPreprocessor:
    lowercase: bool = True
    remove_stopwords: bool = True
    min_word_length: int = 3

    def __post_init__(self) -> None:
        self._stopwords = set(stopwords.words("english")) if self.remove_stopwords else set()
        self._stemmer = PorterStemmer()

    # Regex compiled once at class level for performance
    _DATELINE_RE = re.compile(
        r"^[A-Z][A-Z\s,\.]{0,40}\([A-Za-z]+\)\s*[-–—]+\s*",
        re.MULTILINE,
    )
    _AGENCY_RE = re.compile(
        r"\b(Reuters|Bernama|AFP|Associated Press|UPI|AP|Xinhua|ANI|PTI|IANS|"
        r"NAN|GNA|ANA|APA|Kyodo|Yonhap|IRNA|SANA|TASS|Interfax)\b",
        re.IGNORECASE,
    )

    def preprocess(self, text: str) -> str:
        if text is None:
            text = ""

        clean = str(text)

        # Strip wire-service datelines like "WASHINGTON (Reuters) -" or
        # "KUALA LUMPUR, March 3 (Bernama) --" before lowercasing so the
        # all-caps location pattern still matches.
        clean = self._DATELINE_RE.sub("", clean)
        # Remove bare agency names so "reuter" is not a dominant feature.
        clean = self._AGENCY_RE.sub("", clean)

        if self.lowercase:
            clean = clean.lower()

        clean = re.sub(r"[^a-zA-Z\s]", " ", clean)
        tokens = clean.split()

        processed: list[str] = []
        for token in tokens:
            if len(token) < self.min_word_length:
                continue
            if self._stopwords and token in self._stopwords:
                continue
            processed.append(self._stemmer.stem(token))

        return " ".join(processed)
