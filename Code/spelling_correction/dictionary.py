"""
Dictionary management for spelling correction.
Combines NLTK words corpus with high-frequency Brown corpus terms.
"""

import bisect
import nltk


class Dictionary:
    """Valid-word dictionary with fast lookup and prefix search."""

    def __init__(self, corpus=None):
        """
        Parameters
        ----------
        corpus : CorpusProcessor or None
            If provided, augments the NLTK word list with top Brown-corpus
            words not already present.
        """
        self.words_set = set()
        self.words_sorted = []
        self._build(corpus)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build(self, corpus):
        # Ensure NLTK words corpus is available
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words', quiet=True)

        from nltk.corpus import words as nltk_words

        # Base: NLTK words corpus — lowercase, alpha-only, deduplicated
        for w in nltk_words.words('en'):
            lower = w.lower()
            if lower.isalpha():
                self.words_set.add(lower)

        # Augment with top-50k Brown corpus words not already in the set
        if corpus is not None and corpus.word_freq:
            # word_freq is a Counter; most_common(50000) gives top-50k
            for word, _ in corpus.word_freq.most_common(50000):
                if word.isalpha():
                    self.words_set.add(word)   # already lowercase

        # Sorted list for binary-search prefix queries
        self.words_sorted = sorted(self.words_set)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_valid_word(self, word):
        """O(1) membership test."""
        return word.lower() in self.words_set

    def search(self, prefix):
        """Return words starting with *prefix* (case-insensitive, max 1000).

        Uses bisect for O(log n) start position, then scans forward.
        """
        prefix = prefix.lower()
        if not prefix:
            return self.words_sorted[:1000]

        lo = bisect.bisect_left(self.words_sorted, prefix)
        results = []
        # Upper bound: prefix with last char incremented
        upper = prefix[:-1] + chr(ord(prefix[-1]) + 1)
        hi = bisect.bisect_left(self.words_sorted, upper)
        for i in range(lo, min(hi, lo + 1000)):
            results.append(self.words_sorted[i])
        return results

    def get_all_words(self):
        """Return the full sorted word list."""
        return self.words_sorted

    def get_word_count(self):
        """Return total number of unique words in the dictionary."""
        return len(self.words_set)
