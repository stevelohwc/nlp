"""
Corpus processing for spelling correction.
Loads the NLTK Brown corpus and builds unigram / bigram frequency tables.
"""

from collections import Counter
import nltk


class CorpusProcessor:
    """Load Brown corpus and build word-frequency and bigram-frequency tables."""

    def __init__(self):
        self.word_freq = Counter()       # unigram counts
        self.bigrams = Counter()        # (w1, w2) bigram counts
        self.total_words = 0
        self.vocab_size = 0
        self._built = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self):
        """Load Brown corpus, filter tokens, and populate counters."""
        # Ensure Brown corpus is available
        try:
            nltk.data.find('corpora/brown')
        except LookupError:
            nltk.download('brown', quiet=True)

        from nltk.corpus import brown

        # --- Unigrams (all categories) ---
        for token in brown.words():
            word = token.lower()
            if word.isalpha():
                self.word_freq[word] += 1

        self.total_words = sum(self.word_freq.values())
        self.vocab_size = len(self.word_freq)

        # --- Bigrams (sentence-by-sentence, no cross-sentence pairs) ---
        for sentence in brown.sents():
            filtered = [token.lower() for token in sentence if token.lower().isalpha()]
            for i in range(len(filtered) - 1):
                self.bigrams[(filtered[i], filtered[i + 1])] += 1

        self._built = True

    def get_word_probability(self, word):
        """Return P(word) = count(word) / total_words."""
        if self.total_words == 0:
            return 0.0
        return self.word_freq[word.lower()] / self.total_words

    def get_bigram_count(self, w1, w2):
        """Return raw bigram count for (w1, w2)."""
        return self.bigrams[(w1.lower(), w2.lower())]
