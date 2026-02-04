"""
Bigram language model with Laplace smoothing.
Scores candidate corrections using P(w2|w1) in context.
"""

import math


class BigramModel:
    """Bigram language model built on top of a CorpusProcessor."""

    def __init__(self, corpus):
        """
        Parameters
        ----------
        corpus : CorpusProcessor
            A built CorpusProcessor instance (corpus.build() already called).
        """
        self.corpus = corpus

    # ------------------------------------------------------------------
    # Core probabilities
    # ------------------------------------------------------------------
    def bigram_probability(self, w1, w2):
        """P(w2 | w1) with Laplace (add-1) smoothing.

        Formula: (count(w1, w2) + 1) / (count(w1) + vocab_size)
        """
        bigram_count = self.corpus.get_bigram_count(w1, w2)
        unigram_count = self.corpus.word_freq[w1.lower()]
        return (bigram_count + 1) / (unigram_count + self.corpus.vocab_size)

    def sentence_log_probability(self, words):
        """Sum of log P(w_i | w_{i-1}) for consecutive word pairs.

        Parameters
        ----------
        words : list[str]
            Sequence of words (already lowercased).

        Returns
        -------
        float
            Log-probability of the sentence under the bigram model.
            Returns 0.0 for sequences shorter than 2 words.
        """
        if len(words) < 2:
            return 0.0
        log_prob = 0.0
        for i in range(1, len(words)):
            log_prob += math.log(self.bigram_probability(words[i - 1], words[i]))
        return log_prob

    def score_correction(self, context_before, candidate, context_after):
        """Context-aware bigram score for a candidate correction.

        Computes log P(candidate | context_before)
                + log P(context_after | candidate)

        Edge cases:
        - If context_before is None/empty, the first term is skipped.
        - If context_after is None/empty, the second term is skipped.

        Returns
        -------
        float
            Combined log-probability score.
        """
        score = 0.0
        if context_before:
            score += math.log(self.bigram_probability(context_before, candidate))
        if context_after:
            score += math.log(self.bigram_probability(candidate, context_after))
        return score
