"""
Core spelling-correction logic: error detection, candidate generation,
and bigram-aware ranking.
"""

import re
import math
from .edit_distance import get_all_distances


def tokenize_with_positions(text):
    """Extract word tokens with their character offsets.

    Returns
    -------
    list[dict]
        Each dict has keys: 'word', 'start', 'end'.
    """
    return [
        {'word': m.group(), 'start': m.start(), 'end': m.end()}
        for m in re.finditer(r'[A-Za-z]+', text)
    ]


def _edits1(word):
    """All strings at edit distance 1 (Norvig approach).

    Generates deletes, transposes, replaces, and inserts over a-z.
    Returns a *set* of strings — many will not be real words.
    """
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

    deletes    = [L + R[1:]       for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:]  for L, R in splits if len(R) > 1]
    replaces   = [L + c + R[1:]   for L, R in splits if R for c in 'abcdefghijklmnopqrstuvwxyz']
    inserts    = [L + c + R       for L, R in splits     for c in 'abcdefghijklmnopqrstuvwxyz']

    return set(deletes + transposes + replaces + inserts)


class SpellingCorrector:
    """Detect and correct spelling errors using dictionary + bigram context."""

    # Threshold: a real-word candidate must beat the original word's
    # bigram context score by at least this many log-probability units
    # to be flagged as a real-word error.  Calibrated at 2.5 to keep
    # common true positives (there/their ~3.0) while filtering out
    # frequency-driven false positives from the Brown corpus (~2.4).
    REAL_WORD_THRESHOLD = 2.5

    def __init__(self, dictionary, corpus, bigram_model):
        self.dictionary = dictionary
        self.corpus = corpus
        self.bigram_model = bigram_model

    # ------------------------------------------------------------------
    # Error detection
    # ------------------------------------------------------------------
    def detect_errors(self, text):
        """Identify errors in *text*.

        Returns
        -------
        list[dict or None]
            One entry per word token.  Each dict has:
                word, start, end, error_type ('non_word' | 'real_word')
            None entries mean no error detected for that token.
        """
        tokens = tokenize_with_positions(text)
        words = [t['word'] for t in tokens]
        results = []

        for idx, token in enumerate(tokens):
            word_lower = token['word'].lower()

            if not self.dictionary.is_valid_word(word_lower):
                # --- Non-word error ---
                results.append({
                    'word': token['word'],
                    'start': token['start'],
                    'end': token['end'],
                    'error_type': 'non_word'
                })
            else:
                # --- Real-word check ---
                error = self._check_real_word_error(word_lower, words, idx)
                if error:
                    results.append({
                        'word': token['word'],
                        'start': token['start'],
                        'end': token['end'],
                        'error_type': 'real_word'
                    })
                else:
                    results.append(None)

        return results

    def _check_real_word_error(self, word, words, word_index):
        """Return True if *word* is likely a real-word contextual error.

        Requires both a preceding and a following word.  Words at sentence
        boundaries (first or last) are skipped — a single bigram comparison
        is too noisy to confidently flag an error.
        """
        if word_index == 0 or word_index >= len(words) - 1:
            return False   # not enough context

        context_before = words[word_index - 1].lower()
        context_after  = words[word_index + 1].lower()

        # Score of the original word in context
        original_score = self.bigram_model.score_correction(
            context_before, word, context_after
        )

        # Generate distance-1 candidates that are valid dictionary words
        candidates = [c for c in _edits1(word) if self.dictionary.is_valid_word(c) and c != word]

        for candidate in candidates:
            cand_score = self.bigram_model.score_correction(
                context_before, candidate, context_after
            )
            if cand_score - original_score > self.REAL_WORD_THRESHOLD:
                return True   # found a better alternative — flag as error

        return False

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------
    def get_candidates(self, word, max_distance=2):
        """Generate candidate corrections filtered to dictionary words.

        Strategy (Norvig edits1 / edits2):
        1. All edit-distance-1 edits → keep those in dictionary.
        2. If empty and max_distance >= 2, apply edits1 to each *non-dictionary*
           edit-1 result → keep dictionary hits.

        Parameters
        ----------
        word : str
        max_distance : int
            1 or 2.

        Returns
        -------
        list[str]
            Dictionary words reachable within max_distance edits.
        """
        word_lower = word.lower()
        edits1_set = _edits1(word_lower)
        known1 = {w for w in edits1_set if self.dictionary.is_valid_word(w)}

        if known1 or max_distance < 2:
            return list(known1)

        # Distance-2: apply edits1 to each non-dictionary edit-1 candidate
        known2 = set()
        unknown1 = edits1_set - known1
        for e1 in unknown1:
            for e2 in _edits1(e1):
                if self.dictionary.is_valid_word(e2):
                    known2.add(e2)

        return list(known2)

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------
    def rank_candidates(self, word, candidates, context_words, word_index):
        """Score and rank candidates; return top 5 with edit-distance info.

        Parameters
        ----------
        word : str
            The (possibly misspelled) original word.
        candidates : list[str]
            Dictionary words to evaluate.
        context_words : list[str]
            All word tokens in the text (original case).
        word_index : int
            Index of *word* in context_words.

        Returns
        -------
        list[dict]
            Up to 5 dicts, each with 'word', 'distances', 'score', sorted
            descending by score.
        """
        context_before = context_words[word_index - 1].lower() if word_index > 0 else None
        context_after  = (context_words[word_index + 1].lower()
                          if word_index < len(context_words) - 1 else None)

        scored = []
        word_lower = word.lower()
        for cand in candidates:
            # Unigram log-prob + bigram context score
            p_word = self.corpus.get_word_probability(cand)
            log_p  = math.log(p_word) if p_word > 0 else -20.0   # floor for unseen
            bigram_score = self.bigram_model.score_correction(
                context_before, cand, context_after
            )
            score = log_p + bigram_score

            distances = get_all_distances(word_lower, cand)
            scored.append({
                'word': cand,
                'distances': distances,
                'score': score
            })

        # Sort descending by score, return top 5
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:5]

    # ------------------------------------------------------------------
    # High-level entry point
    # ------------------------------------------------------------------
    def correct_word(self, word, context_words, word_index):
        """Full pipeline: detect → candidates → rank.

        Returns
        -------
        dict
            'error_type': 'non_word' | 'real_word' | None
            'suggestions': list (empty if no error)
        """
        word_lower = word.lower()

        # Determine error type
        if not self.dictionary.is_valid_word(word_lower):
            error_type = 'non_word'
        elif self._check_real_word_error(word_lower,
                                         [w.lower() for w in context_words],
                                         word_index):
            error_type = 'real_word'
        else:
            return {'error_type': None, 'suggestions': []}

        candidates = self.get_candidates(word)
        suggestions = self.rank_candidates(word, candidates, context_words, word_index)

        return {
            'error_type': error_type,
            'suggestions': suggestions
        }
