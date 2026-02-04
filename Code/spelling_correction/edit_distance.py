"""
Edit distance algorithms for spelling correction.
All implementations use full DP matrices (not space-optimised) so the
computation is demonstrable and inspectable.
"""


def levenshtein_distance(s1, s2):
    """Classic Levenshtein distance — insert / delete / substitute each cost 1."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )
    return dp[m][n]


def damerau_levenshtein_distance(s1, s2):
    """Optimal String Alignment (restricted Damerau-Levenshtein) distance.

    Adds adjacent-character transposition at cost 1 on top of the three
    classic operations.  Note: this is the OSA variant, not the true
    unrestricted DL distance (no substring may be edited more than once).
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,          # deletion
                dp[i][j - 1] + 1,          # insertion
                dp[i - 1][j - 1] + cost    # substitution
            )
            # Transposition
            if (i > 1 and j > 1
                    and s1[i - 1] == s2[j - 2]
                    and s1[i - 2] == s2[j - 1]):
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + cost)
    return dp[m][n]


def weighted_edit_distance(s1, s2, ins_cost=1.0, del_cost=1.0,
                           sub_cost=1.5, trans_cost=1.0):
    """Weighted edit distance with configurable per-operation costs.

    Substitution is weighted higher (default 1.5) to make the algorithm
    prefer insertions/deletions over substitutions — pedagogically
    interesting for demonstrating real-word vs non-word trade-offs.
    """
    m, n = len(s1), len(s2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i * del_cost
    for j in range(n + 1):
        dp[0][j] = j * ins_cost

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]          # match — no cost
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + del_cost,
                    dp[i][j - 1] + ins_cost,
                    dp[i - 1][j - 1] + sub_cost
                )
            # Transposition
            if (i > 1 and j > 1
                    and s1[i - 1] == s2[j - 2]
                    and s1[i - 2] == s2[j - 1]):
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + trans_cost)
    return dp[m][n]


def get_all_distances(s1, s2):
    """Return all three edit distances in one call.

    Returns
    -------
    dict with keys 'levenshtein', 'damerau_levenshtein', 'weighted'
    """
    return {
        'levenshtein': levenshtein_distance(s1, s2),
        'damerau_levenshtein': damerau_levenshtein_distance(s1, s2),
        'weighted': weighted_edit_distance(s1, s2)
    }
