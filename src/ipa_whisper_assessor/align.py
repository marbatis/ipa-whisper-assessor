from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .edit_distance import EditOp, edit_counts, levenshtein_ops
from .ipa_tokenize import tokenize_ipa


@dataclass(frozen=True)
class PredWord:
    ipa: str
    start: Optional[float] = None
    end: Optional[float] = None


@dataclass(frozen=True)
class AlignedWord:
    index: int
    reference_word: str
    expected_ipa: str
    predicted_ipa: str
    start: Optional[float]
    end: Optional[float]
    phoneme_ops: list[EditOp]


def _word_cost(expected_ipa: str, predicted_ipa: str) -> int:
    e = tokenize_ipa(expected_ipa)
    p = tokenize_ipa(predicted_ipa)
    ops = levenshtein_ops(e, p)
    subs, ins, dels = edit_counts(ops)
    return subs + ins + dels


def align_words(
    reference_words: list[str],
    expected_ipa_words: list[str],
    predicted_words: list[PredWord],
) -> list[AlignedWord]:
    """
    Word-level alignment by DP using phoneme edit distance as the word cost.
    """
    n = len(reference_words)
    m = len(predicted_words)
    if len(expected_ipa_words) != n:
        raise ValueError("expected_ipa_words must match reference_words length")

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "del_word"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "ins_word"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_sub = _word_cost(expected_ipa_words[i - 1], predicted_words[j - 1].ipa)
            sub = dp[i - 1][j - 1] + cost_sub
            dele = dp[i - 1][j] + 1
            ins = dp[i][j - 1] + 1
            best = min(sub, dele, ins)
            dp[i][j] = best
            if best == sub:
                back[i][j] = "match_word"
            elif best == dele:
                back[i][j] = "del_word"
            else:
                back[i][j] = "ins_word"

    aligned: list[AlignedWord] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = back[i][j]
        if step == "match_word":
            exp = expected_ipa_words[i - 1]
            pred = predicted_words[j - 1]
            ops = levenshtein_ops(tokenize_ipa(exp), tokenize_ipa(pred.ipa))
            aligned.append(
                AlignedWord(
                    index=i - 1,
                    reference_word=reference_words[i - 1],
                    expected_ipa=exp,
                    predicted_ipa=pred.ipa,
                    start=pred.start,
                    end=pred.end,
                    phoneme_ops=ops,
                )
            )
            i -= 1
            j -= 1
        elif step == "del_word":
            exp = expected_ipa_words[i - 1]
            ops = levenshtein_ops(tokenize_ipa(exp), [])
            aligned.append(
                AlignedWord(
                    index=i - 1,
                    reference_word=reference_words[i - 1],
                    expected_ipa=exp,
                    predicted_ipa="",
                    start=None,
                    end=None,
                    phoneme_ops=ops,
                )
            )
            i -= 1
        else:
            # Insertion word: ignore it at the word table level (still impacts overall PER via ops)
            pred = predicted_words[j - 1]
            ops = levenshtein_ops([], tokenize_ipa(pred.ipa))
            aligned.append(
                AlignedWord(
                    index=-1,
                    reference_word="",
                    expected_ipa="",
                    predicted_ipa=pred.ipa,
                    start=pred.start,
                    end=pred.end,
                    phoneme_ops=ops,
                )
            )
            j -= 1

    aligned.reverse()
    # Drop pure insertions for the per-word table, but keep them separate for metrics.
    return [a for a in aligned if a.index >= 0]


def compute_overall_ops(aligned_words: list[AlignedWord]) -> list[EditOp]:
    ops: list[EditOp] = []
    for w in aligned_words:
        ops.extend(w.phoneme_ops)
    return ops

