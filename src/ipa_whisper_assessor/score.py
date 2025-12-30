from __future__ import annotations

from collections import Counter

from .edit_distance import EditOp


def substitution_histogram(ops: list[EditOp]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for o in ops:
        if o.op == "sub" and o.expected is not None and o.predicted is not None:
            c[f"{o.expected}→{o.predicted}"] += 1
    return dict(c)


def apply_default_mistake_rules(ops: list[EditOp]) -> list[tuple[str, str, str, int]]:
    """
    Return (rule, expected, predicted, count).
    Keep this small and conservative; customize later via YAML if desired.
    """
    c = Counter()
    for o in ops:
        if o.op != "sub" or o.expected is None or o.predicted is None:
            continue
        e = o.expected
        p = o.predicted
        if (e, p) in {("z", "s"), ("s", "z")}:
            c[("VOICING_ERROR_FRICATIVE", e, p)] += 1
        elif (e, p) in {("θ", "s"), ("ð", "d"), ("ð", "z")}:
            c[("TH_FRONTING_OR_STOPPING", e, p)] += 1
        elif (e, p) in {("ɪ", "i"), ("i", "ɪ"), ("ʊ", "u"), ("u", "ʊ")}:
            c[("VOWEL_TENSE_LAX", e, p)] += 1
    return [(rule, e, p, n) for (rule, e, p), n in c.items()]

