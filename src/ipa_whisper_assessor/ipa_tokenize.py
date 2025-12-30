from __future__ import annotations

import unicodedata


_STRESS = {"ˈ", "ˌ"}
_WORD_BREAKS = {" ", "\t", "\n"}
_TIE_BARS = {"͡", "͜"}
_DIACRITICS_ATTACH_TO_PREV = {"ː", "̃", "̩"}

# Ordered longest-first for greedy scan.
_MULTI = [
    # Affricates (tie-bar and plain variants)
    "t͡ʃ",
    "d͡ʒ",
    "tʃ",
    "dʒ",
    # Common English diphthongs (model outputs may include these)
    "aɪ",
    "aʊ",
    "ɔɪ",
    "oʊ",
    "eɪ",
]


def _greedy_scan(s: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in _WORD_BREAKS:
            i += 1
            continue

        if ch in _STRESS:
            tokens.append(ch)
            i += 1
            continue

        matched = None
        for m in _MULTI:
            if s.startswith(m, i):
                matched = m
                break
        if matched is not None:
            tokens.append(matched)
            i += len(matched)
            continue

        # Capture tie-bar affricates even if not covered (e.g., with combining marks).
        if i + 2 < len(s) and s[i + 1] in _TIE_BARS:
            tokens.append(s[i : i + 3])
            i += 3
            continue

        # Attach diacritics/length to previous token when possible.
        if ch in _DIACRITICS_ATTACH_TO_PREV and tokens:
            tokens[-1] = tokens[-1] + ch
            i += 1
            continue

        tokens.append(ch)
        i += 1
    return tokens


def tokenize_ipa(text: str) -> list[str]:
    """
    Tokenize an IPA string into a sequence suitable for Levenshtein alignment.

    - Drops whitespace.
    - Keeps stress marks as tokens.
    - Greedy longest-match for multi-symbol phones.
    - Attaches length/diacritics to previous token.
    """
    text = unicodedata.normalize("NFC", text)
    return _greedy_scan(text)


def split_reference_words(reference: str) -> list[str]:
    # Minimal word split (keep punctuation out of word forms for G2P).
    parts: list[str] = []
    buff: list[str] = []
    for ch in reference.strip():
        if ch.isalnum() or ch in {"'", "’", "-"}:
            buff.append(ch)
        else:
            if buff:
                parts.append("".join(buff))
                buff = []
    if buff:
        parts.append("".join(buff))
    return parts

