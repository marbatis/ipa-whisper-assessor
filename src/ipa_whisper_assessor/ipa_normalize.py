from __future__ import annotations

import re
import unicodedata


_SPACES = re.compile(r"\s+")


def normalize_ipa_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = _SPACES.sub(" ", text)
    return text


def normalize_chunk_word(text: str) -> str:
    # Keep raw in JSON; this is for display/alignment stability only.
    return normalize_ipa_text(text).lstrip()

