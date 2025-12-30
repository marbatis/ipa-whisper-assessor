from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from .ipa_normalize import normalize_ipa_text


Backend = Literal["espeak", "cmudict"]


@dataclass(frozen=True)
class G2POptions:
    backend: Backend = "espeak"
    lexicon_path: str | None = None
    language: str = "en-us"


def _load_lexicon(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in {".yml", ".yaml"}:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    elif p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        raise ValueError("Lexicon must be .yml/.yaml or .json")
    if not isinstance(data, dict):
        raise ValueError("Lexicon must be a mapping of word -> IPA")
    return {str(k).lower(): normalize_ipa_text(str(v)) for k, v in data.items()}


def g2p_words(words: list[str], options: G2POptions) -> list[str]:
    lex = _load_lexicon(options.lexicon_path)
    out: list[str] = []
    if options.backend == "espeak":
        out = _g2p_espeak(words, options.language, lex)
    elif options.backend == "cmudict":
        out = _g2p_cmudict(words, lex)
    else:
        raise ValueError(f"Unknown G2P backend: {options.backend}")
    return [normalize_ipa_text(x) for x in out]


def _g2p_espeak(words: list[str], language: str, lex: dict[str, str]) -> list[str]:
    try:
        from phonemizer.backend import EspeakBackend  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing g2p dependencies. Install with: pip install -e '.[g2p]'") from e

    backend = EspeakBackend(language=language, preserve_punctuation=False, with_stress=True)
    out: list[str] = []
    for w in words:
        lw = w.lower()
        if lw in lex:
            out.append(lex[lw])
            continue
        # phonemizer returns a string; for single word, strip.
        out.append(backend.phonemize([w], strip=True)[0])
    return out


_ARPABET_TO_IPA = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}


def _arpabet_pron_to_ipa(pron: str) -> str:
    phones: list[str] = []
    for part in pron.split():
        base = part.rstrip("012")  # ignore stress digits
        ipa = _ARPABET_TO_IPA.get(base)
        if ipa is None:
            continue
        phones.append(ipa)
    return "".join(phones)


def _g2p_cmudict(words: list[str], lex: dict[str, str]) -> list[str]:
    try:
        import pronouncing  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing g2p dependencies. Install with: pip install -e '.[g2p]'") from e

    out: list[str] = []
    for w in words:
        lw = w.lower()
        if lw in lex:
            out.append(lex[lw])
            continue
        prons = pronouncing.phones_for_word(lw)
        if not prons:
            out.append("")
            continue
        out.append(_arpabet_pron_to_ipa(prons[0]))
    return out

