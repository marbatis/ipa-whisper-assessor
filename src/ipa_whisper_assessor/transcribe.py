from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from .ipa_normalize import normalize_chunk_word, normalize_ipa_text
from .schemas import IpaWord, TranscriptionResult

MODEL_ID = "neurlang/ipa-whisper-small"


Device = Literal["auto", "cpu", "cuda", "mps"]


@dataclass(frozen=True)
class TranscribeOptions:
    model: str = MODEL_ID
    device: Device = "auto"
    chunk_length_s: int = 30
    timestamps: Optional[Literal["word", "chunk"]] = None


def _resolve_device(device: Device) -> int | str:
    if device == "auto":
        return "auto"
    if device == "cpu":
        return -1
    if device == "cuda":
        return 0
    if device == "mps":
        return "mps"
    raise ValueError(f"Unknown device: {device}")


_PIPELINE = None


def _get_pipeline(model_id: str, device: Device, chunk_length_s: int):
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    try:
        from transformers import pipeline  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing transcribe dependencies. Install with: pip install -e '.[transcribe]'"
        ) from e

    p = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        chunk_length_s=chunk_length_s,
        device=_resolve_device(device),
    )

    # Model-card-recommended settings for IPA mode.
    try:
        model = p.model  # type: ignore[attr-defined]
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.forced_decoder_ids = None
            model.generation_config.suppress_tokens = []
    except Exception:
        # Best-effort; some pipelines may wrap differently.
        pass

    _PIPELINE = p
    return p


def transcribe_audio(audio_path: str, options: TranscribeOptions) -> TranscriptionResult:
    p = _get_pipeline(options.model, options.device, options.chunk_length_s)
    kwargs: dict[str, Any] = {}
    if options.timestamps is not None:
        kwargs["return_timestamps"] = options.timestamps
    out = p(audio_path, **kwargs)

    ipa_text = normalize_ipa_text(out.get("text", ""))
    words: list[IpaWord] = []
    for chunk in out.get("chunks") or []:
        ts = chunk.get("timestamp")
        start = None
        end = None
        if isinstance(ts, (tuple, list)) and len(ts) == 2:
            start, end = ts[0], ts[1]
        raw = chunk.get("text", "")
        words.append(
            IpaWord(
                ipa=normalize_chunk_word(raw),
                raw=raw,
                start=float(start) if start is not None else None,
                end=float(end) if end is not None else None,
            )
        )

    return TranscriptionResult(audio_path=audio_path, model=options.model, ipa_text=ipa_text, ipa_words=words)

