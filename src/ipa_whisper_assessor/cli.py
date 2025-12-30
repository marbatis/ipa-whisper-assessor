from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .align import AlignedWord, PredWord, align_words
from .edit_distance import edit_counts, levenshtein_ops
from .g2p import G2POptions, g2p_words
from .audio import ffmpeg_available
from .ipa_tokenize import split_reference_words
from .ipa_tokenize import tokenize_ipa
from .report import write_html, write_json
from .schemas import (
    AssessmentMetrics,
    AssessmentResult,
    MistakeEvent,
    PhonemeOp,
    TranscriptionResult,
    WordAlignment,
)
from .score import apply_default_mistake_rules, substitution_histogram
from .transcribe import MODEL_ID, TranscribeOptions, transcribe_audio


app = typer.Typer(
    add_completion=False,
    help="IPA transcription + pronunciation assessment (offline).",
    pretty_exceptions_show_locals=False,
)


def _configure_warnings() -> None:
    """
    Suppress known noisy warnings (set `IPA_ASSESS_SHOW_WARNINGS=1` to keep them).
    """
    import os
    import warnings

    if os.environ.get("IPA_ASSESS_SHOW_WARNINGS", "").strip().lower() in {"1", "true", "yes"}:
        return

    warnings.filterwarnings(
        "ignore",
        message=r"urllib3 v2 only supports OpenSSL .*",
        category=Warning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Using custom `forced_decoder_ids` from the \(generation\) config\..*",
        category=Warning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Transcription using a multilingual Whisper will default to language detection.*",
        category=Warning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Whisper did not predict an ending timestamp.*",
        category=Warning,
    )

    # Transformers often emits these as logs, not warnings.
    try:  # pragma: no cover
        from transformers.utils import logging as hf_logging  # type: ignore[import-not-found]

        hf_logging.set_verbosity_error()
    except Exception:
        pass


def _ensure_parent(path: str | Path | None) -> None:
    if not path:
        return
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)


@app.command()
def doctor() -> None:
    """
    Print a quick environment diagnostic (useful for macOS/MPS setup).
    """
    import platform
    import shutil
    import sys

    typer.echo(f"python: {sys.version.split()[0]}")
    typer.echo(f"platform: {platform.platform()}")
    typer.echo(f"ffmpeg: {'yes' if ffmpeg_available() else 'no'}")

    try:
        import torch  # type: ignore[import-not-found]

        typer.echo(f"torch: {getattr(torch, '__version__', '?')}")
        typer.echo(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()  # type: ignore[attr-defined]
        typer.echo(f"torch.backends.mps.is_available: {mps_ok}")
    except Exception as e:
        typer.echo(f"torch: not importable ({type(e).__name__})")

    try:
        import transformers  # type: ignore[import-not-found]

        typer.echo(f"transformers: {getattr(transformers, '__version__', '?')}")
    except Exception as e:
        typer.echo(f"transformers: not importable ({type(e).__name__})")

    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    typer.echo(f"espeak-ng: {'yes' if espeak else 'no'}")

    try:
        import phonemizer  # type: ignore[import-not-found]

        typer.echo(f"phonemizer: {getattr(phonemizer, '__version__', '?')}")
    except Exception:
        typer.echo("phonemizer: not installed (optional, for --g2p espeak)")

    try:
        import pronouncing  # type: ignore[import-not-found]

        typer.echo(f"pronouncing: {getattr(pronouncing, '__version__', '?')}")
    except Exception:
        typer.echo("pronouncing: not installed (optional, for --g2p cmudict)")


@app.command()
def transcribe(
    audio: str = typer.Argument(..., help="Path to audio file."),
    out_json: Optional[str] = typer.Option(None, "--out-json", help="Write JSON output."),
    out_txt: Optional[str] = typer.Option(None, "--out-txt", help="Write IPA text output."),
    timestamps: Optional[str] = typer.Option(None, "--timestamps", help="word|chunk (optional)"),
    device: str = typer.Option("auto", "--device", help="auto|cpu|cuda|mps"),
    chunk_length: int = typer.Option(30, "--chunk-length", help="Chunk length in seconds (long-form)."),
    model: str = typer.Option(MODEL_ID, "--model", help="HF model id."),
) -> None:
    _configure_warnings()
    ts = None
    if timestamps:
        if timestamps not in {"word", "chunk"}:
            raise typer.BadParameter("--timestamps must be 'word' or 'chunk'")
        ts = timestamps  # type: ignore[assignment]

    opts = TranscribeOptions(model=model, device=device, chunk_length_s=chunk_length, timestamps=ts)  # type: ignore[arg-type]
    result = transcribe_audio(audio, opts)

    if out_json:
        _ensure_parent(out_json)
        Path(out_json).write_text(result.model_dump_json(indent=2), encoding="utf-8")
    if out_txt:
        _ensure_parent(out_txt)
        Path(out_txt).write_text(result.ipa_text + "\n", encoding="utf-8")
    if not out_json and not out_txt:
        typer.echo(result.ipa_text)


@app.command()
def assess(
    audio: str = typer.Argument(..., help="Path to audio file."),
    reference: str = typer.Option(..., "--reference", help="Reference text (script)."),
    g2p: str = typer.Option("espeak", "--g2p", help="espeak|cmudict"),
    lexicon: Optional[str] = typer.Option(None, "--lexicon", help="YAML/JSON word->IPA overrides."),
    out_json: str = typer.Option(..., "--out-json", help="Write JSON report."),
    out_html: Optional[str] = typer.Option(None, "--out-html", help="Write HTML report."),
    timestamps: str = typer.Option("word", "--timestamps", help="word|chunk (recommended: word)"),
    device: str = typer.Option("auto", "--device", help="auto|cpu|cuda|mps"),
    chunk_length: int = typer.Option(30, "--chunk-length", help="Chunk length in seconds (long-form)."),
    model: str = typer.Option(MODEL_ID, "--model", help="HF model id."),
    espeak_language: str = typer.Option("en-us", "--espeak-language", help="eSpeak language (espeak backend)."),
) -> None:
    _configure_warnings()
    if timestamps not in {"word", "chunk"}:
        raise typer.BadParameter("--timestamps must be 'word' or 'chunk'")
    if g2p not in {"espeak", "cmudict"}:
        raise typer.BadParameter("--g2p must be 'espeak' or 'cmudict'")

    # 1) Transcribe
    t_opts = TranscribeOptions(
        model=model, device=device, chunk_length_s=chunk_length, timestamps=timestamps  # type: ignore[arg-type]
    )
    transcription: TranscriptionResult = transcribe_audio(audio, t_opts)

    # 2) Build predicted word list
    pred_words = [
        PredWord(ipa=w.ipa, start=w.start, end=w.end) for w in (transcription.ipa_words or [])
    ]
    if not pred_words:
        # Fallback: treat the whole utterance as one "word"
        pred_words = [PredWord(ipa=transcription.ipa_text, start=None, end=None)]

    # 3) Reference words + expected IPA
    ref_words = split_reference_words(reference)
    g_opts = G2POptions(backend=g2p, lexicon_path=lexicon, language=espeak_language)  # type: ignore[arg-type]
    expected_ipa_words = g2p_words(ref_words, g_opts)

    # 4) Align
    # If the timestamp chunking doesn't match reference word granularity (common in some audio),
    # fall back to utterance-level alignment for a meaningful PER + substitution summary.
    if len(pred_words) < max(5, len(ref_words) // 2):
        expected_full = " ".join(expected_ipa_words)
        predicted_full = transcription.ipa_text
        ops = levenshtein_ops(tokenize_ipa(expected_full), tokenize_ipa(predicted_full))
        start = pred_words[0].start if pred_words else None
        end = pred_words[-1].end if pred_words else None
        aligned = [
            AlignedWord(
                index=0,
                reference_word="(full utterance)",
                expected_ipa=expected_full,
                predicted_ipa=predicted_full,
                start=start,
                end=end,
                phoneme_ops=ops,
            )
        ]
    else:
        aligned = align_words(ref_words, expected_ipa_words, pred_words)

    # 5) Metrics + summaries
    all_ops = []
    for w in aligned:
        all_ops.extend(w.phoneme_ops)

    subs, ins, dels = edit_counts(all_ops)
    denom = max(1, sum(1 for o in all_ops if o.expected is not None))
    per = (subs + ins + dels) / denom

    mistakes = [
        MistakeEvent(rule=rule, expected=e, predicted=p, count=n)
        for (rule, e, p, n) in apply_default_mistake_rules(all_ops)
    ]

    result = AssessmentResult(
        audio_path=audio,
        reference=reference,
        g2p_backend=g2p,
        model=model,
        transcription=transcription,
        word_alignments=[
            WordAlignment(
                word_index=w.index,
                reference_word=w.reference_word,
                expected_ipa=w.expected_ipa,
                predicted_ipa=w.predicted_ipa,
                start=w.start,
                end=w.end,
                phoneme_ops=[
                    PhonemeOp(op=o.op, expected=o.expected, predicted=o.predicted) for o in w.phoneme_ops
                ],
            )
            for w in aligned
        ],
        metrics=AssessmentMetrics(
            phoneme_error_rate=per, substitutions=subs, insertions=ins, deletions=dels
        ),
        substitution_histogram=substitution_histogram(all_ops),
        mistakes=mistakes,
    )

    _ensure_parent(out_json)
    write_json(result, out_json)
    if out_html:
        _ensure_parent(out_html)
        write_html(result, out_html)

    typer.echo(f"Wrote {out_json}")
    if out_html:
        typer.echo(f"Wrote {out_html}")


@app.command()
def batch(
    folder: str = typer.Argument(..., help="Folder of audio files."),
    ref_csv: str = typer.Option(..., "--ref-csv", help="CSV with columns: file,reference"),
    out_dir: str = typer.Option("outputs", "--out-dir", help="Output directory."),
    g2p: str = typer.Option("espeak", "--g2p", help="espeak|cmudict"),
    device: str = typer.Option("auto", "--device", help="auto|cpu|cuda|mps"),
    model: str = typer.Option(MODEL_ID, "--model", help="HF model id."),
) -> None:
    _configure_warnings()
    import csv

    folder_p = Path(folder)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    refs: dict[str, str] = {}
    with open(ref_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            refs[row["file"]] = row["reference"]

    for audio_path in sorted(folder_p.glob("*")):
        if audio_path.name not in refs:
            continue
        stem = audio_path.stem
        out_json = out_p / f"{stem}.json"
        out_html = out_p / f"{stem}.html"
        assess(
            audio=str(audio_path),
            reference=refs[audio_path.name],
            g2p=g2p,
            lexicon=None,
            out_json=str(out_json),
            out_html=str(out_html),
            timestamps="word",
            device=device,
            chunk_length=30,
            model=model,
            espeak_language="en-us",
        )
