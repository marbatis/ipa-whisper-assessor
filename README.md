# ipa-whisper-assessor

Offline IPA transcription + scripted pronunciation error detection for English, using the open-source model `neurlang/ipa-whisper-small`.

## What it does

- **Transcribe**: audio → IPA (optionally with word timestamps)
- **Assess (scripted)**: audio + reference text → alignment + mistake summary (subs/ins/del), plus JSON + HTML report

## Install

Prereqs:

- Python 3.9+
- `ffmpeg` (recommended, used for broad audio support)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[transcribe,espeak,cmudict,dev]"
```

If `python` isn’t on your PATH (common on macOS), use `python3` as above (or `alias python=python3`).

Notes:

- The model runs locally via PyTorch + Transformers. The first run downloads weights from Hugging Face.
- The model card recommends IPA-mode plumbing: disable forced decoder prompt tokens and suppression. This repo applies those settings automatically.

## CLI

### Transcribe

```bash
ipa-assess transcribe path/to/audio.wav --out-json outputs/transcription.json --timestamps word
```

### Assess (scripted)

```bash
ipa-assess assess path/to/audio.wav \
  --reference "The zebra is in the zoo." \
  --g2p espeak \
  --out-json outputs/report.json \
  --out-html outputs/report.html
```

### Batch folder

```bash
ipa-assess batch data/ --ref-csv references.csv --out-dir outputs/
```

## Device selection

Use `--device auto` (default) or explicitly choose: `cpu`, `cuda`, `mps`.

On Apple Silicon (like your M3), `--device auto` should select `mps`. If you hit device issues, run with `--device mps`.

## G2P backends

- `--g2p espeak`: uses the `phonemizer` package + `espeak-ng` installed on your system.
- `--g2p cmudict`: uses `pronouncing` (CMUdict) and a small ARPAbet→IPA mapping.

You can add word overrides with `--lexicon lexicon.yml`:

```yaml
zebra: "ˈziːbrə"
analysis: "əˈnæləsɪs"
```

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## Roadmap

- Streaming / pseudo-realtime (microphone chunking)
- Corpus evaluation harness + confusion matrix
- Optional Gradio demo UI
