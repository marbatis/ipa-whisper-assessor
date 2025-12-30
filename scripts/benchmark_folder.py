from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from ipa_whisper_assessor.transcribe import TranscribeOptions, transcribe_audio


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    folder = Path(args.folder)
    files = [p for p in folder.glob("*") if p.suffix.lower() in {".wav", ".mp3", ".m4a"}]
    if not files:
        print("No audio files found.")
        return 1

    t0 = perf_counter()
    for p in files:
        res = transcribe_audio(str(p), TranscribeOptions(device=args.device))
        print(p.name, len(res.ipa_text))
    dt = perf_counter() - t0
    print(f"Processed {len(files)} files in {dt:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

