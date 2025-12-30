from __future__ import annotations

import argparse
from pathlib import Path

from ipa_whisper_assessor.cli import assess


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--reference-file", required=True)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--g2p", default="espeak")
    args = ap.parse_args()

    ref = Path(args.reference_file).read_text(encoding="utf-8").strip()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.audio).stem
    assess(
        audio=args.audio,
        reference=ref,
        g2p=args.g2p,
        lexicon=None,
        out_json=str(out_dir / f"{stem}.json"),
        out_html=str(out_dir / f"{stem}.html"),
        timestamps="word",
        device="auto",
        chunk_length=30,
        model="neurlang/ipa-whisper-small",
        espeak_language="en-us",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

