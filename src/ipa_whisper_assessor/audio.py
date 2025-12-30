from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def load_audio_16k_mono_ffmpeg(path: str | Path) -> tuple[np.ndarray, int]:
    """
    Decode audio with ffmpeg into float32 PCM, 16kHz, mono.
    """
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg or use a supported WAV input.")

    path = str(path)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    return audio, 16000

