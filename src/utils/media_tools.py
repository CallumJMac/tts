from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def ensure_tool(binary: str, install_hint: str) -> None:
    if shutil.which(binary) is None:
        raise RuntimeError(f"{binary} not found on PATH. Install with: {install_hint}")


def extract_wav_from_video(
    video_path: Path,
    audio_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    overwrite: bool = True,
) -> None:
    ensure_tool("ffmpeg", "brew install ffmpeg")
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        str(audio_path),
    ]
    subprocess.run(cmd, check=True)


def transcribe_with_whisper(
    audio_path: Path,
    text_dir: Path,
    model: str,
    language: str,
) -> Path:
    ensure_tool(
        "whisper",
        "install openai-whisper and retry, or pass an explicit reference text file",
    )
    text_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "whisper",
        str(audio_path),
        "--model",
        model,
        "--output_dir",
        str(text_dir),
        "--output_format",
        "txt",
        "--language",
        language,
    ]
    subprocess.run(cmd, check=True)

    text_path = text_dir / f"{audio_path.stem}.txt"
    if not text_path.exists():
        raise RuntimeError(
            f"Whisper completed but text file was not found: {text_path}"
        )
    return text_path
