#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.media_tools import ensure_tool, extract_wav_from_video


VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert video files (e.g. ref_video/david_1.mov) to WAV.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more video files or directories to process.",
    )
    parser.add_argument(
        "--output-dir",
        default="ref_audio",
        help="Directory to write WAV files into (default: ref_audio).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Output WAV sample rate in Hz (default: 16000).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Output channel count (default: 1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser


def collect_input_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        path = Path(item).expanduser().resolve()
        if not path.exists():
            print(f"[skip] Not found: {path}")
            continue

        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
                    files.append(candidate)
        elif path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            files.append(path)
        else:
            print(f"[skip] Unsupported file type: {path}")

    seen: set[Path] = set()
    deduped: list[Path] = []
    for file_path in files:
        if file_path in seen:
            continue
        seen.add(file_path)
        deduped.append(file_path)
    return deduped


def convert_file(
    input_file: Path,
    output_dir: Path,
    sample_rate: int,
    channels: int,
    overwrite: bool,
) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{input_file.stem}.wav"

    if output_file.exists() and not overwrite:
        print(f"[skip] Exists (use --overwrite): {output_file}")
        return True

    try:
        extract_wav_from_video(
            video_path=input_file,
            audio_path=output_file,
            sample_rate=sample_rate,
            channels=channels,
            overwrite=overwrite,
        )
        print(f"[ok] {input_file} -> {output_file}")
        return True
    except Exception as exc:
        print(f"[fail] {input_file}")
        print(str(exc))
        return False


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        ensure_tool("ffmpeg", "brew install ffmpeg")
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 2

    input_files = collect_input_files(args.inputs)
    if not input_files:
        print("No supported input video files found.")
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    ok_count = 0
    for input_file in input_files:
        if convert_file(
            input_file=input_file,
            output_dir=output_dir,
            sample_rate=args.sample_rate,
            channels=args.channels,
            overwrite=args.overwrite,
        ):
            ok_count += 1

    total = len(input_files)
    print(f"Done: {ok_count}/{total} converted.")
    return 0 if ok_count == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
