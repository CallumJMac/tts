from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from src.utils.media_tools import ensure_tool


SPEAKER_ORDER = ["david", "boris", "joe", "stephen", "matthew"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stitch WAV files from outputs/good in speaker order with loudness "
            "normalization."
        )
    )
    parser.add_argument(
        "--speaker-order",
        default=",".join(SPEAKER_ORDER),
        help=(
            "Comma-separated speaker order, e.g. david,boris,joe "
            "(default: david,boris,joe,stephen,matthew)."
        ),
    )
    parser.add_argument(
        "--input-dir",
        default="outputs/good",
        help="Directory containing input WAV files (default: outputs/good).",
    )
    parser.add_argument(
        "--output",
        default="outputs/good/stitched.wav",
        help="Output WAV path (default: outputs/good/stitched.wav).",
    )
    parser.add_argument(
        "--lufs",
        type=float,
        default=-16.0,
        help="Target integrated loudness for normalization (default: -16 LUFS).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Output sample rate in Hz (default: 48000).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        choices=(1, 2),
        default=1,
        help="Output channel count (1=mono, 2=stereo; default: 1).",
    )
    parser.add_argument(
        "--pause-ms",
        type=int,
        default=0,
        help="Silence to insert between speakers in milliseconds (default: 0).",
    )
    parser.add_argument(
        "--codec",
        choices=("pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le"),
        default="pcm_s16le",
        help="Output PCM codec/bit depth (default: pcm_s16le).",
    )
    return parser


def sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"_(\d+)$", path.stem)
    number = int(match.group(1)) if match else 10**9
    return number, path.name


def ordered_inputs(input_dir: Path, speaker_order: list[str]) -> list[Path]:
    selected: list[Path] = []
    missing_speakers: list[str] = []

    for speaker in speaker_order:
        matches = sorted(input_dir.glob(f"{speaker}_*.wav"), key=sort_key)
        if not matches:
            missing_speakers.append(speaker)
            continue
        selected.extend(matches)

    if not selected:
        raise FileNotFoundError(f"No matching WAV files found in {input_dir}")

    if missing_speakers:
        missing = ", ".join(missing_speakers)
        print(f"Warning: no files found for speakers: {missing}")

    return selected


def build_filter_chain(
    file_count: int,
    lufs: float,
    channels: int,
    sample_rate: int,
    pause_ms: int,
) -> str:
    channel_layout = "mono" if channels == 1 else "stereo"
    chunks: list[str] = []

    for idx in range(file_count):
        chunks.append(
            (
                f"[{idx}:a]"
                f"aformat=sample_fmts=fltp:channel_layouts={channel_layout},"
                f"loudnorm=I={lufs}:LRA=11:TP=-1.5"
                f"[a{idx}]"
            )
        )

    concat_inputs: list[str] = []
    if pause_ms > 0 and file_count > 1:
        pause_seconds = pause_ms / 1000.0
        for idx in range(file_count - 1):
            chunks.append(
                (
                    f"anullsrc=r={sample_rate}:cl={channel_layout},"
                    f"atrim=0:{pause_seconds}"
                    f"[s{idx}]"
                )
            )
            concat_inputs.append(f"[a{idx}]")
            concat_inputs.append(f"[s{idx}]")
        concat_inputs.append(f"[a{file_count - 1}]")
    else:
        concat_inputs = [f"[a{idx}]" for idx in range(file_count)]

    chunks.append(f"{''.join(concat_inputs)}concat=n={len(concat_inputs)}:v=0:a=1[out]")
    return ";".join(chunks)


def main() -> int:
    args = build_parser().parse_args()
    ensure_tool("ffmpeg", "brew install ffmpeg")

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    speaker_order = [
        name.strip().lower() for name in args.speaker_order.split(",") if name.strip()
    ]
    if not speaker_order:
        raise ValueError("--speaker-order must include at least one speaker name")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = ordered_inputs(input_dir, speaker_order=speaker_order)
    filter_complex = build_filter_chain(
        file_count=len(files),
        lufs=args.lufs,
        channels=args.channels,
        sample_rate=args.sample_rate,
        pause_ms=args.pause_ms,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for wav_path in files:
        cmd.extend(["-i", str(wav_path)])

    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-ar",
            str(args.sample_rate),
            "-ac",
            str(args.channels),
            "-c:a",
            args.codec,
            str(output_path),
        ]
    )

    subprocess.run(cmd, check=True)

    print("Stitched files in order:")
    for wav_path in files:
        print(f"- {wav_path}")
    print(f"Output: {output_path}")

    return 0
