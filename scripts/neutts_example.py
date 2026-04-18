#!/usr/bin/env python3
"""Run NeuTTS voice cloning (requires: pip install neutts, brew install espeak-ng).

Usage:
  # Using neutts repo samples (clone first: git clone https://github.com/neuphonic/neutts.git neutts_repo)
  python scripts/neutts_example.py \\
    --ref_audio neutts_repo/samples/jo.wav \\
    --ref_text neutts_repo/samples/jo.txt \\
    --input_text "My name is Andy. I'm 25 and I just moved to London."

  # Using your own ref (e.g. after diarize)
  python scripts/neutts_example.py \\
    --ref_audio ref_audio/coach_ben/segments/coach_ben_000.wav \\
    --ref_text ref_audio/coach_ben/segments/transcripts/coach_ben_000.txt \\
    --input_text "Hello, this is a test of voice cloning." \\
    --output outputs/neutts_test.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run NeuTTS voice cloning.")
    p.add_argument("--ref_audio", required=True, help="Reference audio WAV path.")
    p.add_argument("--ref_text", required=True, help="Reference text file path.")
    p.add_argument(
        "--input_text",
        default=(
            "My name is Andy. I'm 25 and I just moved to London. "
            "The underground is pretty confusing, but it gets me around in no time."
        ),
        help="Text to synthesize.",
    )
    p.add_argument(
        "--output",
        default="outputs/neutts_example.wav",
        help="Output WAV path (default: outputs/neutts_example.wav).",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Device: cpu, mps, or cuda (default: cpu).",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    try:
        from neutts import NeuTTS
        import soundfile as sf
    except RuntimeError as e:
        if "espeak" in str(e).lower():
            print(
                "Error: espeak-ng is required for NeuTTS.\n"
                "Install it with: brew install espeak-ng"
            )
            return 2
        raise

    ref_audio = Path(args.ref_audio)
    ref_text_path = Path(args.ref_text)
    if not ref_audio.exists():
        print(f"Error: Ref audio not found: {ref_audio}")
        return 1
    if not ref_text_path.exists():
        print(f"Error: Ref text not found: {ref_text_path}")
        return 1

    ref_text = ref_text_path.read_text(encoding="utf-8").strip()
    if not ref_text:
        print(f"Error: Ref text is empty: {ref_text_path}")
        return 1

    device = args.device
    if device == "mps":
        try:
            import torch
            if not torch.backends.mps.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"

    print("Loading NeuTTS (backbone + codec)...")
    tts = NeuTTS(
        backbone_repo="neuphonic/neutts-nano",
        backbone_device=device,
        codec_repo="neuphonic/neucodec",
        codec_device=device,
    )

    print("Encoding reference...")
    ref_codes = tts.encode_reference(str(ref_audio))

    print("Synthesizing...")
    wav = tts.infer(args.input_text, ref_codes, ref_text)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav, 24000)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
