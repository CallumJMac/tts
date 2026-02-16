from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from src.utils.media_tools import extract_wav_from_video, transcribe_with_whisper


def parse_dtype(value: str) -> str:
    key = value.strip().lower()
    mapping = {
        "float16": "float16",
        "fp16": "float16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float32": "float32",
        "fp32": "float32",
    }
    if key not in mapping:
        raise argparse.ArgumentTypeError(
            "dtype must be one of: float16, bfloat16, float32"
        )
    return mapping[key]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen3 voice cloning from CLI.")
    parser.add_argument(
        "--ref-video",
        help="Reference video path. If set, derives ref_audio/ref_text/output paths by filename.",
    )
    parser.add_argument("--ref-audio", help="Reference audio path or URL.")
    parser.add_argument("--ref-text", help="Path to the reference text file.")
    parser.add_argument("--target-text", help="Text to synthesize.")
    parser.add_argument(
        "--target-text-file",
        help="Path to a text file containing the text to synthesize.",
    )
    parser.add_argument(
        "--output", default="output_voice_clone.wav", help="Output WAV file path."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Model checkpoint or HF repo.",
    )
    parser.add_argument(
        "--device", default="mps", help="Device map, e.g. mps, cpu, cuda:0."
    )
    parser.add_argument(
        "--dtype",
        type=parse_dtype,
        default="float16",
        help="Torch dtype: float16, bfloat16, float32.",
    )
    parser.add_argument(
        "--language", default="English", help="Language argument for generation."
    )
    parser.add_argument(
        "--flash-attn", action="store_true", help="Enable FlashAttention-2."
    )
    parser.add_argument(
        "--ref-audio-dir",
        default="ref_audio",
        help="Directory for auto-generated reference audio from --ref-video.",
    )
    parser.add_argument(
        "--ref-text-dir",
        default="ref_text",
        help="Directory for auto-generated reference text from --ref-video.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for auto-generated output audio from --ref-video.",
    )
    parser.add_argument(
        "--whisper-model",
        default="turbo",
        help="Whisper model for auto-transcribing missing ref text.",
    )
    parser.add_argument(
        "--whisper-language",
        default="en",
        help="Whisper language code for auto-transcription.",
    )
    return parser


def resolve_reference_paths(
    args: argparse.Namespace,
) -> tuple[str, Path, Optional[Path]]:
    if args.ref_video:
        video_path = Path(args.ref_video)
        if not video_path.exists():
            raise FileNotFoundError(f"Reference video not found: {video_path}")

        stem = video_path.stem
        ref_audio_path = (
            Path(args.ref_audio)
            if args.ref_audio
            else Path(args.ref_audio_dir) / f"{stem}.wav"
        )
        ref_text_path = (
            Path(args.ref_text)
            if args.ref_text
            else Path(args.ref_text_dir) / f"{stem}.txt"
        )

        if not ref_audio_path.exists():
            print(f"Reference audio missing, extracting: {ref_audio_path}")
            extract_wav_from_video(
                video_path=video_path,
                audio_path=ref_audio_path,
                sample_rate=16000,
                channels=1,
                overwrite=True,
            )
        else:
            print(f"Using existing reference audio: {ref_audio_path}")

        if not ref_text_path.exists():
            print(f"Reference text missing, transcribing: {ref_text_path}")
            transcribe_with_whisper(
                audio_path=ref_audio_path,
                text_dir=ref_text_path.parent,
                model=args.whisper_model,
                language=args.whisper_language,
            )
        else:
            print(f"Using existing reference text: {ref_text_path}")

        if args.output == "output_voice_clone.wav":
            args.output = str(Path(args.output_dir) / f"{stem}.wav")

        return str(ref_audio_path), ref_text_path, video_path

    if not args.ref_audio or not args.ref_text:
        raise ValueError(
            "Provide --ref-video, or provide both --ref-audio and --ref-text."
        )

    return args.ref_audio, Path(args.ref_text), None


def main() -> int:
    args = build_parser().parse_args()

    try:
        import soundfile as sf
        import torch
        from qwen_tts import Qwen3TTSModel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency for voice cloning. Activate the project environment and install required packages."
        ) from exc

    ref_audio, ref_text_path, _ = resolve_reference_paths(args)
    if not ref_text_path.exists():
        raise FileNotFoundError(f"Reference text file not found: {ref_text_path}")

    ref_text = ref_text_path.read_text(encoding="utf-8").strip()
    if not ref_text:
        raise ValueError(f"Reference text file is empty: {ref_text_path}")

    if not args.target_text and not args.target_text_file:
        raise ValueError("Provide either --target-text or --target-text-file.")

    if args.target_text and args.target_text_file:
        raise ValueError("Use only one of --target-text or --target-text-file.")

    if args.target_text_file:
        target_text_path = Path(args.target_text_file)
        if not target_text_path.exists():
            raise FileNotFoundError(f"Target text file not found: {target_text_path}")
        target_text = target_text_path.read_text(encoding="utf-8").strip()
        if not target_text:
            raise ValueError(f"Target text file is empty: {target_text_path}")
    else:
        target_text = args.target_text.strip()

    if args.device.startswith("mps") and args.dtype == "float16":
        print(
            "Warning: float16 on MPS can be numerically unstable for this model. "
            "If generation fails with inf/nan probabilities, retry with --dtype float32."
        )

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    attn_impl = "flash_attention_2" if args.flash_attn else None
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=dtype_map[args.dtype],
        attn_implementation=attn_impl,
    )

    try:
        wavs, sample_rate = model.generate_voice_clone(
            text=target_text,
            language=args.language,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "probability tensor contains either `inf`, `nan` or element < 0" in message:
            raise RuntimeError(
                "Generation became numerically unstable. On Apple MPS this is common with "
                "--dtype float16. Retry with --dtype float32, and make sure you pass actual "
                "text (or use --target-text-file) rather than a file path in --target-text."
            ) from exc
        raise

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), wavs[0], sample_rate)
    print(f"Wrote cloned audio to: {output_path}")
    return 0
