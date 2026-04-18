"""Generate demo samples comparing single baseline, concat, and embed avg strategies.

Produces three WAV files in samples/demo/ for the same speaker + target text,
making the Pareto trade-off audible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf
import torch

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

from src.experiment.combiners import ConcatAudioCombiner, EmbedAvgCombiner
from src.experiment.ref_pool import RefItem


def make_ref_item(path: str, text: str) -> RefItem:
    info = sf.info(path)
    p = Path(path)
    return RefItem(
        id=p.stem,
        speaker_id=p.stem,
        text=text,
        path=path,
        sample_rate=info.samplerate,
        duration=info.duration,
    )


TARGET_TEXT = (
    "When multiple reference utterances are available, how should they be combined? "
    "We find a clear trade-off: concatenating references maximises speaker identity, "
    "while averaging embeddings maximises naturalness."
)


def load_model(model_id: str, device: str, dtype: str) -> Qwen3TTSModel:
    torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[dtype]
    print(f"Loading {model_id} on {device} ({dtype})...")
    return Qwen3TTSModel.from_pretrained(model_id, device_map=device, dtype=torch_dtype)


def save_wav(array: np.ndarray, sr: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), array, sr)
    print(f"  Saved: {path}")


def generate_baseline(model, ref_audio: str, ref_text: str, target: str,
                      device: str, out: Path) -> None:
    print("\n[1/3] Baseline — single reference, ICL prompt")
    wavs, sr = model.generate_voice_clone(
        text=target,
        ref_audio=ref_audio,
        ref_text=ref_text,
        language="English",
    )
    save_wav(wavs[0], sr, out)


def generate_concat(model, refs: list[RefItem], target: str, out: Path) -> None:
    print("\n[2/3] Concat — 3 references, longest selection, ICL prompt")
    combiner = ConcatAudioCombiner()
    combined = combiner.combine(refs)
    wavs, sr = model.generate_voice_clone(
        text=target,
        ref_audio=(combined.audio_array, combined.audio_sr),
        ref_text=combined.text,
        language="English",
    )
    save_wav(wavs[0], sr, out)


def generate_embed(model, refs: list[RefItem], target: str, out: Path) -> None:
    print("\n[3/3] Embed avg — 3 references, x-vector only")
    combiner = EmbedAvgCombiner()
    combined = combiner.combine(refs, model=model)
    prompt = combined.voice_clone_prompt
    wavs, sr = model.generate_voice_clone(
        text=target,
        voice_clone_prompt=prompt,
        language="English",
    )
    save_wav(wavs[0], sr, out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate strategy comparison demo samples")
    parser.add_argument("--ref-audio", required=True, help="Primary reference WAV path")
    parser.add_argument("--ref-text", required=True, help="Primary reference text path")
    parser.add_argument("--extra-refs", nargs="*", default=[],
                        help="Additional reference WAV paths for multi-ref strategies (at least 2)")
    parser.add_argument("--extra-ref-texts", nargs="*", default=[],
                        help="Text files for extra refs (same order)")
    parser.add_argument("--target-text", default=TARGET_TEXT,
                        help="Text to synthesize")
    parser.add_argument("--output-dir", default="samples/demo", help="Output directory")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    model = load_model(args.model, args.device, args.dtype)

    ref_text_str = Path(args.ref_text).read_text().strip()

    # Build ref items for multi-ref strategies
    refs: list[RefItem] = [make_ref_item(args.ref_audio, ref_text_str)]
    for wav, txt in zip(args.extra_refs, args.extra_ref_texts):
        refs.append(make_ref_item(wav, Path(txt).read_text().strip()))

    if len(refs) < 3:
        print(f"Warning: only {len(refs)} refs provided; concat/embed will use all available.")

    # Use up to 3 refs for multi-ref strategies
    multi_refs = refs[:3]

    generate_baseline(model, args.ref_audio, ref_text_str, args.target_text,
                      args.device, out_dir / "01_baseline_single.wav")
    generate_concat(model, multi_refs, args.target_text,
                    out_dir / "02_concat_longest_3.wav")
    generate_embed(model, multi_refs, args.target_text,
                   out_dir / "03_embed_avg_3.wav")

    print(f"\nDone. 3 samples saved to {out_dir}/")
    print("Listen in order to hear the fidelity vs naturalness trade-off.")


if __name__ == "__main__":
    main()
