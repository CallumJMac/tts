"""Few-shot voice cloning experiment runner.

Orchestrates TTS generation and evaluation across the experiment matrix
defined by approach, num_refs, selection strategy, speakers, and seeds.

Usage:
    python scripts/run_fewshot.py --manifest data/libritts_r_aligned/manifest.json

For a quick pilot (Phase 0-1):
    python scripts/run_fewshot.py \
        --manifest data/libritts_r_aligned/manifest.json \
        --speakers 1188 \
        --approaches single_baseline concat_audio \
        --num-refs 1 2 \
        --strategies random \
        --seeds 42 \
        --held-out-targets 1
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run few-shot voice cloning experiments.",
    )
    parser.add_argument(
        "--manifest",
        default="data/libritts_r_aligned/manifest.json",
        help="Path to LibriTTS-R manifest.json.",
    )
    parser.add_argument(
        "--speakers",
        nargs="*",
        default=None,
        help="Speaker IDs to test (default: all in manifest).",
    )
    parser.add_argument(
        "--approaches",
        nargs="+",
        default=["single_baseline", "concat_audio"],
        choices=["single_baseline", "concat_audio", "concat_code", "embed_avg"],
        help="Approaches to test.",
    )
    parser.add_argument(
        "--num-refs",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Number of reference clips per approach.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["random"],
        choices=["random", "longest"],
        help="Selection strategies.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds for TTS generation.",
    )
    parser.add_argument(
        "--held-out-targets",
        type=int,
        default=5,
        help="Number of held-out targets per speaker to evaluate.",
    )
    parser.add_argument(
        "--held-out-per-speaker",
        type=int,
        default=5,
        help="Total held-out clips per speaker in the split.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/fewshot",
        help="Base output directory.",
    )
    parser.add_argument(
        "--results-csv",
        default=None,
        help="Path to results CSV (default: {output-dir}/results.csv).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Qwen3-TTS model name or path.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Device: mps, cpu, cuda:0.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype.",
    )
    parser.add_argument(
        "--language",
        default="English",
        help="Language for TTS generation.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation (generate audio only).",
    )
    parser.add_argument(
        "--skip-speechbertscore",
        action="store_true",
        help="Skip SpeechBERTScore (saves memory and time).",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        help="Enable FlashAttention-2 (CUDA only).",
    )
    return parser


def _load_tts_model(args: argparse.Namespace):
    """Load Qwen3-TTS model once."""
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    attn_impl = "flash_attention_2" if args.flash_attn else None

    print(f"Loading TTS model: {args.model} on {args.device} ({args.dtype})")
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=dtype_map[args.dtype],
        attn_implementation=attn_impl,
    )
    return model


def _generate_single(
    model,
    target_text: str,
    ref_audio,
    ref_text: Optional[str],
    voice_clone_prompt=None,
    language: str = "English",
) -> tuple[np.ndarray, int]:
    """Run a single TTS generation, returning (wav, sample_rate)."""
    import soundfile as sf

    if voice_clone_prompt is not None:
        wavs, sr = model.generate_voice_clone(
            text=target_text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
        )
    else:
        wavs, sr = model.generate_voice_clone(
            text=target_text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
    return wavs[0], sr


class EvalModels:
    """Holds pre-loaded evaluation models to avoid reloading per run."""

    def __init__(self, device: str, skip_speechbertscore: bool = False):
        import torch
        import whisper
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        from src.evaluation.evaluate import _WAVLM_SV_MODEL, _resolve_device

        self.device = _resolve_device(device)

        # UTMOS
        print("  Loading UTMOS model ...")
        self.utmos_predictor = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True,
        )
        try:
            self.utmos_predictor = self.utmos_predictor.to(self.device)
        except RuntimeError:
            self.utmos_predictor = self.utmos_predictor.to("cpu")
        self.utmos_predictor.eval()

        # WavLM speaker similarity
        print("  Loading WavLM speaker model ...")
        self.sv_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            _WAVLM_SV_MODEL
        )
        self.sv_model = WavLMForXVector.from_pretrained(_WAVLM_SV_MODEL)
        try:
            self.sv_model = self.sv_model.to(self.device)
        except RuntimeError:
            self.sv_model = self.sv_model.to("cpu")
        self.sv_model.eval()

        # Whisper
        print("  Loading Whisper turbo model ...")
        self.whisper_model = whisper.load_model("turbo")

        # SpeechBERTScore (optional)
        self.sbs_scorer = None
        if not skip_speechbertscore:
            print("  Loading WavLM-Large for SpeechBERTScore ...")
            from discrete_speech_metrics import SpeechBERTScore

            use_gpu = self.device.startswith("cuda")
            self.sbs_scorer = SpeechBERTScore(
                sr=16000, model_type="wavlm-large", layer=14, use_gpu=use_gpu
            )

        print("  Eval models loaded.")

    def compute_utmos(self, wav_16k: np.ndarray) -> float:
        import torch

        tensor = torch.from_numpy(wav_16k).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = self.utmos_predictor(tensor, sr=16000)
        return round(float(score.item()), 3)

    def compute_speaker_similarity(
        self, ref_wav_16k: np.ndarray, gen_wav_16k: np.ndarray
    ) -> float:
        import torch

        inputs = self.sv_feature_extractor(
            [ref_wav_16k, gen_wav_16k],
            sampling_rate=16000,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            embeddings = self.sv_model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        similarity = torch.nn.CosineSimilarity(dim=-1)(
            embeddings[0:1], embeddings[1:2]
        )
        return round(float(similarity.item()), 4)

    def compute_wer(self, gen_wav_path: Path, target_text: str) -> dict:
        from jiwer import cer, wer

        result = self.whisper_model.transcribe(str(gen_wav_path), language="en")
        hypothesis = result["text"].strip()
        reference = target_text.strip()
        return {
            "wer": round(wer(reference, hypothesis), 4),
            "cer": round(cer(reference, hypothesis), 4),
            "transcript": hypothesis,
        }

    def compute_speech_bert_score(
        self, ref_wav_16k: np.ndarray, gen_wav_16k: np.ndarray
    ) -> dict[str, float]:
        if self.sbs_scorer is None:
            return {}
        precision, recall, f1 = self.sbs_scorer.score(ref_wav_16k, gen_wav_16k)
        return {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        }


def _evaluate_single(
    gen_path: Path,
    ref_audio_path: str,
    target_text: str,
    eval_models: EvalModels,
    skip_speechbertscore: bool = False,
) -> dict:
    """Run evaluation metrics on a single generated file using pre-loaded models."""
    import librosa

    gen_wav, _ = librosa.load(str(gen_path), sr=16000, mono=True)
    ref_wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)

    results = {}

    # UTMOS
    try:
        results["utmos"] = eval_models.compute_utmos(gen_wav)
    except Exception as e:
        results["utmos"] = None
        results["utmos_error"] = str(e)

    # Speaker Similarity
    try:
        results["speaker_sim"] = eval_models.compute_speaker_similarity(ref_wav, gen_wav)
    except Exception as e:
        results["speaker_sim"] = None
        results["speaker_sim_error"] = str(e)

    # WER
    try:
        wer_results = eval_models.compute_wer(gen_path, target_text)
        results["wer"] = wer_results["wer"]
        results["cer"] = wer_results["cer"]
        results["transcript"] = wer_results["transcript"]
    except Exception as e:
        results["wer"] = None
        results["cer"] = None
        results["wer_error"] = str(e)

    # SpeechBERTScore (optional — heavy)
    if not skip_speechbertscore:
        try:
            sbs = eval_models.compute_speech_bert_score(ref_wav, gen_wav)
            if sbs:
                results["sbs_precision"] = sbs["precision"]
                results["sbs_recall"] = sbs["recall"]
                results["sbs_f1"] = sbs["f1"]
        except Exception as e:
            results["sbs_f1"] = None
            results["sbs_error"] = str(e)

    return results


def main() -> int:
    args = build_parser().parse_args()

    from src.experiment.combiners import (
        ConcatAudioCombiner,
        ConcatCodeCombiner,
        EmbedAvgCombiner,
    )
    from src.experiment.ref_pool import build_speaker_pools

    import soundfile as sf

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = Path(args.results_csv) if args.results_csv else output_dir / "results.csv"

    # Build speaker pools
    print(f"Loading manifest: {args.manifest}")
    pools = build_speaker_pools(
        args.manifest,
        held_out_per_speaker=args.held_out_per_speaker,
    )

    speaker_ids = args.speakers or sorted(pools.keys())
    print(f"Speakers: {speaker_ids}")
    for sid in speaker_ids:
        if sid not in pools:
            print(f"Error: Speaker {sid} not in manifest.")
            return 1
        pool = pools[sid]
        print(f"  {sid}: {len(pool.refs)} refs, {len(pool.held_out)} held-out")

    # Load TTS model once
    model = _load_tts_model(args)

    # Load eval models once (avoids reloading UTMOS/WavLM/Whisper per run)
    eval_models = None
    if not args.skip_eval:
        print("Loading evaluation models (one-time) ...")
        eval_models = EvalModels(
            device=args.device,
            skip_speechbertscore=args.skip_speechbertscore,
        )

    # Prepare combiners
    concat_audio_combiner = ConcatAudioCombiner()
    concat_code_combiner = ConcatCodeCombiner()
    embed_avg_combiner = EmbedAvgCombiner()

    # Build experiment matrix
    runs = []
    for speaker_id in speaker_ids:
        pool = pools[speaker_id]
        targets = pool.held_out[: args.held_out_targets]

        for target_idx, target in enumerate(targets):
            for approach in args.approaches:
                nums = args.num_refs if approach != "single_baseline" else [1]
                strategies = args.strategies if approach != "single_baseline" else ["random"]

                for n_refs in nums:
                    if n_refs > len(pool.refs):
                        print(
                            f"  Skipping {approach} n={n_refs} for speaker {speaker_id} "
                            f"(only {len(pool.refs)} refs available)"
                        )
                        continue

                    for strategy in strategies:
                        for seed in args.seeds:
                            runs.append(
                                {
                                    "speaker_id": speaker_id,
                                    "target_idx": target_idx,
                                    "target_id": target.id,
                                    "target_text": target.text,
                                    "target_audio": target.path,
                                    "approach": approach,
                                    "n_refs": n_refs,
                                    "strategy": strategy,
                                    "seed": seed,
                                }
                            )

    print(f"\nTotal runs: {len(runs)}")
    print(f"Output directory: {output_dir}")
    print(f"Results CSV: {results_csv}")
    print()

    # CSV header
    fieldnames = [
        "speaker_id",
        "target_id",
        "approach",
        "n_refs",
        "strategy",
        "seed",
        "output_path",
        "tts_time_s",
        "eval_time_s",
        "utmos",
        "speaker_sim",
        "wer",
        "cer",
        "sbs_f1",
        "transcript",
    ]
    csv_exists = results_csv.exists()
    csv_file = open(results_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
    if not csv_exists:
        writer.writeheader()

    try:
        for i, run in enumerate(runs):
            speaker_id = run["speaker_id"]
            pool = pools[speaker_id]
            approach = run["approach"]
            n_refs = run["n_refs"]
            strategy = run["strategy"]
            seed = run["seed"]
            target_text = run["target_text"]
            target_audio = run["target_audio"]

            tag = f"{approach}_{n_refs}refs_{strategy}_seed{seed}"
            wav_name = f"{speaker_id}_{run['target_id']}_{tag}.wav"
            wav_path = output_dir / speaker_id / wav_name
            wav_path.parent.mkdir(parents=True, exist_ok=True)

            print(
                f"[{i + 1}/{len(runs)}] speaker={speaker_id} target={run['target_id']} "
                f"approach={approach} n_refs={n_refs} strategy={strategy} seed={seed}"
            )

            # Select references
            selected_refs = pool.select(n_refs, strategy=strategy, seed=seed)
            ref_ids = [r.id for r in selected_refs]
            print(f"  Selected refs: {ref_ids}")

            # Generate
            t0 = time.time()
            try:
                if approach == "single_baseline":
                    ref = selected_refs[0]
                    wav, sr = _generate_single(
                        model,
                        target_text=target_text,
                        ref_audio=ref.path,
                        ref_text=ref.text,
                        language=args.language,
                    )

                elif approach == "concat_audio":
                    combined = concat_audio_combiner.combine(selected_refs)
                    wav, sr = _generate_single(
                        model,
                        target_text=target_text,
                        ref_audio=(combined.audio_array, combined.audio_sr),
                        ref_text=combined.text,
                        language=args.language,
                    )

                elif approach == "concat_code":
                    combined = concat_code_combiner.combine(selected_refs, model)
                    wav, sr = _generate_single(
                        model,
                        target_text=target_text,
                        ref_audio=None,
                        ref_text=None,
                        voice_clone_prompt=combined.voice_clone_prompt,
                        language=args.language,
                    )

                elif approach == "embed_avg":
                    combined = embed_avg_combiner.combine(selected_refs, model)
                    wav, sr = _generate_single(
                        model,
                        target_text=target_text,
                        ref_audio=None,
                        ref_text=None,
                        voice_clone_prompt=combined.voice_clone_prompt,
                        language=args.language,
                    )
                else:
                    print(f"  Unknown approach: {approach}, skipping")
                    continue

            except RuntimeError as exc:
                msg = str(exc)
                if "probability tensor" in msg:
                    print(f"  TTS FAILED (numerical instability): {msg[:100]}")
                    continue
                raise

            tts_time = time.time() - t0
            sf.write(str(wav_path), wav, sr)
            print(f"  Generated: {wav_path} ({tts_time:.1f}s)")

            # Evaluate
            eval_results = {}
            eval_time = 0.0
            if not args.skip_eval and eval_models is not None:
                t1 = time.time()
                eval_results = _evaluate_single(
                    wav_path,
                    ref_audio_path=target_audio,  # compare against ground truth
                    target_text=target_text,
                    eval_models=eval_models,
                    skip_speechbertscore=args.skip_speechbertscore,
                )
                eval_time = time.time() - t1
                print(
                    f"  Eval ({eval_time:.1f}s): "
                    f"UTMOS={eval_results.get('utmos')} "
                    f"SIM={eval_results.get('speaker_sim')} "
                    f"WER={eval_results.get('wer')}"
                )

            # Write row
            row = {
                "speaker_id": speaker_id,
                "target_id": run["target_id"],
                "approach": approach,
                "n_refs": n_refs,
                "strategy": strategy,
                "seed": seed,
                "output_path": str(wav_path),
                "tts_time_s": round(tts_time, 2),
                "eval_time_s": round(eval_time, 2),
                **{k: v for k, v in eval_results.items() if k in fieldnames},
            }
            writer.writerow(row)
            csv_file.flush()

    finally:
        csv_file.close()

    print(f"\nDone. Results saved to: {results_csv}")
    return 0
