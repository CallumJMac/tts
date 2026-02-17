from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch
import whisper
from discrete_speech_metrics import SpeechBERTScore
from jiwer import cer, wer
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate TTS voice clone quality with multiple metrics.",
    )
    parser.add_argument(
        "generated",
        help="Path to generated (cloned) audio file.",
    )
    parser.add_argument(
        "--ref-audio",
        help="Path to reference speaker audio file.",
    )
    parser.add_argument(
        "--target-text",
        help="Target text string for WER computation.",
    )
    parser.add_argument(
        "--target-text-file",
        help="Path to target text file for WER computation.",
    )
    parser.add_argument(
        "--skip-utmos",
        action="store_true",
        help="Skip UTMOS mean opinion score prediction.",
    )
    parser.add_argument(
        "--skip-speaker-sim",
        action="store_true",
        help="Skip WavLM speaker similarity.",
    )
    parser.add_argument(
        "--skip-wer",
        action="store_true",
        help="Skip Whisper WER computation.",
    )
    parser.add_argument(
        "--skip-speechbertscore",
        action="store_true",
        help="Skip SpeechBERTScore computation.",
    )
    parser.add_argument(
        "--whisper-model",
        default="turbo",
        help="Whisper model size for transcription (default: turbo).",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Compute device: mps, cpu, cuda:0 (default: mps).",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format (default: table).",
    )
    return parser


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _try_import(module_name: str, pip_hint: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ModuleNotFoundError:
        print(f"  [skip] {module_name} not installed. Install with: {pip_hint}")
        return False


def _resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return requested
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested != "cpu":
        print(f"  Warning: {requested} not available, falling back to cpu.")
    return "cpu"


def _load_audio_16k(path: Path) -> np.ndarray:
    wav, _ = librosa.load(str(path), sr=16000, mono=True)
    return wav


# ---------------------------------------------------------------------------
# Metric: UTMOS
# ---------------------------------------------------------------------------

def compute_utmos(wav_16k: np.ndarray, device: str) -> float:
    """Predict Mean Opinion Score using UTMOS22 Strong (1.0–5.0 scale)."""
    print("  Loading UTMOS model (first run downloads ~400 MB) ...")
    predictor = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0",
        "utmos22_strong",
        trust_repo=True,
    )
    try:
        predictor = predictor.to(device)
    except RuntimeError:
        predictor = predictor.to("cpu")
        device = "cpu"
    predictor.eval()

    tensor = torch.from_numpy(wav_16k).unsqueeze(0).to(device)
    with torch.no_grad():
        score = predictor(tensor, sr=16000)
    return round(float(score.item()), 3)


# ---------------------------------------------------------------------------
# Metric: WavLM speaker similarity
# ---------------------------------------------------------------------------

_WAVLM_SV_MODEL = "microsoft/wavlm-base-plus-sv"


def compute_speaker_similarity(
    ref_wav_16k: np.ndarray,
    gen_wav_16k: np.ndarray,
    device: str,
) -> float:
    """Cosine similarity of WavLM speaker embeddings (-1.0 to 1.0)."""
    print("  Loading WavLM speaker model (first run downloads ~360 MB) ...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(_WAVLM_SV_MODEL)
    model = WavLMForXVector.from_pretrained(_WAVLM_SV_MODEL)

    try:
        model = model.to(device)
    except RuntimeError:
        model = model.to("cpu")
        device = "cpu"
    model.eval()

    inputs = feature_extractor(
        [ref_wav_16k, gen_wav_16k],
        sampling_rate=16000,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        embeddings = model(**inputs).embeddings

    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    similarity = torch.nn.CosineSimilarity(dim=-1)(
        embeddings[0:1], embeddings[1:2],
    )
    return round(float(similarity.item()), 4)


# ---------------------------------------------------------------------------
# Metric: WER via Whisper
# ---------------------------------------------------------------------------

def compute_wer(
    gen_wav_path: Path,
    target_text: str,
    whisper_model_size: str,
) -> dict[str, object]:
    """Transcribe generated audio and compute WER/CER against target text."""
    print(f"  Loading Whisper {whisper_model_size} model ...")
    model = whisper.load_model(whisper_model_size)
    result = model.transcribe(str(gen_wav_path), language="en")
    hypothesis = result["text"].strip()
    reference = target_text.strip()

    return {
        "wer": round(wer(reference, hypothesis), 4),
        "cer": round(cer(reference, hypothesis), 4),
        "transcript": hypothesis,
    }


# ---------------------------------------------------------------------------
# Metric: SpeechBERTScore
# ---------------------------------------------------------------------------

def compute_speech_bert_score(
    ref_wav_16k: np.ndarray,
    gen_wav_16k: np.ndarray,
    use_gpu: bool,
) -> dict[str, float]:
    """SpeechBERTScore using WavLM-Large layer 14 features."""
    print("  Loading WavLM-Large for SpeechBERTScore (first run downloads ~1.2 GB) ...")
    metrics = SpeechBERTScore(
        sr=16000,
        model_type="wavlm-large",
        layer=14,
        use_gpu=use_gpu,
    )
    precision, recall, f1 = metrics.score(ref_wav_16k, gen_wav_16k)
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_results_table(
    results: dict[str, str],
    gen_path: Path,
    ref_path: Path | None,
) -> None:
    print()
    print("Evaluation Results")
    print("=" * 55)
    print(f"  Generated:  {gen_path}")
    if ref_path:
        print(f"  Reference:  {ref_path}")
    print()

    key_width = max(len(k) for k in results)
    print(f"  {'Metric':<{key_width}}  Value")
    print(f"  {'-' * key_width}  {'-' * 30}")
    for key, value in results.items():
        display = value if len(value) <= 70 else value[:67] + "..."
        print(f"  {key:<{key_width}}  {display}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = build_parser().parse_args()

    gen_path = Path(args.generated)
    if not gen_path.exists():
        print(f"Error: Generated audio not found: {gen_path}")
        return 1

    ref_path = Path(args.ref_audio) if args.ref_audio else None
    if ref_path and not ref_path.exists():
        print(f"Error: Reference audio not found: {ref_path}")
        return 1

    # Resolve target text
    target_text: str | None = None
    if args.target_text and args.target_text_file:
        print("Error: Use only one of --target-text or --target-text-file.")
        return 1
    if args.target_text:
        target_text = args.target_text.strip()
    elif args.target_text_file:
        tt_path = Path(args.target_text_file)
        if not tt_path.exists():
            print(f"Error: Target text file not found: {tt_path}")
            return 1
        target_text = tt_path.read_text(encoding="utf-8").strip()

    # Auto-skip metrics that lack required inputs
    if not args.skip_speaker_sim and ref_path is None:
        print("Warning: --ref-audio not provided, skipping speaker similarity.")
        args.skip_speaker_sim = True
    if not args.skip_speechbertscore and ref_path is None:
        print("Warning: --ref-audio not provided, skipping SpeechBERTScore.")
        args.skip_speechbertscore = True
    if not args.skip_wer and target_text is None:
        print("Warning: No target text provided, skipping WER.")
        args.skip_wer = True

    device = _resolve_device(args.device)
    results: dict[str, str] = {}

    # Lazy-load audio only when needed
    gen_wav: np.ndarray | None = None
    ref_wav: np.ndarray | None = None

    # --- UTMOS ---
    if not args.skip_utmos:
        print("[1/4] UTMOS (speech naturalness)")
        if gen_wav is None:
            gen_wav = _load_audio_16k(gen_path)
        try:
            score = compute_utmos(gen_wav, device)
            results["UTMOS (MOS 1-5)"] = str(score)
        except Exception as exc:
            results["UTMOS"] = f"error: {exc}"

    # --- Speaker Similarity ---
    if not args.skip_speaker_sim:
        print("[2/4] Speaker Similarity (WavLM)")
        if gen_wav is None:
            gen_wav = _load_audio_16k(gen_path)
        if ref_wav is None:
            ref_wav = _load_audio_16k(ref_path)
        try:
            sim = compute_speaker_similarity(ref_wav, gen_wav, device)
            results["Speaker Similarity"] = str(sim)
        except Exception as exc:
            results["Speaker Similarity"] = f"error: {exc}"

    # --- WER ---
    if not args.skip_wer:
        print("[3/4] WER (Whisper transcription)")
        if not _try_import("jiwer", "pip install jiwer"):
            results["WER"] = "skipped (jiwer not installed)"
        else:
            try:
                wer_results = compute_wer(gen_path, target_text, args.whisper_model)
                results["WER"] = f"{wer_results['wer']:.1%}"
                results["CER"] = f"{wer_results['cer']:.1%}"
                results["Transcript"] = wer_results["transcript"]
            except Exception as exc:
                results["WER"] = f"error: {exc}"

    # --- SpeechBERTScore ---
    if not args.skip_speechbertscore:
        print("[4/4] SpeechBERTScore (acoustic similarity)")
        if not _try_import(
            "discrete_speech_metrics",
            "pip install git+https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics.git",
        ):
            results["SpeechBERTScore"] = "skipped (discrete_speech_metrics not installed)"
        else:
            if gen_wav is None:
                gen_wav = _load_audio_16k(gen_path)
            if ref_wav is None:
                ref_wav = _load_audio_16k(ref_path)
            # MPS is not supported by discrete_speech_metrics; only CUDA counts as GPU
            use_gpu = device.startswith("cuda")
            try:
                sbs = compute_speech_bert_score(ref_wav, gen_wav, use_gpu)
                results["SpeechBERTScore (P)"] = str(sbs["precision"])
                results["SpeechBERTScore (R)"] = str(sbs["recall"])
                results["SpeechBERTScore (F1)"] = str(sbs["f1"])
            except Exception as exc:
                results["SpeechBERTScore"] = f"error: {exc}"

    # --- Output ---
    if not results:
        print("No metrics were computed. Check arguments and installed dependencies.")
        return 1

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        _print_results_table(results, gen_path, ref_path)

    return 0
