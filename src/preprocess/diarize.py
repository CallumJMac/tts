#!/usr/bin/env python3
"""Speaker diarization preprocessing for voice cloning.

Isolates the primary speaker from a multi-speaker recording, splits into
utterances at silence boundaries, optionally transcribes with Whisper, and
writes a manifest.json compatible with src.experiment.ref_pool.load_manifest().

Requires pyannote-audio (local-only, not in requirements.txt):
    pip install pyannote-audio
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from src.utils.media_tools import transcribe_with_whisper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run speaker diarization on a WAV file, isolate the primary speaker, "
            "split into utterances, and optionally transcribe with Whisper."
        ),
    )
    parser.add_argument(
        "input",
        help="Path to input WAV file.",
    )
    parser.add_argument(
        "--speaker-name",
        default=None,
        help="Name for output filenames (default: parent directory name).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <input_dir>/segments/).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Min utterance duration in seconds to keep (default: 1.0).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Max utterance duration in seconds (default: 30.0).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.1,
        help="Seconds of padding around segment boundaries (default: 0.1).",
    )
    parser.add_argument(
        "--silence-thresh",
        type=float,
        default=30,
        help="top_db threshold for librosa silence detection (default: 30).",
    )
    parser.add_argument(
        "--min-energy-db",
        type=float,
        default=-40.0,
        help=(
            "Min RMS energy in dB; filters near-silent utterances "
            "(default: -40.0)."
        ),
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Output sample rate in Hz (default: 16000).",
    )
    parser.add_argument(
        "--stitch",
        action="store_true",
        help="Also produce a single concatenated WAV of all segments.",
    )
    parser.add_argument(
        "--stitch-pause-ms",
        type=int,
        default=300,
        help="Silence gap in stitched output in ms (default: 300).",
    )
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="Transcribe each segment with Whisper.",
    )
    parser.add_argument(
        "--whisper-model",
        default="turbo",
        help="Whisper model size (default: turbo).",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token (default: reads HF_TOKEN env var).",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Compute device (default: mps).",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Expected speaker count hint for pyannote (default: auto-detect).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rms_db(audio: np.ndarray) -> float:
    """RMS energy in decibels, with a floor to avoid log(0)."""
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    return float(20 * np.log10(max(rms, 1e-10)))


def _merge_intervals(
    intervals: list[tuple[float, float]],
    gap: float,
) -> list[tuple[float, float]]:
    """Merge intervals that are closer than *gap* seconds."""
    if not intervals:
        return []
    merged: list[tuple[float, float]] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _silence_split_segment(
    audio: np.ndarray,
    sr: int,
    top_db: float,
    padding_samples: int,
    total_len: int,
) -> list[tuple[int, int]]:
    """Split a single audio segment at silence boundaries.

    Returns a list of (start_sample, end_sample) tuples within *audio*.
    """
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return [(0, len(audio))]

    result: list[tuple[int, int]] = []
    for onset, offset in intervals:
        s = max(0, onset - padding_samples)
        e = min(len(audio), offset + padding_samples)
        result.append((s, e))

    return _merge_sample_intervals(result)


def _merge_sample_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge overlapping sample-level intervals."""
    if not intervals:
        return []
    merged: list[tuple[int, int]] = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # -- dependency check (pyannote is local-only, not in requirements.txt) --
    try:
        from pyannote.audio import Pipeline  # noqa: F811
    except ModuleNotFoundError:
        print(
            "Error: pyannote-audio is not installed.\n"
            "Install with: pip install pyannote-audio"
        )
        return 2

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    speaker_name = args.speaker_name or input_path.parent.name
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_path.parent / "segments"
    )
    sr = args.sample_rate

    if output_dir.exists() and not args.overwrite:
        existing = list(output_dir.glob(f"{speaker_name}_*.wav"))
        if existing:
            print(
                f"[skip] Output dir already has {len(existing)} segment(s). "
                "Use --overwrite to replace."
            )
            return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load audio ------------------------------------------------
    print(f"Loading {input_path} ...")
    audio, _ = librosa.load(str(input_path), sr=sr, mono=True)
    duration_total = len(audio) / sr
    print(f"[ok] Loaded {duration_total:.1f}s audio at {sr} Hz")

    # ---- Step 2: Run diarization -------------------------------------------
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print(
            "Error: HuggingFace token required. Set HF_TOKEN env var or pass --hf-token.\n"
            "Get a free token at https://huggingface.co/settings/tokens"
        )
        return 2

    # pyannote has limited MPS support; fall back to CPU
    device_str = "cpu" if args.device.startswith("mps") else args.device
    device = torch.device(device_str)
    print(f"Running diarization (device={device}) ...")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    pipeline.to(device)

    diarize_kwargs: dict = {}
    if args.num_speakers is not None:
        diarize_kwargs["num_speakers"] = args.num_speakers

    raw_output = pipeline(str(input_path), **diarize_kwargs)
    # Pipeline returns DiarizeOutput; use .speaker_diarization (Annotation)
    diarization = (
        raw_output.speaker_diarization
        if hasattr(raw_output, "speaker_diarization")
        else raw_output
    )
    print("[ok] Diarization complete")

    # ---- Step 3: Identify primary speaker ----------------------------------
    speaker_time: dict[str, float] = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        dur = turn.end - turn.start
        speaker_time[speaker] = speaker_time.get(speaker, 0.0) + dur

    if not speaker_time:
        print("[fail] No speakers detected.")
        return 1

    print("\nSpeaker summary:")
    for spk, total in sorted(speaker_time.items(), key=lambda x: -x[1]):
        pct = 100 * total / duration_total
        print(f"  {spk}: {total:.1f}s ({pct:.1f}%)")

    primary_speaker = max(
        speaker_time, key=speaker_time.get  # type: ignore[arg-type]
    )
    print(
        f"\nPrimary speaker: {primary_speaker} "
        f"({speaker_time[primary_speaker]:.1f}s)"
    )

    # ---- Step 4: Extract and merge primary speaker turns -------------------
    merge_gap = 2.0 * args.padding
    raw_turns: list[tuple[float, float]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker == primary_speaker:
            raw_turns.append((turn.start, turn.end))

    raw_turns.sort(key=lambda x: x[0])
    turns = _merge_intervals(raw_turns, gap=merge_gap)
    print(f"[ok] {len(raw_turns)} raw turns merged into {len(turns)} turns")

    # ---- Step 5: Split turns into utterances at silence boundaries ---------
    padding_samples = int(args.padding * sr)
    utterances: list[tuple[int, int]] = []  # (start_sample, end_sample) in full audio

    for turn_start, turn_end in turns:
        s = max(0, int(turn_start * sr) - padding_samples)
        e = min(len(audio), int(turn_end * sr) + padding_samples)
        turn_audio = audio[s:e]

        sub_intervals = _silence_split_segment(
            turn_audio,
            sr=sr,
            top_db=args.silence_thresh,
            padding_samples=padding_samples,
            total_len=len(turn_audio),
        )
        for sub_s, sub_e in sub_intervals:
            utterances.append((s + sub_s, s + sub_e))

    print(f"[ok] {len(turns)} turns split into {len(utterances)} utterances")

    # ---- Step 6: Filter utterances -----------------------------------------
    kept: list[tuple[int, int]] = []
    skipped_short = 0
    skipped_long = 0
    skipped_quiet = 0

    for utt_s, utt_e in utterances:
        utt_audio = audio[utt_s:utt_e]
        dur = len(utt_audio) / sr
        energy = _rms_db(utt_audio)

        if dur < args.min_duration:
            skipped_short += 1
            continue
        if dur > args.max_duration:
            skipped_long += 1
            continue
        if energy < args.min_energy_db:
            skipped_quiet += 1
            continue
        kept.append((utt_s, utt_e))

    print(
        f"[ok] Kept {len(kept)} utterances "
        f"(skipped: {skipped_short} short, {skipped_long} long, {skipped_quiet} quiet)"
    )

    if not kept:
        print("[fail] No utterances survived filtering.")
        return 1

    # ---- Step 7: Export utterances -----------------------------------------
    manifest_entries: list[dict] = []
    wav_paths: list[Path] = []

    for idx, (utt_s, utt_e) in enumerate(kept):
        utt_id = f"{speaker_name}_{idx:03d}"
        wav_name = f"{utt_id}.wav"
        wav_path = output_dir / wav_name

        utt_audio = audio[utt_s:utt_e]
        sf.write(str(wav_path), utt_audio, sr, subtype="PCM_16")
        wav_paths.append(wav_path)

        manifest_entries.append({
            "id": utt_id,
            "speaker_id": speaker_name,
            "path": str(wav_path),
            "sample_rate": sr,
            "duration": round(len(utt_audio) / sr, 4),
            "start_time": round(utt_s / sr, 4),
            "end_time": round(utt_e / sr, 4),
            "energy_db": round(_rms_db(utt_audio), 2),
            "text_normalized": "",
            "text_path": "",
        })

    print(f"[ok] Exported {len(kept)} utterances to {output_dir}")

    # ---- Step 8: Stitch (optional) -----------------------------------------
    if args.stitch:
        pause_samples = int(args.stitch_pause_ms / 1000.0 * sr)
        silence = np.zeros(pause_samples, dtype=audio.dtype)

        chunks: list[np.ndarray] = []
        for i, (utt_s, utt_e) in enumerate(kept):
            if i > 0:
                chunks.append(silence)
            chunks.append(audio[utt_s:utt_e])

        stitched = np.concatenate(chunks)
        stitched_path = output_dir / f"{speaker_name}_stitched.wav"
        sf.write(str(stitched_path), stitched, sr, subtype="PCM_16")
        stitched_dur = len(stitched) / sr
        print(
            f"[ok] Stitched {len(kept)} utterances -> {stitched_path} "
            f"({stitched_dur:.1f}s)"
        )

    # ---- Step 9: Transcribe (optional) -------------------------------------
    if args.transcribe:
        transcript_dir = output_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)

        ok_count = 0
        for idx, wav_path in enumerate(wav_paths):
            try:
                text_path = transcribe_with_whisper(
                    audio_path=wav_path,
                    text_dir=transcript_dir,
                    model=args.whisper_model,
                    language="en",
                )
                transcript = text_path.read_text(encoding="utf-8").strip()
                manifest_entries[idx]["text_normalized"] = transcript
                manifest_entries[idx]["text_path"] = str(text_path)
                ok_count += 1
            except Exception as exc:
                print(f"[fail] Transcribe {wav_path.name}: {exc}")

        print(f"[ok] Transcribed {ok_count}/{len(wav_paths)} utterances")

    # ---- Step 10: Write manifest -------------------------------------------
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2, ensure_ascii=False)

    print(f"[ok] Manifest written to {manifest_path}")
    print(f"\nDone: {len(kept)} utterances exported to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
