"""Generate WAV stimuli for the listening study on local Mac (MPS).

Generates only the specific trials needed, using locally available speakers.
Requires: Qwen3-TTS model cached, LibriTTS-R audio in data/libritts_r_aligned/audio/.
"""

import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

DATA_ROOT = Path(__file__).parent.parent / "data" / "libritts_r_aligned"
AUDIO_DIR = DATA_ROOT / "audio"
RESULTS_CSV = Path(__file__).parent.parent / "results" / "phase4_40spk" / "results.csv"
OUTPUT_DIR = Path(__file__).parent / "audio"

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
SEED = 42
N_TRIALS = 12
MAX_PER_SPEAKER = 2

CONDITIONS = {
    "single_baseline": {"approach": "single_baseline", "strategy": "random", "n_refs": 1},
    "concat_longest_3": {"approach": "concat_audio", "strategy": "longest", "n_refs": 3},
    "embed_random_3": {"approach": "embed_avg", "strategy": "random", "n_refs": 3},
}


def get_available_speakers():
    if not AUDIO_DIR.exists():
        return []
    return [d.name for d in AUDIO_DIR.iterdir() if d.is_dir() and d.name != "_cache" and list(d.glob("*.wav"))]


def get_speaker_audio(speaker_id: str) -> list[dict]:
    spk_dir = AUDIO_DIR / speaker_id
    items = []
    for wav in sorted(spk_dir.glob("*.wav")):
        txt_file = wav.with_suffix(".normalized.txt")
        if not txt_file.exists():
            txt_file = wav.with_suffix(".txt")
        text = txt_file.read_text().strip() if txt_file.exists() else ""
        info = sf.info(str(wav))
        items.append({"path": str(wav), "text": text, "duration": info.duration})
    return items


def select_trials(df: pd.DataFrame, available_speakers: list[str]) -> list[dict]:
    df = df[df["seed"] == SEED].copy()
    df = df[df["speaker_id"].astype(str).isin(available_speakers)]

    groups = []
    for (spk, tgt), grp in df.groupby(["speaker_id", "target_id"]):
        conds = {}
        for cname, cfilt in CONDITIONS.items():
            match = grp[
                (grp["approach"] == cfilt["approach"])
                & (grp["strategy"] == cfilt["strategy"])
                & (grp["n_refs"] == cfilt["n_refs"])
            ]
            if len(match) == 0:
                break
            conds[cname] = match.iloc[0]
        else:
            try:
                sim_diff = abs(
                    float(conds["concat_longest_3"]["speaker_sim"])
                    - float(conds["embed_random_3"]["speaker_sim"])
                )
                utmos_diff = abs(
                    float(conds["concat_longest_3"]["utmos"])
                    - float(conds["embed_random_3"]["utmos"])
                )
            except (ValueError, TypeError):
                continue
            groups.append({
                "speaker_id": str(spk),
                "target_id": str(tgt),
                "contrast": sim_diff + utmos_diff,
                "conditions": conds,
            })

    groups.sort(key=lambda x: x["contrast"], reverse=True)
    selected = []
    spk_count = defaultdict(int)
    for g in groups:
        if spk_count[g["speaker_id"]] >= MAX_PER_SPEAKER:
            continue
        selected.append(g)
        spk_count[g["speaker_id"]] += 1
        if len(selected) >= N_TRIALS + 1:
            break
    return selected


def find_target_text(speaker_id: str, target_id: str) -> str:
    """Find the target text from LibriTTS-R normalized text files."""
    spk_dir = AUDIO_DIR / speaker_id
    # target_id is like "134500_000038_000000" — need to match
    for txt in spk_dir.glob("*.normalized.txt"):
        if target_id in txt.stem:
            return txt.read_text().strip()
    for txt in spk_dir.glob("*.txt"):
        if target_id in txt.stem:
            return txt.read_text().strip()
    return ""


def find_ground_truth(speaker_id: str, target_id: str) -> str | None:
    """Find ground truth WAV for the target utterance."""
    spk_dir = AUDIO_DIR / speaker_id
    for wav in spk_dir.glob("*.wav"):
        if target_id in wav.stem:
            return str(wav)
    return None


def generate_samples(trials: list[dict]):
    import torch
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map=device,
        dtype=torch.float32,
    )

    for i, trial in enumerate(trials):
        spk = trial["speaker_id"]
        tgt = trial["target_id"]
        print(f"\n[Trial {i+1}/{len(trials)}] Speaker {spk}, Target {tgt}")

        spk_audio = get_speaker_audio(spk)
        if not spk_audio:
            print(f"  ERROR: No audio for speaker {spk}")
            continue

        # Find target text
        target_text = find_target_text(spk, tgt)
        if not target_text:
            print(f"  ERROR: No target text for {tgt}")
            continue

        trial_dir = OUTPUT_DIR / f"trial_{i:02d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Sort by duration for "longest" selection
        spk_audio_sorted = sorted(spk_audio, key=lambda x: x["duration"], reverse=True)

        torch.manual_seed(SEED)

        # --- 1. Single baseline: ICL, 1 ref ---
        ref = spk_audio_sorted[0]
        print(f"  [1/3] single_baseline (ICL, 1 ref)...")
        try:
            wavs, sr = model.generate_voice_clone(
                text=target_text,
                language="English",
                ref_audio=ref["path"],
                ref_text=ref["text"],
            )
            sf.write(str(trial_dir / "single_baseline.wav"), wavs[0], sr)
        except Exception as e:
            print(f"    ERROR: {e}")

        # --- 2. Concat longest 3: ICL, 3 refs concatenated ---
        refs_3 = spk_audio_sorted[:3]
        print(f"  [2/3] concat_longest_3 (ICL, 3 refs concatenated)...")
        try:
            concat_audio_data = []
            concat_text_parts = []
            for r in refs_3:
                audio_data, audio_sr = sf.read(r["path"])
                concat_audio_data.append(audio_data)
                silence = np.zeros(int(audio_sr * 0.3))
                concat_audio_data.append(silence)
                concat_text_parts.append(r["text"])

            concat_wav = np.concatenate(concat_audio_data)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, concat_wav, audio_sr)
                concat_path = tmp.name

            wavs, sr = model.generate_voice_clone(
                text=target_text,
                language="English",
                ref_audio=concat_path,
                ref_text=" ".join(concat_text_parts),
            )
            sf.write(str(trial_dir / "concat_longest_3.wav"), wavs[0], sr)
        except Exception as e:
            print(f"    ERROR: {e}")

        # --- 3. Embed avg random 3: x-vector only, 3 refs averaged ---
        rng = np.random.default_rng(SEED)
        rand_refs = list(spk_audio)
        rng.shuffle(rand_refs)
        refs_rand_3 = rand_refs[:3]
        print(f"  [3/3] embed_random_3 (x-vector, 3 refs averaged)...")
        try:
            import torch as _torch
            from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

            # Extract embeddings from each ref
            prompt_items = []
            for r in refs_rand_3:
                items = model.create_voice_clone_prompt(
                    ref_audio=r["path"],
                    ref_text=r["text"],
                    x_vector_only_mode=True,
                )
                prompt_items.append(items[0])

            # Average embeddings
            embeddings = _torch.stack([item.ref_spk_embedding for item in prompt_items])
            avg_embedding = embeddings.mean(dim=0)

            # Create combined prompt item
            combined_item = VoiceClonePromptItem(
                ref_code=None,
                ref_spk_embedding=avg_embedding,
                x_vector_only_mode=True,
                icl_mode=False,
            )

            wavs, sr = model.generate_voice_clone(
                text=target_text,
                language="English",
                voice_clone_prompt=[combined_item],
            )
            sf.write(str(trial_dir / "embed_random_3.wav"), wavs[0], sr)
        except Exception as e:
            print(f"    ERROR: {e}")

        # --- 4. Reference: ground truth ---
        gt_path = find_ground_truth(spk, tgt)
        if gt_path:
            shutil.copy2(gt_path, trial_dir / "reference.wav")
        else:
            shutil.copy2(ref["path"], trial_dir / "reference.wav")

        # --- 5. Anchor: time-stretched 0.7x ---
        ref_wav = trial_dir / "reference.wav"
        anchor_path = trial_dir / "anchor.wav"
        try:
            subprocess.run(
                ["sox", str(ref_wav), str(anchor_path), "tempo", "0.7"],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError:
            shutil.copy2(ref_wav, anchor_path)

        print(f"  Done: {trial_dir}")


def main():
    print("=== Listening Study Stimulus Generation ===\n")

    available = get_available_speakers()
    print(f"Available speakers (local audio): {available}")

    df = pd.read_csv(RESULTS_CSV)
    print(f"Results CSV: {len(df)} rows")

    trials = select_trials(df, available)
    print(f"Selected {len(trials)} high-contrast trials\n")

    if not trials:
        print("ERROR: No eligible trials! Check that speaker audio exists.")
        return

    for i, t in enumerate(trials):
        print(f"  {i+1}. Speaker {t['speaker_id']}, target {t['target_id']}, contrast={t['contrast']:.3f}")

    print()
    generate_samples(trials)

    # Headphone check
    hc_path = OUTPUT_DIR / "headphone_check.wav"
    hc_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["sox", "-n", "-c", "2", str(hc_path), "synth", "1", "sine", "440", "remix", "1", "0"],
        check=True, capture_output=True,
    )
    print(f"\nHeadphone check generated: {hc_path}")
    print("\nDone! All audio in listening_study/audio/")


if __name__ == "__main__":
    main()
