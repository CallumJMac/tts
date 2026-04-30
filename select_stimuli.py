"""
Select stimuli for the webMUSHRA listening study and generate the final config.

Selects 12 trials (+ 1 practice) balanced across speakers, prioritising
high-contrast pairs where concat_audio and embed_avg differ most on
speaker_sim or UTMOS. This maximises the chance listeners perceive differences.

Usage:
    python select_stimuli.py --results ../results/phase4_40spk/results.csv --output-dir .

Requirements: pandas, numpy, sox (for anchor generation)
"""

import argparse
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# The 3 conditions to compare + hidden ref + anchor = 5 stimuli per trial
CONDITIONS = {
    "single_baseline": {"approach": "single_baseline", "strategy": "random", "n_refs": 1},
    "concat_longest_3": {"approach": "concat_audio", "strategy": "longest", "n_refs": 3},
    "embed_random_3": {"approach": "embed_avg", "strategy": "random", "n_refs": 3},
}

N_TRIALS = 12
N_PRACTICE = 1
MAX_PER_SPEAKER = 2
SEED = 42


def load_and_filter(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Use seed=42 runs only for reproducibility
    return df[df["seed"] == SEED].copy()


def find_eligible_groups(df: pd.DataFrame) -> list[dict]:
    """Find (speaker, target) groups that have all 3 conditions available."""
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
            # Compute contrast score
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
                "speaker_id": spk,
                "target_id": tgt,
                "contrast": sim_diff + utmos_diff,
                "sim_diff": sim_diff,
                "utmos_diff": utmos_diff,
                "conditions": conds,
            })
    return groups


def select_balanced(groups: list[dict], n: int, max_per_spk: int, seed: int) -> list[dict]:
    """Select n trials balanced across speakers, prioritising high contrast."""
    # Sort by contrast descending
    groups.sort(key=lambda x: x["contrast"], reverse=True)

    rng = np.random.default_rng(seed)
    selected = []
    spk_count = defaultdict(int)

    for g in groups:
        if spk_count[g["speaker_id"]] >= max_per_spk:
            continue
        selected.append(g)
        spk_count[g["speaker_id"]] += 1
        if len(selected) >= n:
            break

    # Shuffle trial order for the study
    rng.shuffle(selected)
    return selected


def create_anchor(src: Path, dst: Path) -> None:
    """Create degraded anchor: time-stretch to 0.7x speed using sox."""
    if shutil.which("sox"):
        try:
            subprocess.run(
                ["sox", str(src), str(dst), "tempo", "0.7"],
                check=True, capture_output=True
            )
            return
        except subprocess.CalledProcessError:
            pass
    # Fallback: just copy (placeholder)
    shutil.copy2(src, dst)


def generate_headphone_check(output_dir: Path) -> None:
    """Generate a simple left-channel-only tone for headphone verification."""
    if shutil.which("sox"):
        out = output_dir / "audio" / "headphone_check.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Generate 1s 440Hz tone, left channel only
            subprocess.run(
                ["sox", "-n", "-c", "2", str(out),
                 "synth", "1", "sine", "440",
                 "remix", "1", "0"],
                check=True, capture_output=True
            )
            return
        except subprocess.CalledProcessError:
            pass
    print("WARNING: sox not found. Please manually create audio/headphone_check.wav (left-channel tone)")


def copy_stimuli(selected: list[dict], output_dir: Path, results_base: Path) -> list[dict]:
    """Copy WAV files into serving directory, return trial configs."""
    audio_dir = output_dir / "audio"
    trials = []

    for i, trial in enumerate(selected):
        trial_dir = audio_dir / f"trial_{i:02d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        trial_config = {
            "type": "mushra",
            "id": f"trial_{i:02d}",
            "trialNumber": i + 1,
            "totalTrials": len(selected),
            "content": "Rate each sample compared to the reference speaker.",
            "stimuli": {}
        }

        # Copy condition WAVs
        for cname, row in trial["conditions"].items():
            src = results_base / row["output_path"]
            dst = trial_dir / f"{cname}.wav"
            if src.exists():
                shutil.copy2(src, dst)
            else:
                print(f"  WARNING: {src} not found")
            trial_config["stimuli"][cname] = f"audio/trial_{i:02d}/{cname}.wav"

        # Reference: use the ground-truth target audio from LibriTTS-R
        # For now, use single_baseline as reference proxy (closest to GT)
        ref_src = results_base / trial["conditions"]["single_baseline"]["output_path"]
        ref_dst = trial_dir / "reference.wav"
        if ref_src.exists():
            shutil.copy2(ref_src, ref_dst)
        trial_config["reference"] = f"audio/trial_{i:02d}/reference.wav"

        # Anchor: time-stretched
        anchor_dst = trial_dir / "anchor.wav"
        if ref_src.exists():
            create_anchor(ref_src, anchor_dst)
        trial_config["stimuli"]["anchor"] = f"audio/trial_{i:02d}/anchor.wav"

        trials.append(trial_config)

    return trials


def build_config(trials: list[dict], output_dir: Path) -> None:
    """Build the final study.json config with all pages."""
    # Load template
    config_path = output_dir / "configs" / "study.json"
    with open(config_path) as f:
        config = json.load(f)

    # Append trial pages
    config["pages"].extend(trials)

    # Add completion page
    config["pages"].append({
        "type": "completion",
        "completionCode": config.get("completionCode", "VCMUSHRA2026")
    })

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Wrote config with {len(config['pages'])} pages to {config_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="../results/phase4_40spk/results.csv",
                        help="Path to results CSV")
    parser.add_argument("--results-base", default="../results/phase4_40spk",
                        help="Base path for resolving output_path in CSV")
    parser.add_argument("--output-dir", default=".",
                        help="Listening study root directory")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results_base = Path(args.results_base)

    print("Loading results...")
    df = load_and_filter(args.results)
    print(f"  {len(df)} rows (seed=42)")

    print("Finding eligible trial groups...")
    groups = find_eligible_groups(df)
    print(f"  {len(groups)} eligible (speaker, target) pairs")

    print(f"Selecting {args.n_trials + N_PRACTICE} trials (balanced, high-contrast)...")
    selected = select_balanced(groups, args.n_trials + N_PRACTICE, MAX_PER_SPEAKER, args.seed)
    print(f"  Selected {len(selected)} trials across {len(set(s['speaker_id'] for s in selected))} speakers")
    print(f"  Contrast range: {selected[-1]['contrast']:.4f} - {selected[0]['contrast']:.4f}")

    # First trial becomes practice
    practice = selected[0]
    rated_trials = selected[1:]

    print("Copying stimuli...")
    # Set up practice trial audio
    practice_dir = output_dir / "audio" / "practice"
    practice_dir.mkdir(parents=True, exist_ok=True)
    for cname, row in practice["conditions"].items():
        src = results_base / row["output_path"]
        dst = practice_dir / f"{cname}.wav"
        if src.exists():
            shutil.copy2(src, dst)

    ref_src = results_base / practice["conditions"]["single_baseline"]["output_path"]
    if ref_src.exists():
        shutil.copy2(ref_src, practice_dir / "reference.wav")
        create_anchor(ref_src, practice_dir / "anchor.wav")

    # Copy rated trial stimuli
    trials = copy_stimuli(rated_trials, output_dir, results_base)

    print("Generating headphone check...")
    generate_headphone_check(output_dir)

    print("Building final config...")
    build_config(trials, output_dir)

    # Write selection manifest
    manifest = []
    for s in selected:
        manifest.append({
            "speaker_id": str(s["speaker_id"]),
            "target_id": s["target_id"],
            "contrast": round(s["contrast"], 4),
            "sim_diff": round(s["sim_diff"], 4),
            "utmos_diff": round(s["utmos_diff"], 4),
        })
    manifest_path = output_dir / "configs" / "selected_stimuli.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote selection manifest to {manifest_path}")
    print("Done. Run a local server to test: python -m http.server 8000")


if __name__ == "__main__":
    main()
