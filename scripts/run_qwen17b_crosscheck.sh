#!/usr/bin/env bash
set -euo pipefail

# Reduced cross-model replication on Qwen3-TTS 1.7B.
# Runs only the key operating points:
#   1) single_baseline (n=1, random)
#   2) concat_audio (n=3, longest)
#   3) embed_avg (n=3, random)
#
# Total: 5 speakers x 5 targets x 3 seeds x 3 configs = 225 runs.

OUT_DIR="outputs/fewshot/qwen17b_crosscheck"
RESULTS_CSV="${OUT_DIR}/results.csv"

mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  --manifest data/libritts_r_aligned/manifest.json
  --output-dir "${OUT_DIR}"
  --results-csv "${RESULTS_CSV}"
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base
  --seeds 42 123 456
  --held-out-per-speaker 5
  --held-out-targets 5
  --held-out-seed 0
  --device cuda:0
  --dtype bfloat16
  --skip-speechbertscore
)

# 1) Baseline
python scripts/run_fewshot.py \
  "${COMMON_ARGS[@]}" \
  --approaches single_baseline \
  --num-refs 1 \
  --strategies random

# 2) Concat winner
python scripts/run_fewshot.py \
  "${COMMON_ARGS[@]}" \
  --approaches concat_audio \
  --num-refs 3 \
  --strategies longest

# 3) Embed winner
python scripts/run_fewshot.py \
  "${COMMON_ARGS[@]}" \
  --approaches embed_avg \
  --num-refs 3 \
  --strategies random
