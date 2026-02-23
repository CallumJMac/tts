#!/usr/bin/env bash
set -euo pipefail

# Reduced cross-model replication on Qwen3-TTS 1.7B.
# Keeps the same 5 speakers/targets/seeds and tests only key configs.

python scripts/run_fewshot.py \
  --manifest data/libritts_r_aligned/manifest.json \
  --output-dir outputs/fewshot/qwen17b_crosscheck \
  --results-csv outputs/fewshot/qwen17b_crosscheck/results.csv \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --approaches single_baseline concat_audio embed_avg \
  --num-refs 1 3 \
  --strategies random longest \
  --seeds 42 123 456 \
  --held-out-per-speaker 5 \
  --held-out-targets 5 \
  --held-out-seed 0 \
  --device cuda:0 \
  --dtype float16 \
  --skip-speechbertscore
