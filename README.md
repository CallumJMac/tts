# Multi-Reference Voice Cloning for Qwen3-TTS

Code for the paper: *Multi-Reference Voice Cloning for Qwen3-TTS: A Trade-Off Between Speaker Fidelity and Naturalness*

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

## Abstract

We systematically compare two multi-reference conditioning strategies for Qwen3-TTS — audio concatenation (concat) and embedding averaging (embed/x-vector) — across 1,275 controlled runs on LibriTTS-R. We find a Pareto trade-off: concat maximises speaker similarity (SIM +0.02, Cohen's d=0.62), while embedding averaging maximises naturalness (UTMOS +0.06, d=0.61). Neither strategy dominates on all metrics, and the optimal choice depends on whether downstream use prioritises voice identity or perceived quality.

## Key Results

| Strategy | UTMOS (↑) | Speaker SIM (↑) | WER (↓) | SpeechBERTScore (↑) |
|---|---|---|---|---|
| Baseline (single ref) | — | — | — | — |
| **Concat (audio)** | — | **+0.02** | — | — |
| **Embed avg (x-vector)** | **+0.06** | — | — | — |

*Full result tables are in the paper. Cohen's d reported for each primary metric.*

Models evaluated: Qwen3-TTS-0.6B and Qwen3-TTS-1.7B. Dataset: LibriTTS-R, 5 speakers, 35 reference utterances + 5 targets each.

## Installation

Requires Python 3.10+ and an NVIDIA GPU with 24 GB VRAM.

```bash
pip install -r requirements.txt
```

## Reproducing the Experiment

Run all 1,275 experiment configurations:

```bash
python src/experiment/run_fewshot.py
```

Key arguments (see `--help` for full options):

```bash
python src/experiment/run_fewshot.py \
    --model qwen3-tts-1.7b \
    --combiner concat_audio \   # or embed_avg, concat_code
    --n-refs 5 \
    --output-dir outputs/
```

Combiners available (`src/experiment/combiners.py`):
- `concat_audio` — ConcatAudioCombiner
- `embed_avg` — EmbedAvgCombiner
- `concat_code` — ConcatCodeCombiner

Reference pool management is handled by `src/experiment/ref_pool.py`.

## Evaluating Results

Run the evaluation suite over generated audio:

```bash
python src/evaluation/evaluate.py \
    --generated-dir outputs/ \
    --ref-dir data/libritts_r/refs/ \
    --target-text-dir data/libritts_r/targets/
```

Analyse and aggregate results:

```bash
python src/experiment/analyze.py --results-dir outputs/
```

Metrics computed: UTMOS (naturalness), WavLM Speaker Similarity, WER, SpeechBERTScore.

## Code Structure

```
src/
  experiment/
    run_fewshot.py       # Main runner (1,275 runs)
    combiners.py         # ConcatAudioCombiner, EmbedAvgCombiner, ConcatCodeCombiner
    ref_pool.py          # Reference utterance management
    analyze.py           # Results aggregation and statistics
  evaluation/
    evaluate.py          # UTMOS, SIM, WER, SpeechBERTScore
  tts/
    qwen_voice_clone.py  # Qwen3-TTS voice cloning wrapper
scripts/
  wav_mov.py             # Video-to-WAV conversion
  stitch_good_wavs.py    # WAV stitching
  evaluate.py            # CLI evaluation tool
requirements.txt
```

## Citation

```bibtex
@inproceedings{XXXX,
  title     = {Multi-Reference Voice Cloning for Qwen3-TTS: A Trade-Off Between Speaker Fidelity and Naturalness},
  author    = {XXXX},
  booktitle = {XXXX},
  year      = {XXXX},
  url       = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

## License

MIT
