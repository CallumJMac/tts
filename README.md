<div align="center">

# 🎙️ Multi-Reference Voice Cloning for Qwen3-TTS

### A Trade-Off Between Speaker Fidelity and Naturalness

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GPU: 24GB VRAM](https://img.shields.io/badge/GPU-24GB%20VRAM-green.svg)](#installation)

**[📄 Paper](#citation) · [📊 Results](results/phase3/results.csv) · [🔬 Analysis](results/phase3/analysis/)**

</div>

---

## Overview

When multiple reference utterances are available for voice cloning, how should they be combined? We systematically compare two multi-reference strategies for **Qwen3-TTS** across **1,275 controlled runs** on LibriTTS-R and find a clear Pareto trade-off:

- 🎯 **Audio concatenation (ICL)** → maximises speaker identity (SIM **+0.02**, *d* = 0.62)
- 🌊 **Embedding averaging (x-vector)** → maximises naturalness (UTMOS **+0.06**, *d* = 0.61)

Neither strategy dominates both axes. The right choice depends on whether your application prioritises *sounding like the speaker* or *sounding natural*.

<div align="center">

![Pareto Trade-off](paper/figures/pareto_tradeoff.pdf)

*Pareto frontier in SIM–UTMOS space. Each point is a configuration mean ± 1 SD.*

</div>

---

## 🎧 Audio Demos

Three samples on the same speaker and target text — listen to hear the trade-off:

| Sample | Strategy | What to listen for |
|--------|----------|-------------------|
| [01_baseline_single.wav](samples/demo/01_baseline_single.wav) | Single reference (ICL) | Baseline |
| [02_concat_longest_3.wav](samples/demo/02_concat_longest_3.wav) | Concat 3 refs (ICL) | Stronger speaker identity |
| [03_embed_avg_3.wav](samples/demo/03_embed_avg_3.wav) | Embed avg 3 refs (x-vector) | Smoother, more natural |

Regenerate with `scripts/generate_samples.py` — see [`samples/demo/README.md`](samples/demo/README.md).

---

## 📈 Key Results

| Strategy | Conditioning | UTMOS ↑ | Speaker SIM ↑ | Failure Rate |
|---|---|---|---|---|
| Single baseline | ICL prompt | 4.425 | 0.928 | 0% |
| **Concat + longest** (*n* = 3) | ICL prompt | 4.416 | **0.949** † | 2% |
| **Embed avg + random** (*n* = 3) | x-vector only | **4.483** † | 0.930 | 0% |

† *p* < 0.001, Bonferroni-corrected Wilcoxon signed-rank test vs. baseline.

Results validated across two model sizes: **Qwen3-TTS-0.6B** and **Qwen3-TTS-1.7B**.

<details>
<summary>📋 Full results table (all 17 configurations)</summary>

See [`results/phase3/results.csv`](results/phase3/results.csv) for all 1,275 individual synthesis runs, or [`results/phase3/analysis/summary_stats.csv`](results/phase3/analysis/summary_stats.csv) for aggregated statistics.

</details>

---

## 🔑 Key Findings

1. **Pareto trade-off is robust** — consistent across 5 speakers and two model scales (0.6B, 1.7B)
2. **Naturalness gain appears at *n* = 1** — the UTMOS improvement in embedding averaging is attributable to the x-vector conditioning *pathway*, not multi-reference averaging per se
3. **Reference selection matters more than count** — longest utterances outperform random selection; gains plateau beyond *n* = 3
4. **Concat instability risk** — concat + random at *n* = 3 can produce catastrophic WER failures (max 297.8s synthesis time); embed is stable across all configurations

---

## 🚀 Installation

Requires Python 3.10+ and an NVIDIA GPU with ≥ 24 GB VRAM (tested on A10G).

```bash
git clone https://github.com/CallumJMac/tts.git
cd tts
pip install -r requirements.txt
```

> **Note:** PyTorch and torchaudio are not in `requirements.txt` — install them separately per your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## ⚡ Quick Start

```python
from src.experiment.combiners import ConcatAudioCombiner, EmbedAvgCombiner
from src.experiment.ref_pool import RefPool

# Load reference utterances
pool = RefPool("data/libritts_r_aligned", speaker_id="1188")
refs = pool.select(n=3, strategy="longest")

# Strategy 1: Audio concatenation (maximises speaker similarity)
combiner = ConcatAudioCombiner()
combined = combiner.combine(refs)

# Strategy 2: Embedding averaging (maximises naturalness)
combiner = EmbedAvgCombiner()
combined = combiner.combine(refs, model=tts_model)
```

---

## 🔬 Reproducing the Experiment

**Run all 1,275 configurations:**

```bash
python src/experiment/run_fewshot.py \
    --model qwen3-tts-0.6b \
    --output-dir outputs/phase3
```

**Run a specific combiner:**

```bash
# Audio concatenation, longest 3 references
python src/experiment/run_fewshot.py \
    --combiner concat_audio \
    --n-refs 3 \
    --strategy longest \
    --output-dir outputs/concat_longest_3

# Embedding averaging, random 3 references
python src/experiment/run_fewshot.py \
    --combiner embed_avg \
    --n-refs 3 \
    --strategy random \
    --output-dir outputs/embed_random_3
```

**Evaluate generated audio:**

```bash
python src/evaluation/evaluate.py \
    --generated-dir outputs/phase3 \
    --ref-audio-dir data/libritts_r_aligned \
    --output results/my_run.csv
```

**Analyse and reproduce paper figures:**

```bash
python src/experiment/analyze.py \
    --results results/my_run.csv \
    --output-dir results/my_analysis
```

---

## 🏗️ Code Structure

```
src/
├── experiment/
│   ├── run_fewshot.py      # Main experiment runner (1,275 runs)
│   ├── combiners.py        # ConcatAudioCombiner · EmbedAvgCombiner · ConcatCodeCombiner
│   ├── ref_pool.py         # Reference utterance pool + selection strategies
│   └── analyze.py          # Statistical analysis + figure generation
├── evaluation/
│   └── evaluate.py         # UTMOS · WavLM SIM · WER · SpeechBERTScore
├── tts/
│   └── qwen_voice_clone.py # Qwen3-TTS wrapper
└── preprocess/
    └── diarize.py          # Speaker diarisation utilities

scripts/
├── run_fewshot.py          # CLI entry point
├── evaluate.py             # CLI evaluation tool
├── analyze_fewshot.py      # CLI analysis tool
├── wav_mov.py              # Video → WAV conversion
└── stitch_good_wavs.py     # WAV stitching

paper/
├── main.tex                # Paper source
├── refs.bib                # Bibliography
└── figures/                # All paper figures (PDF + PNG)

results/
└── phase3/
    ├── results.csv         # Raw results (1,275 runs)
    └── analysis/           # Aggregated stats + plots
```

---

## 📐 Method

Three conditioning strategies are compared:

| Strategy | Mode | How references are combined |
|---|---|---|
| **Single baseline** | ICL prompt | One (audio, text) pair as speaker prompt |
| **Concat** | ICL prompt | *N* audio files concatenated; texts joined |
| **Embed avg** | x-vector only | Speaker embeddings extracted and averaged |

> **Conditioning note:** Concat and baseline use Qwen3-TTS's prompt-based ICL pathway; embed avg uses `x_vector_only` mode. The UTMOS advantage of embed avg is likely attributable to the conditioning pathway rather than averaging alone — see §4 of the paper.

---

## 📚 Citation

If you use this code or findings, please cite:

```bibtex
@inproceedings{macpherson2026multiref,
  title     = {Multi-Reference Voice Cloning for {Q}wen3-{TTS}:
               A Trade-Off Between Speaker Fidelity and Naturalness},
  author    = {XXXX},
  booktitle = {XXXX},
  year      = {2026},
  url       = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
<sub>Built on <a href="https://huggingface.co/Qwen/Qwen3-TTS">Qwen3-TTS</a> · Evaluated on <a href="https://www.openslr.org/141/">LibriTTS-R</a></sub>
</div>
