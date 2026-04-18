<div align="center">
  <div>&nbsp;</div>
  <h1>🎙️ Multi-Reference Voice Cloning for Qwen3-TTS</h1>
  <p><em>A systematic study of the trade-off between speaker fidelity and naturalness</em></p>

  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"/></a>
  &nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"/></a>
  &nbsp;
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"/></a>
  &nbsp;
  <img src="https://img.shields.io/badge/GPU-24GB%20VRAM-green.svg" alt="GPU 24GB VRAM"/>

  <div>&nbsp;</div>

  <a href="#-what-this-repo-gives-you">What's inside</a> •
  <a href="#-key-results">Results</a> •
  <a href="#-audio-demos">Audio demos</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick start</a> •
  <a href="#-reproducing-the-experiment">Reproduce</a> •
  <a href="#citation">Citation</a>

  <div>&nbsp;</div>
</div>

---

## 🎁 What this repo gives you

When you have multiple reference recordings of a speaker, how should you combine them for voice cloning? This repo answers that question with a large-scale controlled experiment and provides **production-ready code** you can use today.

**1. A definitive answer on multi-reference strategy.**
1,275 controlled synthesis runs across 5 speakers, 2 model sizes, and 17 configurations. Not a demo — a rigorous study with Bonferroni-corrected statistics and effect sizes.

**2. Plug-and-play combiners for Qwen3-TTS.**
Three drop-in strategies (`ConcatAudioCombiner`, `EmbedAvgCombiner`, `ConcatCodeCombiner`) that slot into any Qwen3-TTS pipeline. The code tells you which to use and when.

**3. A complete TTS evaluation pipeline.**
One script that computes UTMOS (naturalness), WavLM speaker similarity, WER, and SpeechBERTScore — bundled, calibrated, and ready to run on your own outputs.

**4. The raw data.**
All 1,275 result rows, significance tables, and analysis plots included — reproduce every figure in the paper without re-running synthesis.

---

## 🔑 Key Finding

> **There is no single best strategy.** Audio concatenation maximises speaker identity; embedding averaging maximises naturalness. Neither dominates both axes.

The choice depends entirely on your application:

| I need… | Use | Why |
|---|---|---|
| The voice to sound *like the person* | `ConcatAudioCombiner`, longest 3 refs | SIM **+0.02** over baseline (*d* = 0.62, *p* < 0.001) |
| The speech to sound *natural and fluent* | `EmbedAvgCombiner`, any 3 refs | UTMOS **+0.06** over baseline (*d* = 0.61, *p* < 10⁻⁷) |
| Stability above all else | `EmbedAvgCombiner` | 0% catastrophic failure rate across all configs |

Results validated on both **Qwen3-TTS-0.6B** and **Qwen3-TTS-1.7B**.

---

## 📊 Key Results

<div align="center">

| Strategy | Conditioning | UTMOS ↑ | Speaker SIM ↑ | WER ↓ | Fail rate |
|---|---|:---:|:---:|:---:|:---:|
| Single baseline | ICL prompt | 4.425 | 0.928 | 15.3% | 0% |
| **Concat + longest** (*n*=3) | ICL prompt | 4.416 | **0.949** † | 21.3% | 2% |
| **Embed avg + random** (*n*=3) | x-vector | **4.483** † | 0.930 | 14.5% | 0% |

</div>

† *p* < 0.001, Bonferroni-corrected Wilcoxon signed-rank vs baseline. Full results: [`results/phase3/results.csv`](results/phase3/results.csv)

<details>
<summary>📋 View all 17 configurations</summary>
<br>

See [`results/phase3/analysis/summary_stats.csv`](results/phase3/analysis/summary_stats.csv) for aggregated statistics across all configurations, or [`results/phase3/analysis/significance_tests.csv`](results/phase3/analysis/significance_tests.csv) for the full significance table.

</details>

---

## 🎧 Audio Demos

Three clips — same speaker (Boris Johnson), same target text, different strategies. Listen for the fidelity vs. naturalness trade-off:

| # | File | Strategy | What to listen for |
|---|------|----------|-------------------|
| 1 | [`01_baseline_single.wav`](samples/demo/01_baseline_single.wav) | Single ref, ICL | Baseline |
| 2 | [`02_concat_longest_3.wav`](samples/demo/02_concat_longest_3.wav) | Concat 3 refs, ICL | Stronger speaker identity |
| 3 | [`03_embed_avg_3.wav`](samples/demo/03_embed_avg_3.wav) | Embed avg 3 refs, x-vector | Smoother, more natural |

> **Target text:** *"When multiple reference utterances are available, how should they be combined? We find a clear trade-off: concatenating references maximises speaker identity, while averaging embeddings maximises naturalness."*

Regenerate with [`scripts/generate_samples.py`](scripts/generate_samples.py) — see [`samples/demo/README.md`](samples/demo/README.md).

---

## ⚙️ How it works

Three conditioning strategies are compared:

<div align="center">

```
(a) Single baseline    [Ref audio + text] ──────────────────► TTS ► Output

(b) Audio concat       [Refs 1–N audio+text] ── Concat ──────► TTS ► Output

(c) Embedding avg      [Refs 1–N audio] ── Encode ── Avg ──► TTS (x-vec) ► Output
```

</div>

**Why they behave differently:** Concat and baseline use Qwen3-TTS's in-context learning (ICL) pathway — the model sees raw audio tokens and infers the speaker. Embedding averaging uses the `x_vector_only` conditioning pathway — a fundamentally different inference route. The naturalness gain in embedding averaging is primarily attributable to the conditioning pathway, not the averaging itself (the gain is already present at *n*=1, before any averaging occurs).

---

## 🚀 Installation

Requires Python 3.10+ and an NVIDIA GPU with ≥ 24 GB VRAM.

```bash
git clone https://github.com/CallumJMac/tts.git
cd tts
pip install -r requirements.txt
```

> **PyTorch:** Install separately per your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## ⚡ Quick Start

```python
from src.experiment.combiners import ConcatAudioCombiner, EmbedAvgCombiner
from src.experiment.ref_pool import RefItem
import soundfile as sf

# Build reference items
def make_ref(path, text):
    info = sf.info(path)
    return RefItem(id=path, speaker_id="spk", text=text,
                   path=path, sample_rate=info.samplerate, duration=info.duration)

refs = [make_ref("ref1.wav", "text one"), make_ref("ref2.wav", "text two"),
        make_ref("ref3.wav", "text three")]

# Strategy 1: maximise speaker identity
concat = ConcatAudioCombiner().combine(refs)
# → pass concat.audio_array + concat.text to generate_voice_clone()

# Strategy 2: maximise naturalness
embed = EmbedAvgCombiner().combine(refs, model=tts_model)
# → pass embed.voice_clone_prompt to generate_voice_clone()
```

---

## 🔬 Reproducing the Experiment

**Run all 1,275 configurations:**
```bash
python src/experiment/run_fewshot.py --output-dir outputs/phase3
```

**Run a specific strategy:**
```bash
# Concat, longest 3 refs (best for speaker fidelity)
python src/experiment/run_fewshot.py \
    --combiner concat_audio --n-refs 3 --strategy longest \
    --output-dir outputs/concat_longest_3

# Embed avg, random 3 refs (best for naturalness)
python src/experiment/run_fewshot.py \
    --combiner embed_avg --n-refs 3 --strategy random \
    --output-dir outputs/embed_random_3
```

**Evaluate generated audio:**
```bash
python src/evaluation/evaluate.py generated.wav \
    --ref-audio ref.wav --target-text-file target.txt
```

**Reproduce paper figures:**
```bash
python src/experiment/analyze.py \
    --results results/phase3/results.csv \
    --output-dir results/my_analysis
```

---

## 🗂️ Code Structure

```
src/
├── experiment/
│   ├── run_fewshot.py      # Main runner — 1,275 experiment configurations
│   ├── combiners.py        # ConcatAudioCombiner · EmbedAvgCombiner · ConcatCodeCombiner
│   ├── ref_pool.py         # Reference pool + longest/random selection strategies
│   └── analyze.py          # Statistical analysis, significance tests, figure generation
├── evaluation/
│   └── evaluate.py         # UTMOS · WavLM SIM · WER · SpeechBERTScore
└── tts/
    └── qwen_voice_clone.py # Qwen3-TTS wrapper

scripts/
├── generate_samples.py     # Generate strategy comparison demo audio
├── run_fewshot.py          # CLI entry point
├── evaluate.py             # CLI evaluation tool
└── analyze_fewshot.py      # CLI analysis tool

results/phase3/
├── results.csv             # Raw results — all 1,275 runs
└── analysis/               # Aggregated stats, significance tables, plots

paper/
├── main.tex                # Paper LaTeX source
├── refs.bib                # Bibliography
└── figures/                # All paper figures
```

---

## ⚠️ Limitations

- Evaluated on one model family (Qwen3-TTS 0.6B and 1.7B) and one dataset (LibriTTS-R, English only)
- The UTMOS naturalness gain in embedding averaging reflects the x-vector conditioning pathway, not multi-reference averaging per se
- No formal listening test — automated metrics only (UTMOS, WavLM SIM). Human perceptual validation is future work

---

## Citation

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

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
  <sub>Built on <a href="https://huggingface.co/Qwen/Qwen3-TTS">Qwen3-TTS</a> · Evaluated on <a href="https://www.openslr.org/141/">LibriTTS-R</a></sub>
</div>
