<div align="center">

# 🎙️ Multi-Reference Voice Cloning for Qwen3-TTS

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**[Paper](#citation) · [Results](results/phase3/results.csv) · [🎧 Audio Demos](https://callumjmac.github.io/tts/)**

</div>

---

When multiple reference recordings of a speaker are available, how should they be combined for voice cloning? We run 1,275 controlled synthesis runs on Qwen3-TTS and find a clear Pareto trade-off:

| Goal | Strategy | Gain over baseline |
|------|----------|--------------------|
| Speaker identity | `ConcatAudioCombiner`, longest 3 refs | SIM **+0.02**, *d* = 0.62 |
| Natural speech | `EmbedAvgCombiner`, any 3 refs | UTMOS **+0.06**, *d* = 0.61 |
| Stability | `EmbedAvgCombiner` | 0% failure rate |

Results validated on Qwen3-TTS-0.6B and 1.7B. Full results: [`results/phase3/results.csv`](results/phase3/results.csv)

---

## 🎧 Audio Demos

**[→ Open interactive demo with audio players](https://callumjmac.github.io/tts/)**

Two demos from the experiment — selected for maximum contrast on each metric:

**Demo A — Speaker identity** (speaker 1995, largest SIM gain in dataset)

| Strategy | SIM | UTMOS |
|----------|-----|-------|
| Single baseline | 0.870 | 4.481 |
| ConcatAudio · 3 refs (longest) | **0.987** (+0.117) | 4.457 |
| EmbedAvg · 3 refs (random) | 0.949 | 4.472 |

**Demo B — Naturalness** (speaker 1188, largest UTMOS gain in dataset)

| Strategy | UTMOS | SIM |
|----------|-------|-----|
| Single baseline | 4.100 | 0.973 |
| ConcatAudio · 3 refs (longest) | 4.414 | **0.981** |
| EmbedAvg · 3 refs (random) | **4.477** (+0.377) | 0.977 |

---

## Installation

```bash
git clone https://github.com/CallumJMac/tts.git && cd tts
pip install -r requirements.txt
```

Requires Python 3.10+, NVIDIA GPU ≥ 24 GB VRAM. Install PyTorch separately from [pytorch.org](https://pytorch.org).

---

## Quick Start

```python
from src.experiment.combiners import ConcatAudioCombiner, EmbedAvgCombiner
from src.experiment.ref_pool import RefItem
import soundfile as sf

def make_ref(path, text):
    info = sf.info(path)
    return RefItem(id=path, speaker_id="spk", text=text,
                   path=path, sample_rate=info.samplerate, duration=info.duration)

refs = [make_ref("ref1.wav", "text one"), make_ref("ref2.wav", "text two"),
        make_ref("ref3.wav", "text three")]

# Maximise speaker identity (ICL)
combined = ConcatAudioCombiner().combine(refs)

# Maximise naturalness (x-vector)
combined = EmbedAvgCombiner().combine(refs, model=tts_model)
```

---

## Reproducing the Experiment

```bash
# All 1,275 runs
python src/experiment/run_fewshot.py --output-dir outputs/

# Single strategy
python src/experiment/run_fewshot.py --combiner concat_audio --n-refs 3 --strategy longest

# Evaluate
python src/evaluation/evaluate.py generated.wav --ref-audio ref.wav --target-text-file target.txt

# Reproduce figures
python src/experiment/analyze.py --results results/phase3/results.csv
```

---

## Structure

```
src/experiment/     run_fewshot.py · combiners.py · ref_pool.py · analyze.py
src/evaluation/     evaluate.py (UTMOS · WavLM SIM · WER · SpeechBERTScore)
src/tts/            qwen_voice_clone.py
scripts/            CLI entry points + generate_samples.py
results/phase3/     results.csv · analysis/ (stats + plots)
paper/              main.tex · refs.bib · figures/
```

---

## Citation

```bibtex
@inproceedings{macpherson2026multiref,
  title   = {Multi-Reference Voice Cloning for {Q}wen3-{TTS}: A Trade-Off Between Speaker Fidelity and Naturalness},
  author  = {XXXX},
  year    = {2026},
  url     = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

MIT License · Built on [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS) · Evaluated on [LibriTTS-R](https://www.openslr.org/141/)
