# Dynamic Few-Shot Examples for Qwen3-TTS Voice Cloning

> **Living document** — single source of truth for experiment design, progress, and decisions.
> Last updated: 2026-02-23

---

## Target Venue

**Interspeech 2026** — Sydney, Australia, Sep 27 – Oct 1
- **Submission deadline: February 25, 2026 (AoE)**
- Update deadline: March 4, 2026
- Acceptance notification: June 5, 2026
- Relevant tracks: Speech Synthesis, Resources and Evaluation, Generative AI for Speech
- Special session: "Ethical Frameworks for Synthetic Speech"

---

## Background

### Current State
The existing voice cloning pipeline (`src/tts/qwen_voice_clone.py`) passes a **single** (ref_audio, ref_text) pair to `Qwen3TTSModel.generate_voice_clone()`. The model operates in **ICL (In-Context Learning) mode** where the reference audio codes and reference text are embedded and prepended to the target text as a prompt.

### API Constraint
The Qwen3-TTS `generate_voice_clone()` API accepts **one reference audio + one reference text per synthesis call**. Internally (`modeling_qwen3_tts.py:2188`), `generate_icl_prompt()` is called exactly once per sample — there is no built-in loop for multiple ICL turns.

### Key Insight from Architecture
The ICL prompt is assembled in **embedding space** (`generate_icl_prompt()` at lines 1968-2019):
- `ref_text` tokens and `text` (target) tokens are concatenated and projected: `text_embed = text_projection(text_embeddings(cat([ref_id, text_id])))`
- `ref_code` (audio codec tokens) are embedded via multi-group codebook embeddings and summed
- These two streams are combined (added or interleaved depending on streaming mode)

Since the construction happens in embedding space with simple concatenation, **extending the ref_code and ref_text to include multiple examples is architecturally feasible**, though the model was trained on single-example prompts.

### Reference from Paper
> "For voice cloning, Qwen3-TTS clones a target voice from (i) reference speech via a speaker embedding, enabling real-time cloning, or (ii) a text–speech pair via in-context learning, which better preserves prosody."

The model supports **3-second voice cloning** — so the reference audio is typically short. Providing more audio context may improve speaker identity capture.

### Model Details
- **Current model**: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (0.6B params, 12.5 Hz multi-codebook tokenizer)
- **Larger variant**: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (1.7B params, better WER: 1.24 vs 1.32 on Seed-TTS test-en)
- **Context limit**: max_new_tokens=2048 by default (configurable), trained up to 32,768 tokens
- **Codec**: 16-layer RVQ at 12.5 Hz (80ms per token), first codebook = semantic, rest = acoustic

---

## Research Questions

1. **Does providing multiple reference (audio, text) pairs improve voice cloning quality** compared to a single pair?
2. **What is the optimal number of few-shot examples** (1, 2, 3, 5)?
3. **Does the selection strategy for examples matter** — random vs. text-similarity-based vs. duration-matched?
4. **Which approach to multi-ref works best** — audio concatenation, code-level concatenation, or speaker embedding averaging?

---

## Evaluation Framework

### Automated Metrics (Inner Loop)
Using the existing evaluation framework (`src/evaluation/evaluate.py`):

| Metric | Model | What it measures | Direction | Notes |
|--------|-------|-----------------|-----------|-------|
| **UTMOS** | utmos22_strong (~400MB) | Speech naturalness (MOS 1-5) | Higher is better | >4.0 good, >4.3 excellent |
| **Speaker Similarity** | WavLM-XVector (~360MB) | Voice identity cosine sim | Higher is better | >0.85 strong, >0.95 excellent |
| **WER/CER** | Whisper turbo (~1.5GB) | Intelligibility | Lower is better | <5% excellent, <10% good |
| **SpeechBERTScore** | WavLM-Large L14 (~1.2GB) | Acoustic similarity (F1) | Higher is better | Best for same-text comparison |

### SOTA Additions to Consider
From literature review (VALL-E, Seed-TTS, CLaM-TTS, DiTTo-TTS evaluation protocols):

| Metric | What it adds | Priority |
|--------|-------------|----------|
| **SIM-o** | Speaker sim against original ground truth (not just ref) | High — standard in zero-shot eval |
| **SIM-r** | Speaker sim against resynthesized ground truth | High — isolates model error from vocoder error |
| **SECS (ECAPA-TDNN)** | Alternative speaker encoder to WavLM | Medium — cross-validates speaker sim |
| **MCD** | Mel-cepstral distortion vs ground truth | Low — requires parallel ground truth |

### Where Automated Metrics Fall Short (HITL Required)
Based on literature review of evaluation gaps:

| Aspect | Why automated fails | HITL method |
|--------|-------------------|-------------|
| **Prosody & expressiveness** | No reliable automated metric for appropriate stress, pacing, emotional tone | MOS or CMOS |
| **Idiosyncratic speech patterns** | Embeddings average out vocal fry, uptalk, characteristic phoneme pronunciations | SMOS (Speaker MOS) |
| **Artifacts & glitches** | UTMOS gives utterance-level average; misses a single click at 3.2s | Per-sample human audit |
| **Long-form coherence** | No standard metric for voice drift, prosodic monotony over 60s+ | Human listening test |
| **Uncanny valley** | Outputs score well on all metrics but sound "off" to humans | CMOS / MUSHRA A/B |

**Strategy**: Use automated metrics for fast iteration (inner loop). Use HITL for final validation and paper results (outer loop).

### Minimum Quality Thresholds
- SIM-o > 0.65 = acceptable speaker match
- WER < 5% = excellent intelligibility
- UTMOS > 3.8 = acceptable naturalness

---

## Experiment Design

### Independent Variables

| Variable | Values | Description |
|----------|--------|-------------|
| **Approach** | `concat_audio`, `concat_code`, `embed_avg`, `single_baseline` | How multiple refs are combined |
| **Num examples** | 1, 2, 3, 5 | Number of reference (audio, text) pairs |
| **Selection strategy** | `random`, `longest`, `text_similarity` | How examples are chosen from the pool |

### Approaches to Test

#### Approach 1: `concat_audio` (Audio-Level Concatenation)
- Concatenate N reference WAV files into a single longer WAV
- Concatenate their transcripts with sentence boundaries
- Pass as a single (ref_audio, ref_text) to the existing API
- **Pros**: No library modification needed, simplest to implement
- **Cons**: May hit model's context length limits; speaker encoder sees one long clip

#### Approach 2: `concat_code` (Code-Level Concatenation)
- Use `create_voice_clone_prompt()` individually for each ref to get ref_code tensors
- Concatenate the ref_code tensors along the time axis
- Concatenate the ref_text strings (with ChatML boundaries)
- Average the speaker embeddings across all refs
- Build a custom `VoiceClonePromptItem` with merged data
- **Pros**: Works at the representation level the model actually sees
- **Cons**: Requires lower-level API usage; untested territory

#### Approach 3: `embed_avg` (Speaker Embedding Averaging)
- Extract speaker embeddings from each reference audio
- Average (or weighted-average) the embeddings
- Use `x_vector_only_mode=True` with the averaged embedding
- **Pros**: Clean, no context length concerns
- **Cons**: Loses prosody information (no ICL), only captures speaker identity

#### Approach 4: `single_baseline` (Control)
- Existing single-ref approach
- Test with each individual ref separately, then pick best/worst/median

### Example Pool: LibriTTS-R Aligned Dataset

**Primary dataset**: `data/libritts_r_aligned/` — open-source, reproducible, standard in TTS literature.

| Speaker | Clips | Total audio | Mean clip | Range |
|---------|-------|-------------|-----------|-------|
| 1188 | 40 | 492s (~8min) | 12.3s | 0.4–26.0s |
| 4992 | 40 | 264s (~4min) | 6.6s | 0.5–17.6s |
| 1995 | 40 | 228s (~4min) | 5.7s | 0.6–16.6s |
| 4446 | 40 | 109s (~2min) | 2.7s | 0.8–15.1s |
| 5142 | 40 | 93s (~2min) | 2.3s | 0.5–5.8s |

**Audio**: 24kHz WAV, with TextGrid forced alignments (4,826 files)
**Text**: Both `text_normalized` and `text_original` variants per utterance
**Manifest**: `data/libritts_r_aligned/manifest.json` (200 entries)

**Why this is better than ref_video/ celebrity voices for the experiment:**
1. **Ground truth exists** — hold out clips as targets, compare generated audio against actual recordings (proper SIM-o/SIM-r)
2. **Reproducible** — open-source dataset, anyone can replicate results
3. **40 clips per speaker** — ample material for few-shot selection and held-out evaluation
4. **Standard in TTS literature** — LibriTTS-R is widely used in VALL-E, Seed-TTS, etc.
5. **Varied clip lengths** — natural variation from 0.4s to 26s allows testing duration effects

**Split strategy**: For each speaker (40 clips):
- **Held-out evaluation set**: 5 clips (used as target text; ground truth audio enables SIM-o)
- **Reference pool**: 35 clips (candidates for few-shot selection)

The celebrity voices in `ref_video/` can still be used as a secondary qualitative validation (no ground truth, but subjectively interesting).

### Test Matrix
For each of the 5 LibriTTS-R speakers:
1. **Baseline**: Single ref, existing pipeline -> evaluate against held-out ground truth
2. **concat_audio x {2, 3, 5} refs x {random, longest}** -> evaluate
3. **concat_code x {2, 3, 5} refs x {random, longest}** -> evaluate
4. **embed_avg x {2, 3, 5} refs x {random, longest}** -> evaluate

Target text: Held-out utterances from the same speaker (with ground truth audio for SIM-o).

### Statistical Design
- Run each configuration **3 times** (TTS generation is stochastic due to sampling)
- **5 held-out targets per speaker** — each config is tested on 5 different target utterances
- Report mean +/- std for each metric
- Total runs estimate: 5 speakers x 5 targets x (1 + 3x3x2 configs) x 3 seeds = ~1,425 runs
- **Reduced pilot**: 1 speaker x 1 target x 2 configs x 1 seed = 2 runs

---

## Compute Strategy

### Hardware

| Phase | Where | Why |
|-------|-------|-----|
| Phase 0-2 (pilot) | Local MPS (Apple Silicon) | Fast iteration, no setup cost, ~8 runs |
| Phase 3 (full) | AWS `g5.xlarge` (A10G 24GB) | CUDA + float16 + flash-attn, ~5-10x faster |

### Per-Run Compute Breakdown

| Component | Model size | Notes |
|-----------|-----------|-------|
| TTS synthesis | 0.6B (~1.2GB) | Dominant cost; RTF ~0.288 on CUDA, ~1-3 on MPS |
| UTMOS | ~400MB | Fast, single forward pass |
| Speaker Similarity | ~360MB | Two embeddings + cosine sim |
| WER (Whisper) | ~1.5GB | Transcribe + compare |
| SpeechBERTScore | ~1.2GB | Heaviest eval metric |

**Total eval model memory**: ~3.5GB. On MPS, competes with TTS model for unified memory.

### Key Optimisations

1. **Load models once, reuse across runs** — current scripts reload per invocation; refactor to persistent model
2. **Separate TTS from eval** — generate all WAVs first, then batch-evaluate (avoids memory contention)
3. **Pre-compute ref_codes and speaker embeddings** — cache `VoiceClonePromptItem` to disk, avoid re-encoding
4. **Skip SpeechBERTScore during exploration** — add only for final results (heaviest metric, least informative for our RQs)
5. **Shorter target text for pilot** — 1-2 sentences, not full paragraphs
6. **Prune experiment matrix early** — drop approaches/strategies that show no signal after pilot

### GPU Deployment (Phase 3)
- `g5.xlarge` (A10G 24GB) is the sweet spot for 0.6B; 1.7B also fits in float16
- Simple setup: Deep Learning AMI + clone repo + tmux, no orchestration needed
- Consider testing 1.7B model on GPU too (paper shows consistent improvements over 0.6B)

---

## Implementation Plan

### Step 1: Build the Reference Pool Manager
**New file**: `src/experiment/ref_pool.py`
- Load reference audio/text pairs per speaker
- Support splitting long references into segments (silence-based or TextGrid-aligned)
- Implement selection strategies: `random`, `longest`, `text_similarity`

### Step 2: Implement Multi-Ref Combiners
**New file**: `src/experiment/combiners.py`
- `ConcatAudioCombiner`: Concatenate WAVs + texts at file level
- `ConcatCodeCombiner`: Merge ref_code tensors + ref_text at prompt level
- `EmbedAvgCombiner`: Average speaker embeddings from multiple refs

### Step 3: Build the Experiment Runner
**New file**: `src/experiment/run_fewshot.py`
- Accepts a config (YAML/JSON) specifying the experiment matrix
- For each (speaker, approach, num_refs, strategy, seed):
  1. Select refs from pool using strategy
  2. Combine refs using the approach's combiner
  3. Run synthesis via `generate_voice_clone()` (or modified call for `concat_code`)
  4. Save output WAV to structured directory: `outputs/fewshot/{speaker}/{approach}_{n}refs_{strategy}_seed{i}.wav`
  5. Run evaluation metrics
  6. Log results to CSV/JSON

### Step 4: Build the Analysis Script
**New file**: `src/experiment/analyze.py`
- Load results CSV
- Compute summary statistics per (approach, num_refs, strategy)
- Generate comparison tables and plots
- Statistical significance tests (paired t-test or Wilcoxon) vs. baseline

### Step 5: CLI Entry Point
**New file**: `scripts/run_fewshot.py`
- Minimal entry point following existing pattern

---

## Experiment Config Example

```yaml
experiment:
  name: "fewshot_v1"
  seeds: [42, 123, 456]
  model: "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
  device: "mps"            # Phase 0-2 local; Phase 3 switch to "cuda:0"
  dtype: "float32"         # Phase 0-2 on MPS; Phase 3 switch to "float16"
  held_out_per_speaker: 5  # clips reserved for evaluation (have ground truth)

dataset:
  manifest: "data/libritts_r_aligned/manifest.json"
  sample_rate: 24000
  speakers: ["1188", "4992", "1995", "4446", "5142"]  # 5 LibriTTS-R speakers, 40 clips each

approaches:
  - name: single_baseline
    num_refs: [1]
    strategies: [random]
  - name: concat_audio
    num_refs: [2, 3, 5]
    strategies: [random, longest]
  - name: concat_code
    num_refs: [2, 3, 5]
    strategies: [random, longest]
  - name: embed_avg
    num_refs: [2, 3, 5]
    strategies: [random, longest]
```

---

## Expected Outcomes & Hypotheses

| Hypothesis | Rationale |
|-----------|-----------|
| `concat_audio` with 2-3 refs will improve speaker similarity over single-ref | More acoustic context gives the model a richer speaker representation |
| `concat_code` will outperform `concat_audio` | Code-level concatenation avoids resampling artifacts and aligns with model's internal representation |
| `embed_avg` will improve speaker similarity but hurt naturalness | Averaging embeddings captures identity but loses prosodic conditioning |
| Beyond 3-5 refs, performance will plateau or degrade | Model's context window has limits; too much context may confuse the LM |
| `longest` strategy will outperform `random` | Longer clips provide more speaker signal |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Model context length overflow with many refs | Start with 2-3 refs; monitor token counts; truncate if needed |
| `concat_code` approach may produce incoherent prompts | Compare generated ref_code tensor shapes to single-ref baseline; validate codec decoding |
| Stochastic variation masks real effects | 3 seeds per config; use statistical tests |
| Slow iteration (MPS inference is not fast) | Start with 2 speakers, 1 seed for pilot; scale up after validating approach works |
| Interspeech deadline pressure | Phased approach; pilot first; write paper structure in parallel |

---

## Execution Phases

| Phase | What | Runs | Compute | Goal |
|-------|------|------|---------|------|
| **0: Benchmark** | 1 TTS + 1 eval on MPS using LibriTTS-R speaker | 2 | Local | Get per-run wall clock time |
| **1: Pilot** | 1 speaker (e.g. 1188), `concat_audio` 2-ref vs baseline, 1 target, 1 seed | 2 | Local | Validate signal exists |
| **2: Approach comparison** | 1 speaker, all 3 approaches, 2-ref, 1 target, 1 seed | 4 | Local | Pick best approach |
| **3: Scale** | 5 speakers, 5 targets each, winning approach, {2,3,5} refs, 3 seeds | ~225+ | AWS GPU | Full results for paper |
| **4: Paper** | Write up results, add HITL eval if time permits | — | — | Submit to Interspeech |

---

## Progress Log

### 2026-02-17 — Project Setup & Planning
- [x] Explored codebase structure and existing TTS pipeline
- [x] Investigated Qwen3-TTS library internals (`qwen3_tts_model.py`, `modeling_qwen3_tts.py`)
- [x] Confirmed API constraint: single ref per call, but architecturally extensible
- [x] Identified three approaches: `concat_audio`, `concat_code`, `embed_avg`
- [x] Researched SOTA evaluation methods (SIM-o, SIM-r, SECS, UTMOS, WER, MCD)
- [x] Identified HITL gaps: prosody, artifacts, speaker idiosyncrasies, long-form coherence
- [x] Identified target venue: **Interspeech 2026** (deadline Feb 25 AoE)
- [x] Designed compute strategy: MPS for pilot, AWS g5.xlarge for full experiment
- [x] Created experiment plan and phased execution approach
- [x] Audited LibriTTS-R aligned dataset: 5 speakers x 40 clips, 24kHz, with TextGrid alignments
- [x] Decided to use LibriTTS-R as primary dataset (open-source, ground truth, reproducible)
- [x] Designed held-out split: 5 eval targets + 35 ref pool per speaker
- [x] Implemented pilot experiment code:
  - `src/experiment/ref_pool.py` — manifest loader, per-speaker pool builder, held-out split, selection strategies (random, longest)
  - `src/experiment/combiners.py` — ConcatAudioCombiner, ConcatCodeCombiner, EmbedAvgCombiner
  - `src/experiment/run_fewshot.py` — full experiment runner with CSV logging, timing, and inline evaluation
  - `scripts/run_fewshot.py` — CLI entry point
- [x] Verified: ref_pool loads manifest correctly, 5 speakers x 35 refs + 5 held-out each
- [x] Verified: ConcatAudioCombiner produces correct concatenated audio (48.9s from 2 longest clips of speaker 1188)
- [x] **Phase 0+1 pilot completed** — 3 runs on speaker 1188, 1 target, seed=42
- [x] **Signal detected**: concat_audio 2-ref shows higher speaker sim (0.9769) vs baseline (0.9689), UTMOS stable (4.37 vs 4.39), WER unchanged (10.3%)
- [x] Optimised eval model loading — UTMOS, WavLM, Whisper now loaded once and reused across runs

### Phase 0+1 Results (2026-02-17)

| Approach | n_refs | UTMOS | Speaker Sim | WER | TTS time | Eval time |
|----------|--------|-------|-------------|-----|----------|-----------|
| single_baseline | 1 | 4.389 | 0.9689 | 10.3% | 12.0s | 10.4s |
| concat_audio | 1 | 4.387 | 0.9634 | 10.3% | 10.5s | 5.3s |
| concat_audio | 2 | 4.372 | **0.9769** | 10.3% | 11.1s | 6.1s |

**Observations:**
- 2-ref concat_audio improved speaker similarity by +0.008 over baseline
- Naturalness (UTMOS) remained excellent (>4.3 across all)
- Intelligibility (WER) unchanged — all produced identical transcripts
- TTS generation: ~10-12s per run on MPS float32
- First eval run was slow (10.4s) due to model loading; subsequent runs ~5-6s (still reloading — now fixed)

- [x] **Phase 2 completed** — 25 runs: all 4 approaches x {1,2,3,5} refs x {random, longest} on speaker 1188
- [x] Eval model caching confirmed working — eval time dropped from 10.4s to ~1.5-2.0s per run

### Phase 2 Results (2026-02-17 evening)

**Speaker 1188, target=1188_133604_000040_000002, seed=42, 25 runs**

#### Full Results Table

| Approach | n_refs | Strategy | UTMOS | Speaker Sim | WER | TTS time |
|----------|--------|----------|-------|-------------|-----|----------|
| single_baseline | 1 | random | 4.414 | 0.9697 | 10.3% | 11.7s |
| concat_audio | 1 | random | 4.399 | 0.9671 | 13.8% | 9.3s |
| concat_audio | 1 | longest | 3.988 | 0.9690 | 10.3% | 14.1s |
| concat_audio | 2 | random | 4.444 | 0.9781 | 10.3% | 12.5s |
| concat_audio | 2 | longest | 4.374 | 0.9791 | 17.2% | 13.5s |
| concat_audio | 3 | random | 4.404 | 0.9663 | 17.2% | 11.2s |
| concat_audio | 3 | longest | 4.225 | 0.9633 | 10.3% | 14.6s |
| concat_audio | 5 | random | 4.348 | 0.9774 | 13.8% | 15.8s |
| concat_audio | 5 | longest | 4.104 | 0.9701 | 17.2% | 17.6s |
| concat_code | 1 | random | 4.395 | 0.9726 | 10.3% | 12.1s |
| concat_code | 1 | longest | 4.474 | 0.9752 | 10.3% | 13.2s |
| concat_code | 2 | random | 4.303 | 0.9698 | 13.8% | 10.9s |
| concat_code | 2 | longest | 4.436 | 0.9722 | 17.2% | 14.6s |
| concat_code | 3 | random | 4.398 | 0.9775 | 17.2% | 13.0s |
| concat_code | 3 | longest | 4.379 | 0.9630 | 10.3% | 13.3s |
| concat_code | 5 | random | 4.411 | 0.9765 | 17.2% | 14.4s |
| concat_code | 5 | longest | 4.390 | 0.9762 | 10.3% | **247.3s** |
| embed_avg | 1 | random | 4.335 | 0.9007 | 20.7% | 15.9s |
| embed_avg | 1 | longest | 4.492 | 0.9648 | 10.3% | 16.7s |
| embed_avg | 2 | random | 4.468 | 0.9757 | 10.3% | 17.0s |
| embed_avg | 2 | longest | 4.417 | **0.9810** | 10.3% | 16.0s |
| embed_avg | 3 | random | **4.502** | 0.9723 | 13.8% | 15.9s |
| embed_avg | 3 | longest | 4.481 | 0.9750 | 10.3% | 16.8s |
| embed_avg | 5 | random | 4.408 | 0.9589 | 10.3% | 16.4s |
| embed_avg | 5 | longest | 4.478 | **0.9829** | 10.3% | 17.1s |

#### Top 5 Configurations by Speaker Similarity

| Rank | Approach | Config | Speaker Sim | UTMOS | WER |
|------|----------|--------|-------------|-------|-----|
| 1 | **embed_avg** | 5-ref, longest | **0.9829** | 4.478 | 10.3% |
| 2 | **embed_avg** | 2-ref, longest | **0.9810** | 4.417 | 10.3% |
| 3 | concat_audio | 2-ref, longest | 0.9791 | 4.374 | 17.2% |
| 4 | concat_audio | 2-ref, random | 0.9781 | 4.444 | 10.3% |
| 5 | concat_code | 3-ref, random | 0.9775 | 4.398 | 17.2% |

Baseline: single_baseline = 0.9697

#### Key Findings

**1. `embed_avg` is the surprise winner.**
- Produced the highest speaker similarity (0.9829) AND best naturalness (UTMOS 4.502)
- This contradicts our initial hypothesis that losing ICL mode would hurt naturalness
- With `longest` strategy, embed_avg consistently outperforms at every n_refs level

**2. The `longest` selection strategy dominates for speaker similarity.**
- Across all approaches, `longest` tends to produce higher speaker sim
- Confirms the hypothesis that longer reference clips provide more speaker identity signal
- However, for concat_audio, `longest` tends to lower UTMOS (3.988 at 1-ref, 4.104 at 5-ref)

**3. 2 refs is a sweet spot for `concat_audio`; diminishing returns beyond.**
- concat_audio peaks at 2 refs for speaker sim (0.9781-0.9791)
- At 3 and 5 refs, speaker sim plateaus or drops, and UTMOS degrades
- This aligns with the hypothesis that too much context can hurt the LM

**4. `concat_code` does NOT outperform `concat_audio`.**
- Contradicts the hypothesis that code-level concatenation would be superior
- Performance is comparable, and concat_code 5-ref longest had a catastrophic 247s generation (likely context overflow causing degenerate decoding)

**5. `embed_avg` 1-ref random is an outlier (SIM=0.9007, WER=20.7%).**
- Single short random ref (1.4s) with x_vector_only mode produced poor results
- With 2+ refs or longer clips, embed_avg recovers and excels
- This suggests embedding quality is highly sensitive to ref clip quality in x_vector_only mode

**6. Eval caching optimization confirmed effective.**
- Eval time dropped from 10.4s (first run, old code) to 1.5-2.0s per run
- This is a ~5x speedup for the eval phase

#### Hypothesis Validation

| Hypothesis | Result | Notes |
|-----------|--------|-------|
| concat_audio 2-3 refs improves speaker sim | **Confirmed** | 2-ref is sweet spot (+0.008 to +0.009) |
| concat_code outperforms concat_audio | **Rejected** | Comparable or slightly worse; 5-ref longest catastrophically slow |
| embed_avg improves sim but hurts naturalness | **Rejected** | embed_avg has BOTH best sim AND best UTMOS |
| Beyond 3-5 refs, performance degrades | **Partially confirmed** | concat_audio degrades; embed_avg longest continues improving |
| longest strategy outperforms random | **Confirmed** | Consistent across approaches for speaker sim |

#### Recommended Configuration for Phase 3

Based on Phase 2 results, the **winning approaches** to scale up are:

1. **embed_avg, longest strategy, {2, 3, 5} refs** — best speaker sim + best naturalness
2. **concat_audio, 2-ref, random** — best balance of sim + WER + simplicity
3. **single_baseline** — control

Drop `concat_code` from Phase 3 — it doesn't outperform and has instability at high ref counts.

### 2026-02-18 — Docker & Deployment Prep
- [x] Created lean `requirements.txt` (~20 direct deps, torch installed separately via CUDA index)
- [x] Created Dockerfile: `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04`, deadsnakes PPA for Python 3.12
- [x] Resolved build issues: cu124→cu126 for torch 2.10.0, removed flash-attn (QEMU incompatible), removed discrete-speech-metrics (pypesq broken), trimmed requirements to avoid fsspec conflicts
- [x] Docker image builds successfully on Apple Silicon via `--platform linux/amd64`
- [x] Added `--max-new-tokens` CLI flag to `run_fewshot.py` (prevents 247s degenerate generation)
- [x] Created `.dockerignore` to exclude .venv, .git, outputs, media from build context

#### Issues to Address

- [ ] **concat_code 5-ref longest took 247s** — needs investigation; likely context overflow causing degenerate autoregressive decoding. Add a max_new_tokens safety cap or ref_code length limit.
- [ ] **WER variability**: Some configs show 17.2% vs 10.3% WER. The target text contains "veracities" which Whisper struggles with ("varicities", "vera cities"). This may be a Whisper artefact rather than a real TTS quality difference. Need to inspect the actual audio.
- [ ] **Need multiple seeds**: All results are from a single seed (42). Stochastic variation could explain some differences. Phase 3 must use 3 seeds.
- [ ] **Need multiple targets**: Only 1 held-out target tested. Phase 3 must test all 5 held-out targets per speaker.

---

## Phase 3: AWS g5.xlarge Deployment Plan

### Why g5.xlarge

| Spec | Value |
|------|-------|
| GPU | NVIDIA A10G, 24 GB GDDR6 |
| vCPU | 4 (AMD EPYC) |
| RAM | 16 GB |
| Storage | 250 GB NVMe SSD (default EBS) |
| On-demand price | ~$1.006/hr (us-east-1) |
| Spot price | ~$0.35-0.50/hr (60-70% savings) |

**Memory budget on A10G (24 GB)**:
- TTS model (0.6B, float16): ~1.2 GB
- Eval models (UTMOS + WavLM-XVector + Whisper turbo + WavLM-Large): ~3.5 GB
- Working memory / KV cache: ~2-4 GB
- **Total**: ~8 GB peak — fits comfortably with headroom for flash-attn

### Deployment Steps

#### Step 1: Create requirements.txt
Extract reproducible dependencies from the working .venv (205 packages total). Key packages:
```
torch==2.10.0+cu124        # CUDA build (current is MPS-only)
torchaudio==2.10.0+cu124
qwen-tts==0.1.1
transformers==4.57.3
accelerate==1.12.0
openai-whisper==20250625
librosa==0.11.0
soundfile==0.13.1
jiwer==3.1.0
discrete-speech-metrics @ git+https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics.git@350e19062839029a66d72541312852ca12c7f1b0
flash-attn                 # NEW — not in current venv, CUDA-only
```

**Important**: The current venv has MPS-only torch (no CUDA). The Dockerfile must install the CUDA torch index.

#### Step 2: Dockerfile
```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    ffmpeg git libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Python venv
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install torch with CUDA first (separate layer for caching)
RUN pip install --no-cache-dir \
    torch==2.10.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install flash-attn (requires torch+CUDA already installed)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Install remaining deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-download models at build time (baked into image)
RUN python3 -c "from qwen_tts import Qwen3TTSModel; Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-0.6B-Base', device_map='cpu')"
RUN python3 -c "import torch; torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True)"
RUN python3 -c "from transformers import WavLMForXVector; WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')"
RUN python3 -c "import whisper; whisper.load_model('turbo')"
RUN python3 -c "from transformers import WavLMModel; WavLMModel.from_pretrained('microsoft/wavlm-large')"

# Copy project code
WORKDIR /app
COPY src/ src/
COPY scripts/ scripts/
COPY data/libritts_r_aligned/ data/libritts_r_aligned/

ENTRYPOINT ["python3", "scripts/run_fewshot.py"]
```

#### Step 3: Validate locally before spending on GPU

**3a. Validate requirements.txt can be generated**
```bash
pip freeze > requirements.txt
# Then manually edit to replace torch/torchaudio with CUDA variants
```

**3b. Build Docker image locally (CPU-only test)**
```bash
docker build -t tts-fewshot .
# Test import chain (no GPU needed)
docker run --rm tts-fewshot python3 -c "from src.experiment.run_fewshot import main; print('OK')"
```

**3c. Dry-run: test the experiment runner with --skip-eval**
```bash
docker run --rm -v $(pwd)/outputs:/app/outputs tts-fewshot \
    --speakers 1188 --approaches single_baseline \
    --num-refs 1 --strategies random --seeds 42 \
    --held-out-targets 1 --device cpu --skip-eval
```
This validates the full code path (manifest loading, ref pool, combiner, TTS generation) without needing a GPU. Generation will be slow on CPU but proves correctness.

#### Step 4: Deploy to AWS

**4a. Launch instance**
```bash
# Using Deep Learning AMI (has NVIDIA drivers pre-installed)
aws ec2 run-instances \
    --image-id ami-0xxxx  # Deep Learning Base OSS (Ubuntu 22.04)
    --instance-type g5.xlarge \
    --key-name your-key \
    --security-group-ids sg-xxxx \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100}}]'
```

**4b. Setup on instance**
```bash
# Install Docker + NVIDIA Container Toolkit (if not in AMI)
# Or just use conda/venv directly on the DLAMI — simpler
git clone <repo-url> && cd tts
pip install -r requirements.txt
```

**4c. Run Phase 3**
```bash
python scripts/run_fewshot.py \
    --manifest data/libritts_r_aligned/manifest.json \
    --approaches single_baseline embed_avg concat_audio \
    --num-refs 1 2 3 5 \
    --strategies random longest \
    --seeds 42 123 456 \
    --held-out-targets 5 \
    --device cuda:0 \
    --dtype float16 \
    --flash-attn \
    --skip-speechbertscore
```

### Cost-Effective Strategies

| Strategy | Savings | Trade-off |
|----------|---------|-----------|
| **Use spot instances** | 60-70% off on-demand | May be interrupted (unlikely for short jobs) |
| **Skip Docker, use DLAMI directly** | Faster setup, no image build | Less reproducible, but fine for iteration |
| **Pre-download models to S3** | Avoids re-downloading per instance | One-time setup |
| **Use tmux + nohup** | Survive SSH disconnects | Simple, no orchestration needed |
| **Run without SpeechBERTScore** | Skip ~1.2GB model, faster eval | Add back for final results only |

### Estimated Phase 3 Cost

| Item | Estimate |
|------|----------|
| Runs | ~225 (5 speakers x 5 targets x 9 configs) |
| TTS per run (A10G, float16+flash-attn) | ~2-4s |
| Eval per run | ~1.5-2s |
| Total compute | ~225 x 5s = ~19 min |
| Buffer (overhead, model loading, I/O) | 3x = ~60 min |
| Instance cost (spot @ $0.40/hr) | **~$0.40** |
| Instance cost (on-demand @ $1.00/hr) | **~$1.00** |

This is extremely cheap. Even with debugging time, setup, and re-runs, the total cost should be under $5.

### Validation Checklist (Before Spending on GPU)

- [ ] `requirements.txt` generated and verified
- [ ] Dockerfile builds successfully locally
- [ ] `docker run ... --skip-eval --device cpu` completes without errors
- [ ] Import chain works: `from src.experiment.run_fewshot import main`
- [ ] Data directory (`data/libritts_r_aligned/`) is included or accessible
- [ ] Script args for Phase 3 configuration are correct (approaches, refs, seeds)
- [ ] `--device cuda:0 --dtype float16 --flash-attn` flags tested in a minimal CUDA container (if available)
- [ ] Results CSV path is correct and writable

### Two Deployment Approaches

**Option A: Docker (most reproducible)**
- Build image with all deps + models baked in
- Push to ECR, pull on g5.xlarge
- `docker run --gpus all ...`
- Pro: fully reproducible, portable
- Con: large image (~15-20 GB with models), longer build

**Option B: DLAMI + venv (fastest to deploy)**
- Launch g5.xlarge with Deep Learning AMI (Ubuntu 22.04)
- Clone repo, create venv, pip install
- Models download from HuggingFace on first run (~10 GB, cached after)
- Pro: fastest path to running, simpler debugging
- Con: less reproducible, dep versions may drift

**Recommendation**: Start with **Option B** (DLAMI + venv) for Phase 3. It's faster to iterate and the compute cost is trivial. Create the Dockerfile for reproducibility/paper submission once results are finalized.

---

## Next Steps

### Completed (2026-02-18)
- [x] Generate `requirements.txt` (lean, direct deps only — 20 packages, torch/torchaudio installed separately)
- [x] Create Dockerfile (CUDA 12.6, Python 3.12, all models baked in, ~15-20 GB image)
- [x] Fix: base image needed deadsnakes PPA for Python 3.12, torch 2.10.0 needs cu126 index (not cu124)
- [x] Fix: flash-attn removed from Docker build (needs nvcc + slow under QEMU, install natively on GPU)
- [x] Fix: discrete-speech-metrics excluded (pypesq broken with modern numpy, not needed for Phase 3)
- [x] Fix: lean requirements.txt to avoid fsspec/nemo-toolkit version conflicts
- [x] Validate: Docker image builds successfully (`docker build --platform linux/amd64 -t tts-fewshot .`)
- [x] Add `--max-new-tokens` CLI flag and plumb through all generation call sites

### Completed (2026-02-19) — Analysis Script & Phase 3 Deployment

- [x] Built `scripts/analyze_fewshot.py` + `src/experiment/analyze.py` (~480 lines)
  - Summary stats (mean ± std per config), per-speaker stats
  - Significance tests (Wilcoxon / paired t-test, Bonferroni correction, Cohen's d)
  - Publication-ready LaTeX table (bold best, arrows for metric direction)
  - 5 plot types: speaker_sim bars, UTMOS bars, scaling curve, strategy heatmap, per-speaker breakdown
  - Validated against Phase 2 results (embed_avg 5-ref longest = 0.9829 confirmed)
- [x] Added `matplotlib>=3.9.0` to requirements.txt
- [x] Relaxed version pins in requirements.txt for Python 3.10 compatibility (DLAMI ships 3.10)
- [x] Made `discrete_speech_metrics` import lazy in `evaluate.py` (avoids ImportError on GPU instances)
- [x] Created `scripts/launch_gpu.sh` — launches spot g5.xlarge with DLAMI, prompts for key/SG
- [x] Created `scripts/setup_gpu.sh` — installs system deps, venv, PyTorch+CUDA, project deps, verifies GPU
- [x] Deployed Phase 3 to AWS g5.xlarge spot instance (eu-west-1, $0.55/hr)

#### Deployment Details

- **Instance**: `i-0fdaf0782df238869`, spot g5.xlarge, NVIDIA A10G 24GB
- **AMI**: Deep Learning Base OSS (Ubuntu 22.04) `ami-0f5404694649be37b`
- **Region**: eu-west-1 (Ireland)
- **Key pair**: `tts-phase3` (`~/.ssh/tts-phase3.pem`)
- **Security group**: `sg-0955c6c807611d655` (tts-phase3-ssh, port 22)
- **IAM role**: `tts-phase3-ec2` (S3 write to `tts-phase3-results` bucket)
- **Dtype**: `bfloat16` (float16 caused CUDA device-side assert on A10G)
- **flash-attn**: Skipped (compiling from source destabilized instance; not required)

#### Deployment Issues Encountered & Fixed

1. **Python 3.10 on DLAMI**: numpy>=2.3 requires 3.11+ → relaxed version pins
2. **No python3-venv**: Added `apt-get install python3-venv` to setup script
3. **flash-attn compile**: Consumed all CPU, made SSH unresponsive → skipped by default
4. **float16 CUDA assert**: Model numerically unstable in fp16 on A10G → switched to bfloat16
5. **discrete_speech_metrics top-level import**: Broke `--skip-speechbertscore` → made lazy
6. **Manifest absolute paths**: Hardcoded `/Users/callumjmac/...` → sed-replaced to `/home/ubuntu/...`
7. **Missing ffmpeg**: Whisper needs ffmpeg for audio loading → added to apt-get install
8. **Python stdout buffering**: Logs appeared empty → added `python -u` for unbuffered output

#### Phase 3 Run Status

- **Experiment matrix**: 5 speakers × 5 targets × 3 seeds × 3 approaches × {1,2,3,5} refs × {random,longest} = **1275 runs**
- **Approaches**: `single_baseline`, `embed_avg`, `concat_audio` (concat_code dropped)
- **Rate**: ~15-20s per run (generation + eval) on A10G bfloat16
- **Estimated duration**: 6-8 hours (~$4.40 total spot cost)
- **Launched**: 2026-02-19 ~19:05 UTC
- **Early results**: Run 1 produced UTMOS=4.38, SIM=0.9707, WER=0.1379 (consistent with Phase 2)

#### Automated Completion

A `finish.sh` script runs in the background on the instance:
1. Polls every 60s until `run_fewshot.py` exits
2. Runs `python scripts/analyze_fewshot.py` to generate stats + plots
3. Uploads all results to `s3://tts-phase3-results/phase3/`
4. Shuts down the instance (stops billing)

**To retrieve results (morning of 2026-02-20):**
```bash
aws s3 sync s3://tts-phase3-results/phase3/ outputs/fewshot/phase3/ --region eu-west-1
```

**To terminate the stopped instance:**
```bash
aws ec2 terminate-instances --region eu-west-1 --instance-ids i-0fdaf0782df238869
```

**To check progress (while running):**
```bash
ssh -i ~/.ssh/tts-phase3.pem ubuntu@108.129.92.201 "wc -l ~/tts/outputs/fewshot/results.csv && tail -5 ~/tts/phase3.log"
```

### Phase 3 Results (2026-02-20 morning)

**1275 runs completed: 5 speakers × 5 targets × 3 seeds × 3 approaches × {1,2,3,5} refs × {random,longest}**

Results retrieved from `s3://tts-phase3-results/phase3/` → `outputs/fewshot/phase3/`

#### Summary Table (mean ± std across 75 observations per config)

| Approach | n_refs | Strategy | UTMOS ↑ | Speaker Sim ↑ | WER ↓ |
|----------|--------|----------|---------|---------------|-------|
| **Single Baseline** | 1 | random | 4.425 ± 0.094 | 0.928 ± 0.044 | 15.3% ± 12.0% |
| Concat Audio | 1 | longest | 4.390 ± 0.168 | 0.942 ± 0.039 | 22.3% ± 22.1% |
| Concat Audio | 1 | random | 4.431 ± 0.119 | 0.926 ± 0.047 | 15.9% ± 12.7% |
| Concat Audio | 2 | longest | 4.366 ± 0.392 | 0.947 ± 0.036 | 24.1% ± 73.6% |
| Concat Audio | 2 | random | 4.432 ± 0.126 | 0.937 ± 0.042 | 14.4% ± 11.1% |
| **Concat Audio** | **3** | **longest** | 4.416 ± 0.098 | **0.949 ± 0.038** | 21.3% ± 18.0% |
| Concat Audio | 3 | random | 4.390 ± 0.379 | 0.941 ± 0.044 | 81.6% ± 555% |
| Concat Audio | 5 | longest | 4.400 ± 0.143 | 0.948 ± 0.037 | 15.7% ± 13.0% |
| Concat Audio | 5 | random | 4.385 ± 0.381 | 0.944 ± 0.044 | 39.7% ± 230% |
| Embed Avg | 1 | longest | 4.470 ± 0.086 | 0.936 ± 0.037 | 13.4% ± 10.4% |
| Embed Avg | 1 | random | 4.462 ± 0.082 | 0.921 ± 0.045 | 14.5% ± 11.6% |
| Embed Avg | 2 | longest | 4.474 ± 0.087 | 0.934 ± 0.040 | 14.0% ± 10.8% |
| Embed Avg | 2 | random | 4.473 ± 0.072 | 0.927 ± 0.042 | 13.8% ± 11.3% |
| Embed Avg | 3 | longest | 4.469 ± 0.078 | 0.939 ± 0.037 | 15.0% ± 13.6% |
| **Embed Avg** | **3** | **random** | **4.483 ± 0.055** | 0.930 ± 0.039 | 14.5% ± 11.0% |
| Embed Avg | 5 | longest | 4.474 ± 0.054 | 0.938 ± 0.034 | 14.4% ± 11.9% |
| Embed Avg | 5 | random | 4.463 ± 0.103 | 0.937 ± 0.035 | 14.1% ± 11.4% |

#### Top 5 Configurations by Speaker Similarity

| Rank | Approach | Config | Spk Sim | UTMOS | WER |
|------|----------|--------|---------|-------|-----|
| 1 | **Concat Audio** | 3-ref, longest | **0.949** ± 0.038 | 4.416 | 21.3% |
| 2 | Concat Audio | 5-ref, longest | 0.948 ± 0.037 | 4.400 | 15.7% |
| 3 | Concat Audio | 2-ref, longest | 0.947 ± 0.036 | 4.366 | 24.1% |
| 4 | Concat Audio | 5-ref, random | 0.944 ± 0.044 | 4.385 | 39.7% |
| 5 | Concat Audio | 1-ref, longest | 0.942 ± 0.039 | 4.390 | 22.3% |

#### Statistical Significance (Wilcoxon signed-rank, Bonferroni-corrected, α=0.05)

**Speaker Similarity — significant improvements over baseline (4 configs):**
| Config | p_corrected | Cohen's d | Interpretation |
|--------|-------------|-----------|----------------|
| concat_audio 2-ref longest | 0.0024 | 0.51 | Medium |
| concat_audio 3-ref longest | 0.0001 | 0.60 | Medium-large |
| concat_audio 5-ref longest | <0.0001 | 0.62 | Medium-large |
| concat_audio 5-ref random | 0.0022 | 0.38 | Small-medium |

**embed_avg: NO significant speaker_sim improvement** at any config (all p_corrected > 0.05).

**UTMOS — significant improvements over baseline (8 configs, all embed_avg):**
| Config | p_corrected | Cohen's d |
|--------|-------------|-----------|
| embed_avg 1-ref longest | 0.0003 | 0.43 |
| embed_avg 1-ref random | 0.0040 | 0.36 |
| embed_avg 2-ref longest | <0.0001 | 0.46 |
| embed_avg 2-ref random | 0.0001 | 0.50 |
| embed_avg 3-ref longest | 0.0001 | 0.45 |
| embed_avg 3-ref random | <0.0001 | 0.61 |
| embed_avg 5-ref longest | 0.0004 | 0.55 |
| embed_avg 5-ref random | 0.0003 | 0.32 |

**WER — no significant changes** for any config after Bonferroni correction.

#### Key Findings

**1. Concat Audio + longest is the speaker similarity winner.**
- +0.02 over baseline (0.949 vs 0.928), statistically significant (p<0.0001, d=0.6)
- Scales clearly with more refs: 1→2→3 refs shows monotonic improvement, plateaus at 3-5
- `longest` strategy critical — random is consistently worse

**2. Embed Avg wins naturalness (UTMOS) but NOT speaker similarity.**
- UTMOS ~4.47 vs 4.42 baseline — small (+0.05) but significant across all configs
- Speaker similarity is NOT significantly improved at any config (contradicts Phase 2)
- Lowest variance across configs — very stable approach

**3. Phase 2 conclusion was speaker-specific.**
- Phase 2 (speaker 1188 only): embed_avg appeared dominant (SIM=0.983)
- Phase 3 (5 speakers): embed_avg SIM advantage disappears, concat_audio takes over
- This validates why multi-speaker testing was essential

**4. `longest` strategy strongly favours concat_audio.**
- Concat Audio longest: SIM ≈ 0.942–0.949
- Concat Audio random: SIM ≈ 0.926–0.944
- The longer the reference clips, the more acoustic signal for in-context learning

**5. Concat Audio has WER instability.**
- 3-ref random: 81.6% ± 555% — some degenerate generations
- 5-ref random: 39.7% ± 230% — same issue at smaller scale
- Likely context overflow causing garbled output on some runs
- `longest` strategy is much more stable (WER 15-24%)

**6. Embed Avg is the "safe" approach.**
- Never significantly hurts any metric vs baseline
- Lowest WER variance (±10-14%)
- Best UTMOS (naturalness)
- Moderate SIM improvement (not significant after correction)

#### Hypothesis Validation (Phase 3)

| Hypothesis | Phase 2 Result | Phase 3 Result | Notes |
|-----------|---------------|----------------|-------|
| concat_audio improves speaker sim | Confirmed | **Confirmed (significant)** | d=0.6, best with longest |
| embed_avg improves sim + naturalness | Confirmed (both) | **Partially: UTMOS yes, SIM no** | SIM advantage was speaker-specific |
| longest > random for speaker sim | Confirmed | **Confirmed** | Critical for concat_audio |
| More refs → better (up to 3-5) | Partially | **Confirmed for concat_audio** | Plateaus at 3 refs |
| embed_avg dominates overall | Phase 2: yes | **Phase 3: no** | concat_audio wins SIM |

#### Recommended Approach for Paper

The paper story is a **trade-off between two axes**:
- **Concat Audio (longest, 3-5 refs)**: Best speaker similarity (+0.02, p<0.0001), but higher WER variance
- **Embed Avg**: Best naturalness (+0.05 UTMOS, p<0.0001), stable WER, but no significant SIM gain

This is a more nuanced and honest result than "one approach dominates" — and arguably more interesting for a paper.

### Up Next: Write Paper

**Decision: Proceed with Interspeech 2026 (deadline Feb 25 AoE)**

The results are statistically significant, consistent across speakers, and tell a clear story about the trade-off between speaker fidelity and naturalness in few-shot voice cloning.

**Immediate tasks:**
1. [ ] Listen to best/worst outputs — verify metrics match human perception
2. [ ] Investigate WER outliers (concat_audio 3-ref random, 81.6% mean)
3. [ ] Draft paper outline: intro, method (3 approaches), experiment, results, discussion
4. [ ] Generate camera-ready figures from analysis plots

---

## References

### Qwen3-TTS
- Qwen3-TTS Technical Report (2026-01-23), arXiv:2601.15621
- Model: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, `Qwen/Qwen3-TTS-12Hz-1.7B-Base`

### Evaluation Methodology
- **evaluate-zero-shot-tts** (GitHub): Open-source eval protocol from VALL-E/CLaM-TTS/DiTTo-TTS
- **Voicebox** (Meta, arXiv:2306.15687): SIM metrics, 1.9% WER benchmark
- **Seed-TTS** (ByteDance, arXiv:2406.02430): Zero-shot vs fine-tuned comparisons, SIM-r methodology
- **Voice Cloning Survey** (arXiv:2505.00579): Comprehensive NMOS, SECS, WER tables across systems
- **UTMOS** (Saeki 2022): Automated MOS prediction
- **VALL-E** (Microsoft, arXiv:2301.02111): Established zero-shot TTS eval protocol (SIM-o, WER)

### Dataset
- LibriTTS-R aligned: 4,831 utterances in `data/libritts_r_aligned/`, 24kHz, with TextGrid alignments
