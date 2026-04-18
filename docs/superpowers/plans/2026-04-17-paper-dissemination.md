# TTS Paper Dissemination Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Disseminate the multi-reference voice cloning paper quickly to establish priority and maximise reach before findings become stale.

**Architecture:** Fix one key framing issue in the paper → post arXiv immediately to establish timestamp priority → submit NeurIPS 2026 (deadline May 6) → fallback to TASLP journal.

**Key Deadlines:**
- arXiv: ASAP (this week)
- NeurIPS 2026 abstract: May 4, 2026
- NeurIPS 2026 full paper: May 6, 2026
- NeurIPS 2026 workshops: ~August/September 2026

---

### Task 1: Fix the conditioning mode confound framing

**Files:**
- Modify: `paper/main.tex`

The current paper claims "embedding averaging improves naturalness" but the data shows the UTMOS gain appears at n=1 (before any averaging). The real finding on the UTMOS axis is that x-vector conditioning outperforms ICL conditioning. This must be addressed before posting publicly or a reviewer will reject on this basis.

- [ ] **Step 1: Rewrite contribution #2 in the introduction (lines 46-48)**

Replace:
```latex
  \item The discovery of a statistically significant Pareto trade-off: audio concatenation optimises speaker fidelity while embedding averaging optimises naturalness, with medium-to-large effect sizes.
```
With:
```latex
  \item The discovery of a statistically significant Pareto trade-off: audio concatenation (ICL conditioning) optimises speaker fidelity while x-vector conditioning optimises naturalness, with medium-to-large effect sizes. Notably, the naturalness gain appears at $n{=}1$, suggesting it is attributable to the conditioning pathway rather than multi-reference averaging per se.
```

- [ ] **Step 2: Promote the confound caveat in the Discussion (line 332)**

The sentence currently reads:
```
Notably, \textsc{embed} uses the model's \texttt{x\_vector\_only} conditioning pathway, whereas single and \textsc{concat} use prompt-based conditioning; thus part of the naturalness gain may reflect conditioning mode differences rather than averaging alone.
```

Replace "part of the naturalness gain may reflect" with a stronger claim:
```latex
Notably, \textsc{embed} uses the model's \texttt{x\_vector\_only} conditioning pathway, whereas single and \textsc{concat} use prompt-based conditioning. The fact that \textsc{embed} achieves significant UTMOS improvement already at $n{=}1$ (before any averaging occurs) strongly suggests the naturalness gain is primarily attributable to the conditioning pathway rather than multi-reference averaging. This is an important confound: our results characterise the two strategies as deployed, but cannot isolate the averaging effect from the conditioning mode effect.
```

- [ ] **Step 3: Update the abstract to reflect the honest framing**

Current abstract sentence:
```
audio concatenation improves speaker similarity (SIM +0.02, p $<$ 0.001, d up to 0.62) while embedding averaging improves naturalness (UTMOS +0.06, p $<$ 1e-7, d up to 0.61)
```

Replace with:
```latex
audio concatenation (ICL conditioning) improves speaker similarity (SIM $+$0.02, $p{<}0.001$, $d$ up to 0.62) while x-vector conditioning improves naturalness (UTMOS $+$0.06, $p{<}10^{-7}$, $d$ up to 0.61); the naturalness gain appears already at $n{=}1$, implicating the conditioning pathway rather than multi-reference averaging alone
```

- [ ] **Step 4: Build the PDF and verify it compiles**

```bash
cd /Users/callumjmac/Repos/tts/paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
Expected: `main.pdf` produced with no errors.

- [ ] **Step 5: Read through the compiled PDF**

Check: abstract, contributions list, and discussion section read consistently and honestly. The confound is visible to a reviewer but framed as a finding rather than a flaw.

- [ ] **Step 6: Commit**

```bash
git add paper/main.tex
git commit -m "paper: clarify conditioning mode confound in abstract, contributions, and discussion"
```

---

### Task 2: Prepare and post arXiv preprint

**Goal:** Establish public timestamp priority before NeurIPS submission.

- [ ] **Step 1: Check Interspeech 2026 double-blind policy**

Interspeech submissions are typically double-blind. Verify whether posting a preprint before the Interspeech decision violates their anonymity policy. If the Interspeech submission is already in review, check their specific preprint rules before posting.

- [ ] **Step 2: Prepare arXiv submission package**

Compile a clean submission:
```bash
cd /Users/callumjmac/Repos/tts/paper
# Ensure figures are embedded, no broken refs
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Files needed for arXiv:
- `main.tex`
- `refs.bib`
- `figures/pareto_tradeoff.pdf`
- `figures/scaling_curve.pdf`
- `figures/stability_fail_rate.pdf`
- `Interspeech.cls` (or replace with a generic class if required)

- [ ] **Step 3: Create arXiv account (if not already done)**

Go to arxiv.org → Register → verify email.

- [ ] **Step 4: Submit to arXiv**

- Primary category: `eess.AS` (Audio and Speech Processing)
- Cross-list: `cs.SD` (Sound)
- Title: "Multi-Reference Voice Cloning for Qwen3-TTS: A Trade-Off Between Speaker Fidelity and Naturalness"
- Authors: as per paper
- Abstract: paste from `main.tex`

- [ ] **Step 5: Note arXiv ID and share**

Once the arXiv ID is issued (typically next business day), share to:
- Any relevant Slack/Discord communities (e.g. Hugging Face Discord, AI speech channels)
- Twitter/X with tags: `#TTS #VoiceCloning #SpeechSynthesis`
- LinkedIn post with 2-3 sentence summary of the Pareto finding

---

### Task 3: Submit to NeurIPS 2026

**Deadlines:**
- Abstract: May 4, 2026
- Full paper: May 6, 2026

This is a stretch target (~2.5 weeks away) but achievable since the paper is near-complete.

- [ ] **Step 1: Check NeurIPS 2026 scope fit**

NeurIPS is ML-broad. Frame the paper around the **methodology** (large-scale controlled experiment, Pareto trade-off analysis, statistical rigor) rather than the TTS domain. The contribution is empirical ML methodology applied to TTS — that's in scope.

- [ ] **Step 2: Adapt the introduction framing for an ML audience**

NeurIPS reviewers are less familiar with TTS specifics. Add 1-2 sentences in the intro explaining why Qwen3-TTS and LibriTTS-R are standard benchmarks. Current intro assumes speech community knowledge.

- [ ] **Step 3: Strengthen related work**

Current related work (lines 39-41) is thin for NeurIPS. Add:
- CosyVoice multi-speaker specifics
- Seed-TTS multi-reference discussion (if any)
- Any relevant multi-reference/few-shot learning papers from ML literature
- A sentence distinguishing from fine-tuning approaches (YourTTS)

- [ ] **Step 4: Register abstract on NeurIPS submission system by May 4**

Go to NeurIPS 2026 submission portal (openreview.net) → create submission → paste abstract and author list.

- [ ] **Step 5: Submit full paper by May 6**

Upload PDF. Verify all figures are embedded and paper meets NeurIPS formatting requirements (9 pages + references).

- [ ] **Step 6: Commit final NeurIPS version**

```bash
git add paper/main.tex paper/refs.bib
git commit -m "paper: NeurIPS 2026 submission version"
git tag v1.0-neurips2026
```

---

### Task 4: Fallback — IEEE TASLP journal submission

**Use if:** NeurIPS is rejected, or you want parallel journal track.

IEEE Transactions on Audio, Speech, and Language Processing is the top journal for this work. Rolling submission, no deadline.

- [ ] **Step 1: Expand the paper to journal length**

TASLP expects 8-12 pages. Areas to expand:
- Add the conditioning mode ablation if feasible (single x-vector vs. averaged x-vector)
- Expand per-speaker analysis with speaker-level breakdown table
- Add a proper listening test (even small-scale crowdsourced MOS via Prolific)
- Expand related work to full journal depth

- [ ] **Step 2: Run the conditioning mode ablation**

This would significantly strengthen the paper. The experiment is simple:
```bash
# Compare single-ref x-vector vs baseline (already in your data)
# Then compare single-ref prompt-based vs single-ref x-vector (NEW — 1 run per speaker/target/seed)
python scripts/run_fewshot.py --approach embed --n-refs 1 --strategy random
```
This directly tests whether the n=1 embed gain is from conditioning mode or reference quality.

- [ ] **Step 3: Convert to IEEE Transactions format**

Download IEEE TASLP LaTeX template from IEEE Author Center. Adapt `main.tex` to new template.

- [ ] **Step 4: Submit via IEEE ScholarOne Manuscripts**

Go to mc.manuscriptcentral.com/taslp → New Submission.

---

### Task 5: NeurIPS 2026 Workshop track (longer horizon)

**Timeline:** Workshop calls typically open August 2026 for December workshops.

- [ ] **Step 1: Monitor NeurIPS 2026 workshop announcements in August**

Check neurips.cc and openreview.net in August for:
- "Generative Models for Audio" workshops
- "Evaluation of Generative Models" workshops
- "Spoken Language Processing" workshops

- [ ] **Step 2: Prepare a 4-page workshop version**

Condense the paper to key findings only: Pareto figure, Table 1, statistical results. Drop per-speaker analysis. Leads with practical implications.

- [ ] **Step 3: Submit to 1-2 relevant workshops**

Workshop papers are lightly reviewed and a good venue for getting community feedback before the full journal version.

---

## Summary Timeline

| Date | Action |
|------|--------|
| Apr 18-20 | Fix confound framing, compile, commit |
| Apr 21-22 | Post to arXiv |
| Apr 24 - May 3 | Strengthen related work + NeurIPS framing |
| May 4 | NeurIPS abstract deadline |
| May 6 | NeurIPS full paper deadline |
| May onwards | TASLP journal (if running ablation) |
| Aug-Sep 2026 | NeurIPS workshop submissions |
