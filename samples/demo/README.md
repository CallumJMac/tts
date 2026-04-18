# Demo Samples

Three audio samples comparing multi-reference voice cloning strategies on the same speaker (Boris Johnson, LibriTTS-R) and target text.

**Target text:**
> "When multiple reference utterances are available, how should they be combined? We find a clear trade-off: concatenating references maximises speaker identity, while averaging embeddings maximises naturalness."

| File | Strategy | Conditioning | What to listen for |
|------|----------|-------------|-------------------|
| `01_baseline_single.wav` | Single reference | ICL prompt | Baseline voice identity and naturalness |
| `02_concat_longest_3.wav` | Concat, 3 refs (longest) | ICL prompt | Stronger speaker identity — does it sound more like the reference? |
| `03_embed_avg_3.wav` | Embed avg, 3 refs | x-vector only | Smoother, more natural delivery — is the speech more fluent? |

## Regenerating

```bash
.venv/bin/python scripts/generate_samples.py \
  --ref-audio ref_audio/boris_1.wav \
  --ref-text ref_text/boris_1.txt \
  --extra-refs ref_audio/boris_2.wav \
  --extra-ref-texts ref_text/boris_2.txt \
  --output-dir samples/demo \
  --device mps \
  --dtype float32
```
