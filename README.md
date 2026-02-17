# tts

Minimal CLI workflow for reference-audio prep, voice cloning, and WAV stitching.

## Requirements

- `python3`
- `ffmpeg` (`brew install ffmpeg`)
- `whisper` CLI (only needed when auto-generating reference text)
- Qwen runtime deps (`qwen_tts`, `torch`, `soundfile`) for voice cloning

## Commands

Convert reference videos to WAV:

```bash
python3 scripts/wav_mov.py ref_video --output-dir ref_audio --overwrite
```

Run Qwen voice clone (video-first mode):

```bash
python3 scripts/qwen_voice_clone.py --ref-video ref_video/david_1.mov --target-text-file target_text/david_1.txt --device mps --dtype float32
```

Stitch generated WAVs in speaker order:

```bash
python3 scripts/stitch_good_wavs.py --input-dir outputs/v2 --output outputs/v2/stitched.wav
```

## Evaluation

Evaluate voice clone quality across four metrics:

```bash
python3 scripts/evaluate.py outputs/v2/matthew_1.wav \
    --ref-audio ref_audio/matthew_1.wav \
    --target-text-file target_text/v2/matthew.txt
```

Use `--format json` for machine-readable output. Skip individual metrics with `--skip-utmos`, `--skip-speaker-sim`, `--skip-wer`, `--skip-speechbertscore`.

### Metrics

| Metric | What it measures | How to interpret | Limitations |
|---|---|---|---|
| **UTMOS** | Speech naturalness (predicted MOS) | 1-5 scale. >4.0 is good, >4.3 is excellent. Human speech typically scores 4.0-4.5. | Reference-free — does not measure speaker similarity. Trained on English challenge data; may not generalise to all voices/languages. |
| **Speaker Similarity** | Whether the clone sounds like the target speaker (WavLM cosine similarity) | -1 to 1. >0.85 strong match, >0.95 excellent. Different speakers typically score 0.6-0.85. | Captures voice identity, not prosody or speaking style. Threshold is dataset-dependent. Primarily English-trained. |
| **WER / CER** | Intelligibility — can the words be understood? (Whisper ASR transcription vs target text) | 0% is perfect. WER <10% excellent, <20% good. CER is typically lower than WER for the same audio. | Conflates ASR errors with TTS errors. Inflated when target text contains fillers (um, uh) that the model cleans up. |
| **SpeechBERTScore** | Fine-grained acoustic similarity using WavLM-Large neural features | 0-1 (precision, recall, F1). Higher is better. Expect moderate scores (~0.7) when ref and generated say different text. | Computationally heavy. Scores are lower when comparing different utterances — best suited for same-text resynthesis evaluation. |
