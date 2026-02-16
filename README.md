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
