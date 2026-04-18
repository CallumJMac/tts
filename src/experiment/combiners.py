"""Multi-reference combiners for few-shot voice cloning experiments.

Each combiner takes a list of RefItems and produces the inputs needed
for a single generate_voice_clone() call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

from .ref_pool import RefItem


@dataclass
class CombinedRef:
    """Output of a combiner — ready to feed into generate_voice_clone()."""

    audio_path: Optional[str] = None
    audio_array: Optional[np.ndarray] = None
    audio_sr: int = 16000
    text: Optional[str] = None
    # For embed_avg / concat_code, we pass pre-built prompt items
    voice_clone_prompt: Optional[object] = None
    x_vector_only_mode: bool = False


def _load_and_resample(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio and resample to target_sr."""
    wav, _ = librosa.load(path, sr=target_sr, mono=True)
    return wav


class ConcatAudioCombiner:
    """Concatenate N reference audio files and their transcripts.

    Produces a single longer (audio, text) pair that can be passed
    directly to the existing generate_voice_clone() API.
    """

    def __init__(self, silence_ms: int = 300, target_sr: int = 16000):
        self.silence_ms = silence_ms
        self.target_sr = target_sr

    def combine(self, refs: list[RefItem]) -> CombinedRef:
        if not refs:
            raise ValueError("No refs to combine")

        silence_samples = int(self.target_sr * self.silence_ms / 1000)
        silence = np.zeros(silence_samples, dtype=np.float32)

        segments: list[np.ndarray] = []
        texts: list[str] = []
        for ref in refs:
            wav = _load_and_resample(ref.path, self.target_sr)
            if segments:
                segments.append(silence)
            segments.append(wav)
            texts.append(ref.text.strip())

        combined_audio = np.concatenate(segments)
        combined_text = " ".join(texts)

        return CombinedRef(
            audio_array=combined_audio,
            audio_sr=self.target_sr,
            text=combined_text,
        )


class EmbedAvgCombiner:
    """Average speaker embeddings from multiple refs.

    Uses x_vector_only_mode=True (no ICL), so only the speaker embedding
    is used for conditioning.  Requires the Qwen3TTSModel to extract
    embeddings.
    """

    def combine(
        self,
        refs: list[RefItem],
        model: object,
    ) -> CombinedRef:
        """Combine refs by averaging speaker embeddings.

        Args:
            refs: Reference items to combine.
            model: Qwen3TTSModel instance (needed for create_voice_clone_prompt).
        """
        import torch

        if not refs:
            raise ValueError("No refs to combine")

        # Build individual prompt items to get speaker embeddings
        prompt_items = []
        for ref in refs:
            items = model.create_voice_clone_prompt(
                ref_audio=ref.path,
                ref_text=ref.text,
                x_vector_only_mode=True,
            )
            prompt_items.append(items[0])

        # Average the speaker embeddings
        embeddings = torch.stack([item.ref_spk_embedding for item in prompt_items])
        avg_embedding = embeddings.mean(dim=0)

        # Build a synthetic prompt item with the averaged embedding
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

        combined_item = VoiceClonePromptItem(
            ref_code=None,
            ref_spk_embedding=avg_embedding,
            x_vector_only_mode=True,
            icl_mode=False,
        )

        return CombinedRef(
            voice_clone_prompt=[combined_item],
            x_vector_only_mode=True,
        )


class ConcatCodeCombiner:
    """Concatenate ref_code tensors and ref_texts at the prompt level.

    This works at the model's native representation — tokenized audio
    codes are concatenated along the time axis, texts are joined, and
    speaker embeddings are averaged.
    """

    def combine(
        self,
        refs: list[RefItem],
        model: object,
    ) -> CombinedRef:
        """Combine refs by concatenating codec tokens and texts.

        Args:
            refs: Reference items to combine.
            model: Qwen3TTSModel instance.
        """
        import torch

        if not refs:
            raise ValueError("No refs to combine")

        prompt_items = []
        for ref in refs:
            items = model.create_voice_clone_prompt(
                ref_audio=ref.path,
                ref_text=ref.text,
                x_vector_only_mode=False,
            )
            prompt_items.append(items[0])

        # Concatenate ref_codes along time axis
        ref_codes = [item.ref_code for item in prompt_items if item.ref_code is not None]
        if not ref_codes:
            raise ValueError("No ref_codes found in prompt items")
        combined_code = torch.cat(ref_codes, dim=0)

        # Average speaker embeddings
        embeddings = torch.stack([item.ref_spk_embedding for item in prompt_items])
        avg_embedding = embeddings.mean(dim=0)

        # Concatenate texts
        combined_text = " ".join(ref.text.strip() for ref in refs)

        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

        combined_item = VoiceClonePromptItem(
            ref_code=combined_code,
            ref_spk_embedding=avg_embedding,
            x_vector_only_mode=False,
            icl_mode=True,
            ref_text=combined_text,
        )

        return CombinedRef(
            voice_clone_prompt=[combined_item],
            text=combined_text,
        )
