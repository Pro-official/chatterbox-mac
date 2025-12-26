#!/usr/bin/env python3
"""
Chatterbox TTS Wrapper for truenaad-ai
Provides a consistent interface matching mlx-audio TTS patterns.

Usage:
    from chatterbox_wrapper import ChatterboxTTS

    # Initialize (choose model: 'standard' or 'turbo')
    tts = ChatterboxTTS(model='turbo')

    # Basic generation
    audio = tts.generate("Hello world!")
    tts.save(audio, "output.wav")

    # Voice cloning
    audio = tts.generate("Hello in cloned voice!", ref_audio="reference.wav")
"""

import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Generator, Union

import torch
import torchaudio as ta
import numpy as np

# Apply patches before importing chatterbox
_torch_load_original = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = 'cpu'
    return _torch_load_original(*args, **kwargs)
torch.load = _patched_torch_load

# Fix Perth watermarker for macOS
import perth
perth.PerthImplicitWatermarker = perth.DummyWatermarker


@dataclass
class GenerationResult:
    """Result from TTS generation, matching mlx-audio pattern."""
    audio: np.ndarray
    sample_rate: int
    processing_time_seconds: float
    real_time_factor: float
    duration_seconds: float


class ChatterboxTTS:
    """
    Chatterbox TTS wrapper with mlx-audio compatible interface.

    Supports both standard (500M) and turbo (350M) models.
    Turbo is ~3x faster but English-only.
    """

    MODELS = {
        'standard': 'ResembleAI/chatterbox',
        'turbo': 'ResembleAI/chatterbox-turbo'
    }

    def __init__(
        self,
        model: str = 'turbo',
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize Chatterbox TTS.

        Args:
            model: 'standard' (500M, voice cloning, emotion) or 'turbo' (350M, faster)
            device: 'mps', 'cuda', or 'cpu'. Auto-detected if None.
            verbose: Print loading progress
        """
        self.model_name = model
        self.verbose = verbose

        # Device detection
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        if verbose:
            print(f"[ChatterboxTTS] Using device: {self.device}")
            print(f"[ChatterboxTTS] Loading {model} model...")

        start = time.time()
        self._load_model(model)

        if verbose:
            print(f"[ChatterboxTTS] Model loaded in {time.time()-start:.1f}s")

    def _load_model(self, model: str):
        """Load the specified model."""
        from huggingface_hub import snapshot_download

        if model == 'turbo':
            local_path = snapshot_download(
                repo_id=self.MODELS['turbo'],
                token=None,
                allow_patterns=['*.safetensors', '*.json', '*.txt', '*.pt', '*.model']
            )
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            self.model = ChatterboxTurboTTS.from_local(local_path, self.device)
        else:
            from chatterbox.tts import ChatterboxTTS as StandardTTS
            self.model = StandardTTS.from_pretrained(device=self.device)

        self.sample_rate = self.model.sr

    @property
    def sr(self) -> int:
        """Sample rate of generated audio."""
        return self.sample_rate

    def generate(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            ref_audio: Path to reference audio for voice cloning (6+ seconds)
            exaggeration: Emotion expressiveness (0.25-2.0), standard model only
            cfg_weight: Classifier-free guidance weight (0.2-1.0)
            temperature: Sampling temperature
            seed: Random seed for reproducibility

        Returns:
            GenerationResult with audio, timing info, etc.
        """
        if seed is not None:
            torch.manual_seed(seed)

        start = time.time()

        if self.model_name == 'turbo':
            if ref_audio:
                wav = self.model.generate(text, audio_prompt_path=ref_audio)
            else:
                wav = self.model.generate(text)
        else:
            # Standard model with more options
            kwargs = {
                'exaggeration': exaggeration,
                'cfg_weight': cfg_weight,
            }
            if ref_audio:
                kwargs['audio_prompt_path'] = ref_audio
            wav = self.model.generate(text, **kwargs)

        processing_time = time.time() - start

        # Convert to numpy
        audio_np = wav.cpu().numpy().squeeze()
        duration = len(audio_np) / self.sample_rate

        return GenerationResult(
            audio=audio_np,
            sample_rate=self.sample_rate,
            processing_time_seconds=processing_time,
            real_time_factor=processing_time / duration if duration > 0 else 0,
            duration_seconds=duration
        )

    def generate_stream(
        self,
        text: str,
        chunk_size: int = 250,
        **kwargs
    ) -> Generator[GenerationResult, None, None]:
        """
        Generate speech in chunks for streaming.

        Args:
            text: Text to synthesize
            chunk_size: Characters per chunk
            **kwargs: Passed to generate()

        Yields:
            GenerationResult for each chunk
        """
        import re

        # Split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                yield self.generate(current_chunk.strip(), **kwargs)
                current_chunk = sentence
            else:
                current_chunk += " " + sentence

        if current_chunk.strip():
            yield self.generate(current_chunk.strip(), **kwargs)

    def save(
        self,
        result: Union[GenerationResult, np.ndarray, torch.Tensor],
        path: str
    ):
        """Save audio to file."""
        if isinstance(result, GenerationResult):
            audio = result.audio
            sr = result.sample_rate
        else:
            audio = result
            sr = self.sample_rate

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        ta.save(path, audio, sr)
        if self.verbose:
            print(f"[ChatterboxTTS] Saved to {path}")

    def play(self, result: Union[GenerationResult, np.ndarray]):
        """Play audio through speakers."""
        import sounddevice as sd

        if isinstance(result, GenerationResult):
            audio = result.audio
            sr = result.sample_rate
        else:
            audio = result
            sr = self.sample_rate

        sd.play(audio, sr)
        sd.wait()


# Convenience function
def create_tts(model: str = 'turbo', **kwargs) -> ChatterboxTTS:
    """Create a ChatterboxTTS instance."""
    return ChatterboxTTS(model=model, **kwargs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Chatterbox TTS CLI')
    parser.add_argument('text', help='Text to synthesize')
    parser.add_argument('-m', '--model', default='turbo', choices=['standard', 'turbo'])
    parser.add_argument('-o', '--output', default='output.wav', help='Output file')
    parser.add_argument('-r', '--ref-audio', help='Reference audio for voice cloning')
    parser.add_argument('--play', action='store_true', help='Play audio after generation')

    args = parser.parse_args()

    tts = ChatterboxTTS(model=args.model)
    result = tts.generate(args.text, ref_audio=args.ref_audio)

    print(f"Generated {result.duration_seconds:.1f}s audio in {result.processing_time_seconds:.1f}s")
    print(f"Real-time factor: {result.real_time_factor:.2f}x")

    tts.save(result, args.output)

    if args.play:
        tts.play(result)
