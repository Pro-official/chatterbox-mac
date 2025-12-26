"""
MLX Chatterbox TTS

A hybrid MLX-PyTorch implementation of Chatterbox TTS that uses:
- MLX for VoiceEncoder and T3 (text-to-tokens) - fastest components
- PyTorch/MPS for S3Gen and HiFiGAN - efficient on Apple Silicon

Usage:
    from mlx_chatterbox import ChatterboxMLX

    model = ChatterboxMLX.from_pretrained(turbo=True)
    audio = model.generate(
        text="Hello world!",
        audio_prompt="reference.wav",
    )
"""

from .chatterbox import ChatterboxMLX, GenerationConfig
from .voice_encoder import VoiceEncoder, VoiceEncConfig
from .t3 import T3, T3Config, T3Cond

__all__ = [
    "ChatterboxMLX",
    "GenerationConfig",
    "VoiceEncoder",
    "VoiceEncConfig",
    "T3",
    "T3Config",
    "T3Cond",
]

__version__ = "0.1.0"
