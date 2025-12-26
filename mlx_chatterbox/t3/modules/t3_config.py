"""
MLX T3 Configuration - Converted from PyTorch
Original: chatterbox/models/t3/modules/t3_config.py
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


# Llama 520M configuration (matches original)
LLAMA_520M_CONFIG = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_hidden_layers": 30,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "head_dim": 64,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    "max_position_embeddings": 131072,
    "attention_bias": False,
    "mlp_bias": False,
}

# GPT2 Medium configuration (for turbo model)
GPT2_MEDIUM_CONFIG = {
    "hidden_size": 1024,
    "intermediate_size": 4096,  # n_embd * 4
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "head_dim": 64,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_position_embeddings": 8196,
    "attention_bias": True,
    "mlp_bias": True,
    "is_gpt": True,
}

LLAMA_CONFIGS = {
    "Llama_520M": LLAMA_520M_CONFIG,
    "GPT2_medium": GPT2_MEDIUM_CONFIG,
}


@dataclass(frozen=True)
class T3Config:
    """T3 model configuration (frozen for hashability)."""

    # Text tokens
    start_text_token: int = 255
    stop_text_token: int = 0
    text_tokens_dict_size: int = 704
    max_text_tokens: int = 2048

    # Speech tokens
    start_speech_token: int = 6561
    stop_speech_token: int = 6562
    speech_tokens_dict_size: int = 8194
    max_speech_tokens: int = 4096

    # Model architecture
    llama_config_name: str = "Llama_520M"
    input_pos_emb: str = "learned"
    speech_cond_prompt_len: int = 150

    # Conditioning
    encoder_type: str = "voice_encoder"
    speaker_embed_size: int = 256
    use_perceiver_resampler: bool = True
    emotion_adv: bool = True

    @property
    def n_channels(self) -> int:
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]

    @property
    def is_multilingual(self) -> bool:
        return self.text_tokens_dict_size == 2454

    @property
    def llama_config(self) -> Dict[str, Any]:
        return LLAMA_CONFIGS[self.llama_config_name]

    @classmethod
    def english_only(cls) -> "T3Config":
        """Create configuration for English-only TTS model."""
        return cls(text_tokens_dict_size=704)

    @classmethod
    def multilingual(cls) -> "T3Config":
        """Create configuration for multilingual TTS model."""
        return cls(text_tokens_dict_size=2454)


@dataclass
class T3Cond:
    """
    Container for T3 conditioning data.
    Unlike the frozen config, this is mutable for runtime data.
    """
    speaker_emb: Any  # mx.array
    clap_emb: Optional[Any] = None
    cond_prompt_speech_tokens: Optional[Any] = None
    cond_prompt_speech_emb: Optional[Any] = None
    emotion_adv: float = 0.5
