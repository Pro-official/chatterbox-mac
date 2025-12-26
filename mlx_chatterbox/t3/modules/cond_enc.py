"""
MLX T3 Conditioning Encoder - Converted from PyTorch
Original: chatterbox/models/t3/modules/cond_enc.py
"""

import mlx.core as mx
import mlx.nn as nn

from .t3_config import T3Config, T3Cond
from .perceiver import Perceiver


class T3CondEnc(nn.Module):
    """
    Handle all non-text conditioning for T3:
    - Speaker embeddings
    - Speech prompts (with optional perceiver resampling)
    - Emotion adversarial conditioning
    """

    def __init__(self, hp: T3Config):
        super().__init__()
        self.hp = hp

        # Speaker embedding projection
        if hp.encoder_type == "voice_encoder":
            self.spkr_enc = nn.Linear(hp.speaker_embed_size, hp.n_channels)
        else:
            raise NotImplementedError(f"Unknown encoder type: {hp.encoder_type}")

        # Emotion adversarial projection
        self.emotion_adv_fc = None
        if hp.emotion_adv:
            self.emotion_adv_fc = nn.Linear(1, hp.n_channels, bias=False)

        # Perceiver resampler for speech prompts
        self.perceiver = None
        if hp.use_perceiver_resampler:
            self.perceiver = Perceiver(
                pre_attention_query_token=32,
                pre_attention_query_size=hp.n_channels,
                embedding_dim=hp.n_channels,
                num_attn_heads=4
            )

    def __call__(self, cond: T3Cond) -> mx.array:
        """
        Encode conditioning data.

        Args:
            cond: T3Cond containing speaker_emb, emotion_adv, etc.

        Returns:
            Conditioning embeddings (B, len_cond, dim)
        """
        # Speaker embedding projection: (B, speaker_dim) -> (B, 1, dim)
        speaker_emb = cond.speaker_emb
        if speaker_emb.ndim == 1:
            speaker_emb = speaker_emb[None, :]  # Add batch dim

        cond_spkr = self.spkr_enc(speaker_emb)[:, None, :]  # (B, 1, dim)

        # Empty tensor for unused conditions
        B = cond_spkr.shape[0]
        dim = cond_spkr.shape[-1]
        empty = mx.zeros((B, 0, dim))

        # CLAP embedding (not implemented yet)
        cond_clap = empty

        # Speech prompt conditioning
        cond_prompt_speech_emb = cond.cond_prompt_speech_emb
        if cond_prompt_speech_emb is None:
            cond_prompt_speech_emb = empty
        elif self.perceiver is not None:
            cond_prompt_speech_emb = self.perceiver(cond_prompt_speech_emb)

        # Emotion adversarial conditioning
        cond_emotion_adv = empty
        if self.emotion_adv_fc is not None and cond.emotion_adv is not None:
            emotion_val = cond.emotion_adv
            if isinstance(emotion_val, (int, float)):
                emotion_val = mx.array([[[emotion_val]]])  # (1, 1, 1)
            elif emotion_val.ndim == 0:
                emotion_val = emotion_val.reshape(1, 1, 1)
            elif emotion_val.ndim == 1:
                emotion_val = emotion_val.reshape(-1, 1, 1)

            cond_emotion_adv = self.emotion_adv_fc(emotion_val)  # (B, 1, dim)

        # Concatenate all conditioning
        cond_embeds = mx.concatenate([
            cond_spkr,
            cond_clap,
            cond_prompt_speech_emb,
            cond_emotion_adv,
        ], axis=1)

        return cond_embeds
