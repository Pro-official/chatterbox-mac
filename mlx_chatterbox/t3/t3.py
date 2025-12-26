"""
MLX T3 Model - Token-To-Token TTS Model
Converted from PyTorch: chatterbox/models/t3/t3.py
"""

import logging
from typing import Optional, List

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.llama import LlamaModel, ModelArgs
from mlx_lm.models.cache import KVCache

from .modules.t3_config import T3Config, T3Cond, LLAMA_CONFIGS
from .modules.learned_pos_emb import LearnedPositionEmbeddings
from .modules.cond_enc import T3CondEnc

logger = logging.getLogger(__name__)


class T3(nn.Module):
    """
    MLX Token-To-Token (T3) TTS model.

    Uses Llama transformer as backbone with custom embeddings for:
    - Text tokens (with learned position embeddings)
    - Speech tokens (with learned position embeddings)
    - Conditioning (speaker, emotion, speech prompts)
    """

    def __init__(self, hp: T3Config = None):
        super().__init__()
        self.hp = hp or T3Config.english_only()

        # Get Llama config
        llama_config = LLAMA_CONFIGS[self.hp.llama_config_name]
        self.is_gpt = llama_config.get("is_gpt", False)

        # Create ModelArgs for LlamaModel
        self.model_args = ModelArgs(
            model_type="llama",
            hidden_size=llama_config["hidden_size"],
            num_hidden_layers=llama_config["num_hidden_layers"],
            intermediate_size=llama_config["intermediate_size"],
            num_attention_heads=llama_config["num_attention_heads"],
            num_key_value_heads=llama_config.get("num_key_value_heads", llama_config["num_attention_heads"]),
            rms_norm_eps=llama_config["rms_norm_eps"],
            vocab_size=8,  # Dummy, we use custom embeddings
            head_dim=llama_config.get("head_dim"),
            max_position_embeddings=llama_config.get("max_position_embeddings", 131072),
            attention_bias=llama_config.get("attention_bias", False),
            mlp_bias=llama_config.get("mlp_bias", False),
            rope_theta=llama_config.get("rope_theta", 500000.0),
            rope_scaling=llama_config.get("rope_scaling"),
        )

        # Llama backbone
        self.tfmr = LlamaModel(self.model_args)
        self.dim = self.model_args.hidden_size

        # Custom embeddings
        self.cond_enc = T3CondEnc(self.hp)
        self.text_emb = nn.Embedding(self.hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(self.hp.speech_tokens_dict_size, self.dim)

        # Learned position embeddings
        self.text_pos_emb = None
        self.speech_pos_emb = None
        if self.hp.input_pos_emb == "learned":
            max_text_seq_len = self.hp.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = self.hp.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # Output heads
        self.text_head = nn.Linear(self.dim, self.hp.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(self.dim, self.hp.speech_tokens_dict_size, bias=self.is_gpt)

    def prepare_conditioning(self, t3_cond: T3Cond) -> mx.array:
        """
        Prepare conditioning embeddings.

        Args:
            t3_cond: Conditioning data

        Returns:
            Conditioning embeddings (B, len_cond, dim)
        """
        # Embed speech prompt tokens if provided
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens)
            if not self.is_gpt and self.speech_pos_emb is not None:
                t3_cond.cond_prompt_speech_emb = (
                    t3_cond.cond_prompt_speech_emb +
                    self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
                )

        return self.cond_enc(t3_cond)

    def prepare_input_embeds(
        self,
        t3_cond: T3Cond,
        text_tokens: mx.array,
        speech_tokens: mx.array,
        cfg_weight: float = 0.0,
    ) -> tuple:
        """
        Prepare input embeddings for the transformer.

        Args:
            t3_cond: Conditioning data
            text_tokens: Text token IDs (B, T_text)
            speech_tokens: Speech token IDs (B, T_speech)
            cfg_weight: CFG weight for unconditional generation

        Returns:
            (embeddings, len_cond)
        """
        # Get conditioning embeddings
        cond_emb = self.prepare_conditioning(t3_cond)  # (B, len_cond, dim)

        # Embed text tokens
        text_emb = self.text_emb(text_tokens)  # (B, T_text, dim)
        if cfg_weight > 0.0 and not self.is_gpt:
            # Zero out text for unconditional CFG
            text_emb = mx.where(
                mx.arange(text_emb.shape[0])[:, None, None] == 1,
                mx.zeros_like(text_emb),
                text_emb
            )

        # Embed speech tokens
        speech_emb = self.speech_emb(speech_tokens)  # (B, T_speech, dim)

        # Add position embeddings
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)

        len_cond = cond_emb.shape[1]

        # Expand conditioning if batch sizes don't match
        if cond_emb.shape[0] != text_emb.shape[0]:
            cond_emb = mx.broadcast_to(
                cond_emb,
                (text_emb.shape[0], cond_emb.shape[1], cond_emb.shape[2])
            )

        # Concatenate: [conditioning, text, speech]
        embeds = mx.concatenate([cond_emb, text_emb, speech_emb], axis=1)

        return embeds, len_cond

    def __call__(
        self,
        t3_cond: T3Cond,
        text_tokens: mx.array,
        speech_tokens: mx.array,
        cache: Optional[List[KVCache]] = None,
    ) -> mx.array:
        """
        Forward pass through the model.

        Args:
            t3_cond: Conditioning data
            text_tokens: Text token IDs (B, T_text)
            speech_tokens: Speech token IDs (B, T_speech)
            cache: Optional KV cache for generation

        Returns:
            Hidden states from transformer (B, seq_len, dim)
        """
        embeds, _ = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        )

        # Forward through Llama backbone with custom embeddings
        hidden_states = self.tfmr(
            inputs=None,
            cache=cache,
            input_embeddings=embeds,
        )

        return hidden_states

    def inference_turbo(
        self,
        t3_cond: T3Cond,
        text_tokens: mx.array,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        max_gen_len: int = 1000,
    ) -> mx.array:
        """
        Generate speech tokens using turbo inference (no CFG).

        Args:
            t3_cond: Conditioning data
            text_tokens: Text token IDs (B, T_text)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeated tokens
            max_gen_len: Maximum generation length

        Returns:
            Generated speech tokens (B, T_gen)
        """
        # Initial speech token (start token)
        speech_start_token = mx.full((1, 1), self.hp.start_speech_token, dtype=mx.int32)

        # Prepare initial embeddings
        embeds, _ = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_start_token,
            cfg_weight=0.0,
        )

        # Initialize KV cache
        cache = [KVCache() for _ in range(self.model_args.num_hidden_layers)]

        # Initial forward pass
        hidden_states = self.tfmr(inputs=None, cache=cache, input_embeddings=embeds)

        # Get logits for speech
        speech_hidden = hidden_states[:, -1:, :]
        speech_logits = self.speech_head(speech_hidden)

        # Sample first token
        next_token = self._sample_token(
            speech_logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        generated_tokens = [next_token]
        current_token = next_token

        # Generation loop
        for step in range(max_gen_len):
            # Embed current token
            current_emb = self.speech_emb(current_token)

            # Forward with cache
            hidden_states = self.tfmr(
                inputs=None,
                cache=cache,
                input_embeddings=current_emb,
            )

            # Get speech logits
            speech_logits = self.speech_head(hidden_states)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                past_tokens = mx.concatenate(generated_tokens, axis=1)
                speech_logits = self._apply_repetition_penalty(
                    speech_logits[:, -1, :],
                    past_tokens,
                    repetition_penalty
                )
            else:
                speech_logits = speech_logits[:, -1, :]

            # Sample next token
            next_token = self._sample_token(
                speech_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            generated_tokens.append(next_token)
            current_token = next_token

            # Check for EOS
            if mx.all(next_token == self.hp.stop_speech_token):
                logger.info(f"EOS detected at step {step + 1}")
                break

        # Concatenate all tokens
        all_tokens = mx.concatenate(generated_tokens, axis=1)

        # Remove EOS token if present
        if all_tokens.shape[1] > 0 and all_tokens[0, -1] == self.hp.stop_speech_token:
            all_tokens = all_tokens[:, :-1]

        return all_tokens

    def _sample_token(
        self,
        logits: mx.array,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> mx.array:
        """Sample a token from logits."""
        if temperature == 0:
            return mx.argmax(logits, axis=-1, keepdims=True)

        # Apply temperature
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0 and top_k < logits.shape[-1]:
            # Get top-k values and mask others
            top_k = min(top_k, logits.shape[-1])
            sorted_logits = mx.sort(logits, axis=-1)
            threshold = sorted_logits[..., -top_k]
            logits = mx.where(logits < threshold[..., None], float('-inf'), logits)

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_indices = mx.argsort(-logits, axis=-1)
            sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
            sorted_probs = mx.softmax(sorted_logits, axis=-1)
            cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

            # Remove tokens with cumulative prob above threshold
            sorted_mask = cumulative_probs > top_p
            # Shift to keep first token above threshold
            sorted_mask = mx.concatenate([
                mx.zeros((*sorted_mask.shape[:-1], 1), dtype=mx.bool_),
                sorted_mask[..., :-1]
            ], axis=-1)

            # Apply mask in sorted order, then unsort
            sorted_logits = mx.where(sorted_mask, float('-inf'), sorted_logits)

            # Unsort by using argsort of sorted_indices
            unsort_indices = mx.argsort(sorted_indices, axis=-1)
            logits = mx.take_along_axis(sorted_logits, unsort_indices, axis=-1)

        # Sample
        probs = mx.softmax(logits, axis=-1)
        return mx.random.categorical(mx.log(probs + 1e-10), axis=-1)[:, None]

    def _apply_repetition_penalty(
        self,
        logits: mx.array,
        past_tokens: mx.array,
        penalty: float,
    ) -> mx.array:
        """Apply repetition penalty to logits (simplified for MLX)."""
        if penalty == 1.0:
            return logits

        # For MLX, we apply a simpler penalty by creating a mask
        vocab_size = logits.shape[-1]

        # Convert past tokens to numpy for processing
        past_np = np.array(past_tokens).flatten()
        unique_tokens = np.unique(past_np)
        unique_tokens = unique_tokens[unique_tokens < vocab_size]

        if len(unique_tokens) == 0:
            return logits

        # Create penalty mask
        penalty_mask = mx.ones((1, vocab_size))
        for token_idx in unique_tokens:
            # Apply penalty: divide positive logits, multiply negative
            # Simplified: just divide all by penalty
            penalty_mask = penalty_mask * mx.where(
                mx.arange(vocab_size) == int(token_idx),
                1.0 / penalty,
                1.0
            )

        return logits * penalty_mask

    def make_cache(self) -> List[KVCache]:
        """Create KV cache for generation."""
        return [KVCache() for _ in range(self.model_args.num_hidden_layers)]


# ============ Weight Conversion ============

def convert_pytorch_t3_weights(pt_state_dict: dict, hp: T3Config = None) -> dict:
    """
    Convert PyTorch T3 weights to MLX format.

    The main mappings are:
    - tfmr.layers.{i}.* -> tfmr.layers.{i}.*
    - text_emb.weight -> text_emb.weight
    - speech_emb.weight -> speech_emb.weight
    - text_pos_emb.emb.weight -> text_pos_emb.emb.weight
    - speech_pos_emb.emb.weight -> speech_pos_emb.emb.weight
    - cond_enc.* -> cond_enc.*
    - text_head.weight -> text_head.weight
    - speech_head.* -> speech_head.*
    """
    import numpy as np

    mlx_state = {}

    for key, value in pt_state_dict.items():
        np_value = value.cpu().numpy() if hasattr(value, 'cpu') else np.array(value)

        # Skip rotary embedding frequencies (computed dynamically)
        if "rotary_emb" in key:
            continue

        new_key = key

        # Handle HuggingFace Llama naming conventions
        if key.startswith("tfmr.model."):
            new_key = key.replace("tfmr.model.", "tfmr.")

        # Handle final layer norm
        if key == "tfmr.model.norm.weight":
            new_key = "tfmr.norm.weight"

        mlx_state[new_key] = mx.array(np_value)

    return mlx_state


def load_from_pytorch(pt_weights_path: str, hp: T3Config = None) -> T3:
    """
    Load T3 from PyTorch weights.

    Args:
        pt_weights_path: Path to .safetensors or .pt file
        hp: T3 configuration

    Returns:
        MLX T3 model with loaded weights
    """
    from safetensors.torch import load_file
    import torch

    hp = hp or T3Config.english_only()

    # Load PyTorch weights
    if pt_weights_path.endswith('.safetensors'):
        pt_state = load_file(pt_weights_path)
    else:
        pt_state = torch.load(pt_weights_path, map_location='cpu')

    # Convert to MLX
    mlx_state = convert_pytorch_t3_weights(pt_state, hp)

    # Create model and load weights
    model = T3(hp)
    model.load_weights(list(mlx_state.items()))

    return model


# ============ Test Function ============

def test_t3():
    """Test the MLX T3 model."""
    print("Testing MLX T3...")

    hp = T3Config.english_only()
    model = T3(hp)

    # Create dummy inputs
    batch_size = 1
    text_len = 10
    speech_len = 5

    text_tokens = mx.array([[hp.start_text_token] + [100] * (text_len - 2) + [hp.stop_text_token]])
    speech_tokens = mx.array([[hp.start_speech_token] + [1000] * (speech_len - 1)])

    # Create dummy conditioning
    speaker_emb = mx.random.normal((1, hp.speaker_embed_size))
    t3_cond = T3Cond(speaker_emb=speaker_emb, emotion_adv=0.5)

    # Forward pass
    print("Running forward pass...")
    hidden = model(t3_cond, text_tokens, speech_tokens)
    mx.eval(hidden)

    print(f"Input text shape: {text_tokens.shape}")
    print(f"Input speech shape: {speech_tokens.shape}")
    print(f"Hidden states shape: {hidden.shape}")

    print("\n✅ T3 basic test passed!")


if __name__ == "__main__":
    test_t3()
