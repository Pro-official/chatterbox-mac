"""
MLX VoiceEncoder - Converted from PyTorch
Original: chatterbox/models/voice_encoder/voice_encoder.py
"""

from dataclasses import dataclass
from typing import List, Optional, Union
from functools import lru_cache

import numpy as np
from numpy.lib.stride_tricks import as_strided
import librosa
from scipy import signal

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class VoiceEncConfig:
    """Voice encoder configuration (frozen for hashability)."""
    num_mels: int = 40
    sample_rate: int = 16000
    speaker_embed_size: int = 256
    ve_hidden_size: int = 256
    n_fft: int = 400
    hop_size: int = 160
    win_size: int = 400
    fmax: int = 8000
    fmin: int = 0
    preemphasis: float = 0.0
    mel_power: float = 2.0
    mel_type: str = "amp"
    normalized_mels: bool = False
    ve_partial_frames: int = 160
    ve_final_relu: bool = True
    stft_magnitude_min: float = 1e-4


# ============ Mel Spectrogram (stays in numpy/librosa) ============

@lru_cache()
def mel_basis(hp: VoiceEncConfig):
    """Get mel filterbank."""
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(
        sr=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax
    )


def preemphasis(wav: np.ndarray, hp: VoiceEncConfig) -> np.ndarray:
    """Apply pre-emphasis filter."""
    assert hp.preemphasis != 0
    wav = signal.lfilter([1, -hp.preemphasis], [1], wav)
    wav = np.clip(wav, -1, 1)
    return wav


def melspectrogram(wav: np.ndarray, hp: VoiceEncConfig, pad: bool = True) -> np.ndarray:
    """Extract mel spectrogram from waveform."""
    if hp.preemphasis > 0:
        wav = preemphasis(wav, hp)
        assert np.abs(wav).max() - 1 < 1e-07

    # STFT
    spec_complex = librosa.stft(
        wav,
        n_fft=hp.n_fft,
        hop_length=hp.hop_size,
        win_length=hp.win_size,
        center=pad,
        pad_mode="reflect",
    )

    # Magnitudes
    spec_magnitudes = np.abs(spec_complex)
    if hp.mel_power != 1.0:
        spec_magnitudes **= hp.mel_power

    # Mel filterbank
    mel = np.dot(mel_basis(hp), spec_magnitudes)

    if hp.mel_type == "db":
        mel = 20 * np.log10(np.maximum(hp.stft_magnitude_min, mel))

    if hp.normalized_mels:
        min_level_db = 20 * np.log10(hp.stft_magnitude_min)
        mel = (mel - min_level_db) / (-min_level_db + 15)
        mel = mel.astype(np.float32)

    return mel  # (M, T)


# ============ Utility Functions ============

def get_num_wins(n_frames: int, step: int, min_coverage: float, hp: VoiceEncConfig):
    """Calculate number of windows for partial utterances."""
    assert n_frames > 0
    win_size = hp.ve_partial_frames
    n_wins, remainder = divmod(max(n_frames - win_size + step, 0), step)
    if n_wins == 0 or (remainder + (win_size - step)) / win_size >= min_coverage:
        n_wins += 1
    target_n = win_size + step * (n_wins - 1)
    return n_wins, target_n


def get_frame_step(overlap: float, rate: float, hp: VoiceEncConfig) -> int:
    """Calculate frame step for overlapping windows."""
    assert 0 <= overlap < 1
    if rate is None:
        frame_step = int(np.round(hp.ve_partial_frames * (1 - overlap)))
    else:
        frame_step = int(np.round((hp.sample_rate / rate) / hp.ve_partial_frames))
    assert 0 < frame_step <= hp.ve_partial_frames
    return frame_step


def stride_as_partials(
    mel: np.ndarray,
    hp: VoiceEncConfig,
    overlap: float = 0.5,
    rate: float = None,
    min_coverage: float = 0.8,
) -> np.ndarray:
    """Convert mel spectrogram to overlapping partial windows."""
    assert 0 < min_coverage <= 1
    frame_step = get_frame_step(overlap, rate, hp)

    n_partials, target_len = get_num_wins(len(mel), frame_step, min_coverage, hp)

    # Pad or trim
    if target_len > len(mel):
        mel = np.concatenate((mel, np.full((target_len - len(mel), hp.num_mels), 0)))
    elif target_len < len(mel):
        mel = mel[:target_len]

    mel = mel.astype(np.float32, order="C")

    # Stride trick for overlapping windows
    shape = (n_partials, hp.ve_partial_frames, hp.num_mels)
    strides = (mel.strides[0] * frame_step, mel.strides[0], mel.strides[1])
    partials = as_strided(mel, shape, strides)
    return partials


def pack_arrays(arrays: List[np.ndarray], seq_len: int = None, pad_value: float = 0) -> np.ndarray:
    """Pack list of arrays into a single padded array."""
    if seq_len is None:
        seq_len = max(len(arr) for arr in arrays)

    packed_shape = (len(arrays), seq_len, *arrays[0].shape[1:])
    packed = np.full(packed_shape, pad_value, dtype=np.float32)

    for i, arr in enumerate(arrays):
        packed[i, :len(arr)] = arr

    return packed


# ============ MLX VoiceEncoder ============

class VoiceEncoder(nn.Module):
    """
    MLX Voice Encoder for speaker embedding extraction.

    Converts audio to speaker embeddings via:
    1. Mel spectrogram extraction
    2. 3-layer LSTM processing
    3. Linear projection
    4. L2 normalization
    """

    def __init__(self, hp: VoiceEncConfig = None):
        super().__init__()
        self.hp = hp or VoiceEncConfig()

        # 3-layer LSTM (stacked manually since MLX LSTM doesn't have num_layers)
        self.lstm_layers = [
            nn.LSTM(input_size=self.hp.num_mels, hidden_size=self.hp.ve_hidden_size, bias=True),
            nn.LSTM(input_size=self.hp.ve_hidden_size, hidden_size=self.hp.ve_hidden_size, bias=True),
            nn.LSTM(input_size=self.hp.ve_hidden_size, hidden_size=self.hp.ve_hidden_size, bias=True),
        ]

        # Projection to embedding space
        self.proj = nn.Linear(self.hp.ve_hidden_size, self.hp.speaker_embed_size)

        # Similarity parameters (for training, not used in inference)
        self.similarity_weight = mx.array([10.0])
        self.similarity_bias = mx.array([-5.0])

    def __call__(self, mels: mx.array) -> mx.array:
        """
        Compute embeddings for a batch of partial utterances.

        Args:
            mels: (B, T, M) mel spectrograms where T=ve_partial_frames

        Returns:
            (B, E) L2-normalized speaker embeddings
        """
        # Pass through stacked LSTM layers
        x = mels
        for lstm in self.lstm_layers:
            x, _ = lstm(x)

        # Get final hidden state from last timestep of output
        # MLX LSTM output shape: (B, T, H) - take last timestep
        last_hidden = x[:, -1, :]

        # Project to embedding space
        raw_embeds = self.proj(last_hidden)

        # Optional ReLU
        if self.hp.ve_final_relu:
            raw_embeds = nn.relu(raw_embeds)

        # L2 normalize
        norm = mx.linalg.norm(raw_embeds, axis=1, keepdims=True)
        embeds = raw_embeds / (norm + 1e-8)

        return embeds

    def embed_utterance(
        self,
        mel: np.ndarray,
        overlap: float = 0.5,
        rate: float = None,
        min_coverage: float = 0.8,
    ) -> np.ndarray:
        """
        Compute speaker embedding for a single utterance.

        Args:
            mel: (T, M) mel spectrogram
            overlap: overlap ratio for partial windows
            rate: partials per second (overrides overlap)
            min_coverage: minimum coverage for last partial

        Returns:
            (E,) speaker embedding
        """
        # Split into overlapping partials
        partials = stride_as_partials(mel, self.hp, overlap, rate, min_coverage)

        # Convert to MLX and process
        partials_mx = mx.array(partials)
        partial_embeds = self(partials_mx)
        mx.eval(partial_embeds)

        # Average partial embeddings
        raw_embed = mx.mean(partial_embeds, axis=0)

        # L2 normalize
        norm = mx.linalg.norm(raw_embed)
        embed = raw_embed / (norm + 1e-8)

        return np.array(embed)

    def embed_wav(
        self,
        wav: np.ndarray,
        sample_rate: int = 16000,
        trim_top_db: float = 20,
        rate: float = 1.3,
    ) -> np.ndarray:
        """
        Compute speaker embedding from waveform.

        Args:
            wav: audio waveform
            sample_rate: sample rate of input audio
            trim_top_db: dB threshold for silence trimming
            rate: partials per second

        Returns:
            (E,) speaker embedding
        """
        # Resample if needed
        if sample_rate != self.hp.sample_rate:
            wav = librosa.resample(
                wav,
                orig_sr=sample_rate,
                target_sr=self.hp.sample_rate,
                res_type="kaiser_fast"
            )

        # Trim silence
        if trim_top_db:
            wav, _ = librosa.effects.trim(wav, top_db=trim_top_db)

        # Extract mel spectrogram
        mel = melspectrogram(wav, self.hp).T  # (T, M)

        return self.embed_utterance(mel, rate=rate)


# ============ Weight Conversion ============

def convert_pytorch_weights(pt_state_dict: dict) -> dict:
    """
    Convert PyTorch VoiceEncoder weights to MLX format.

    PyTorch LSTM weights:
        lstm.weight_ih_l{layer}: (4*hidden, input)
        lstm.weight_hh_l{layer}: (4*hidden, hidden)
        lstm.bias_ih_l{layer}: (4*hidden,)
        lstm.bias_hh_l{layer}: (4*hidden,)

    MLX stacked LSTM weights:
        lstm_layers.{layer}.Wx: (4*hidden, input)
        lstm_layers.{layer}.Wh: (4*hidden, hidden)
        lstm_layers.{layer}.bias: (4*hidden,)
    """
    mlx_state = {}
    temp_biases = {}

    for key, value in pt_state_dict.items():
        np_value = value.cpu().numpy() if hasattr(value, 'cpu') else np.array(value)

        if key.startswith('lstm.'):
            # Parse layer number and weight type
            # e.g., "lstm.weight_ih_l0" -> layer 0, weight_ih
            parts = key.replace('lstm.', '').split('_')
            layer_num = int(parts[-1].replace('l', ''))

            if 'weight_ih' in key:
                mlx_state[f'lstm_layers.{layer_num}.Wx'] = mx.array(np_value)
            elif 'weight_hh' in key:
                mlx_state[f'lstm_layers.{layer_num}.Wh'] = mx.array(np_value)
            elif 'bias_ih' in key:
                temp_biases[f'ih_{layer_num}'] = np_value
            elif 'bias_hh' in key:
                temp_biases[f'hh_{layer_num}'] = np_value

        elif key.startswith('proj.'):
            mlx_state[key] = mx.array(np_value)

        elif key in ['similarity_weight', 'similarity_bias']:
            mlx_state[key] = mx.array(np_value)

    # Combine ih and hh biases for each layer
    for layer_num in range(3):
        ih_key = f'ih_{layer_num}'
        hh_key = f'hh_{layer_num}'
        if ih_key in temp_biases and hh_key in temp_biases:
            combined_bias = temp_biases[ih_key] + temp_biases[hh_key]
            mlx_state[f'lstm_layers.{layer_num}.bias'] = mx.array(combined_bias)

    return mlx_state


def load_from_pytorch(pt_weights_path: str, hp: VoiceEncConfig = None) -> VoiceEncoder:
    """
    Load VoiceEncoder from PyTorch weights.

    Args:
        pt_weights_path: path to .safetensors or .pt file
        hp: voice encoder config

    Returns:
        MLX VoiceEncoder with loaded weights
    """
    from safetensors.torch import load_file
    import torch

    # Load PyTorch weights
    if pt_weights_path.endswith('.safetensors'):
        pt_state = load_file(pt_weights_path)
    else:
        pt_state = torch.load(pt_weights_path, map_location='cpu')

    # Convert to MLX
    mlx_state = convert_pytorch_weights(pt_state)

    # Create model and load weights
    model = VoiceEncoder(hp)
    model.load_weights(list(mlx_state.items()))

    return model


# ============ Test Function ============

def test_voice_encoder():
    """Test the MLX VoiceEncoder."""
    print("Testing MLX VoiceEncoder...")

    hp = VoiceEncConfig()
    model = VoiceEncoder(hp)

    # Create dummy input
    batch_size = 2
    seq_len = hp.ve_partial_frames
    mels = mx.random.normal((batch_size, seq_len, hp.num_mels))

    # Forward pass
    embeds = model(mels)
    mx.eval(embeds)

    print(f"Input shape: {mels.shape}")
    print(f"Output shape: {embeds.shape}")
    print(f"Output norm (should be ~1.0): {mx.linalg.norm(embeds[0]):.4f}")

    # Test from wav
    print("\nTesting embed_wav...")
    dummy_wav = np.random.randn(hp.sample_rate * 3).astype(np.float32) * 0.1  # 3 seconds
    embed = model.embed_wav(dummy_wav, sample_rate=hp.sample_rate)
    print(f"Wav embedding shape: {embed.shape}")
    print(f"Wav embedding norm: {np.linalg.norm(embed):.4f}")

    print("\n✅ VoiceEncoder test passed!")


if __name__ == "__main__":
    test_voice_encoder()
