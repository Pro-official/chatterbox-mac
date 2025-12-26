"""
MLX Learned Position Embeddings - Converted from PyTorch
Original: chatterbox/models/t3/modules/learned_pos_emb.py
"""

import mlx.core as mx
import mlx.nn as nn


class LearnedPositionEmbeddings(nn.Module):
    """Learned positional embeddings for T3 model."""

    def __init__(self, seq_len: int, model_dim: int, init_std: float = 0.02):
        """
        Initialize learned position embeddings.

        Args:
            seq_len: Maximum sequence length
            model_dim: Embedding dimension
            init_std: Standard deviation for weight initialization
        """
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Note: MLX doesn't have direct weight initialization like PyTorch
        # Weights will be loaded from converted checkpoint

    def __call__(self, x: mx.array) -> mx.array:
        """
        Return positional embeddings for indices 0 to len(x).

        Args:
            x: Input tensor of shape (B, T, ...) - only uses shape to determine T

        Returns:
            Positional embeddings of shape (T, dim)
        """
        seq_len = x.shape[1]
        positions = mx.arange(seq_len)
        return self.emb(positions)

    def get_fixed_embedding(self, idx: int) -> mx.array:
        """
        Get positional embedding for a specific index.

        Args:
            idx: Position index (scalar)

        Returns:
            Positional embedding of shape (1, 1, dim)
        """
        idx_arr = mx.array([[idx]])
        return self.emb(idx_arr)  # (1, 1, dim)
