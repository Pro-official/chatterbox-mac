"""
MLX Perceiver Resampler - Converted from PyTorch
Original: chatterbox/models/t3/modules/perceiver.py
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class RelativePositionBias(nn.Module):
    """Relative position bias for attention."""

    def __init__(
        self,
        scale: float,
        causal: bool = False,
        num_buckets: int = 32,
        max_distance: int = 128,
        heads: int = 8
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    def _relative_position_bucket(
        self,
        relative_position: mx.array,
        causal: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128
    ) -> mx.array:
        """Compute relative position buckets."""
        ret = mx.zeros_like(relative_position)
        n = -relative_position

        if not causal:
            num_buckets //= 2
            ret = ret + (n < 0).astype(mx.int32) * num_buckets
            n = mx.abs(n)
        else:
            n = mx.maximum(n, mx.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        # Logarithmic bucketing for larger distances
        val_if_large = max_exact + (
            mx.log(n.astype(mx.float32) / max_exact) /
            math.log(max_distance / max_exact) *
            (num_buckets - max_exact)
        ).astype(mx.int32)
        val_if_large = mx.minimum(val_if_large, mx.full(val_if_large.shape, num_buckets - 1))

        ret = ret + mx.where(is_small, n, val_if_large)
        return ret.astype(mx.int32)

    def __call__(self, qk_dots: mx.array) -> mx.array:
        """Apply relative position bias to attention scores."""
        i, j = qk_dots.shape[-2:]

        q_pos = mx.arange(i)
        k_pos = mx.arange(j)
        rel_pos = k_pos[None, :] - q_pos[:, None]

        rp_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )

        values = self.relative_attention_bias(rp_bucket)  # (i, j, heads)
        bias = mx.transpose(values, (2, 0, 1))[None, :, :, :]  # (1, heads, i, j)

        return qk_dots + (bias * self.scale)


class AttentionQKV(nn.Module):
    """Multi-head attention with separate Q, K, V inputs."""

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        dropout_rate: float = 0.1,
        scale: Optional[float] = None
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = scale if scale is not None else head_dim ** -0.5

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Apply multi-head attention.

        Args:
            q: Query tensor (B, T_q, H*D)
            k: Key tensor (B, T_k, H*D)
            v: Value tensor (B, T_k, H*D)
            mask: Optional attention mask

        Returns:
            Output tensor (B, T_q, H*D)
        """
        q, k, v = [self._split_heads(x) for x in [q, k, v]]

        # Scaled dot-product attention
        # q: (B, H, T_q, D), k: (B, H, T_k, D)
        sim = mx.einsum("bhqd,bhkd->bhqk", q, k) * self.scale

        if mask is not None:
            sim = mx.where(mask == 0, mx.full(sim.shape, float('-inf')), sim)

        attn = mx.softmax(sim, axis=-1)

        # Apply attention to values
        out = mx.einsum("bhqk,bhkd->bhqd", attn, v)

        return self._combine_heads(out)

    def _split_heads(self, x: mx.array) -> mx.array:
        """Split heads: (B, T, H*D) -> (B, H, T, D)."""
        B, T, _ = x.shape
        x = x.reshape(B, T, self.n_heads, self.head_dim)
        return mx.transpose(x, (0, 2, 1, 3))

    def _combine_heads(self, x: mx.array) -> mx.array:
        """Combine heads: (B, H, T, D) -> (B, T, H*D)."""
        B, H, T, D = x.shape
        x = mx.transpose(x, (0, 2, 1, 3))
        return x.reshape(B, T, H * D)


class AttentionBlock(nn.Module):
    """Attention block with layer norm and projections."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        relative_pos_embeddings: bool = False,
        dropout_rate: float = 0.2,
        scale: Optional[float] = None
    ):
        super().__init__()
        self.channels = channels

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels

        self.norm = nn.LayerNorm(channels)

        # Separate linear layers for Q, K, V (with bias to match PyTorch)
        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_k = nn.Linear(channels, channels, bias=True)
        self.to_v = nn.Linear(channels, channels, bias=True)

        self.attention = AttentionQKV(
            self.num_heads,
            channels // self.num_heads,
            dropout_rate=dropout_rate,
            scale=scale
        )

        self.proj_out = nn.Linear(channels, channels, bias=True)

        self.relative_pos_embeddings = None
        if relative_pos_embeddings:
            self.relative_pos_embeddings = RelativePositionBias(
                scale=(channels // self.num_heads) ** 0.5,
                causal=False,
                heads=num_heads,
                num_buckets=32,
                max_distance=64
            )

    def __call__(
        self,
        x1: mx.array,
        x2: mx.array,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Cross-attention between x1 (query) and x2 (key/value).

        Args:
            x1: Query tensor (B, T1, C)
            x2: Key/Value tensor (B, T2, C)
            mask: Optional attention mask

        Returns:
            Output tensor (B, T1, C)
        """
        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)

        q = self.to_q(x1_norm)
        k = self.to_k(x2_norm)
        v = self.to_v(x2_norm)

        h = self.attention(q, k, v, mask=mask)
        h = self.proj_out(h)

        return x1 + h


class Perceiver(nn.Module):
    """
    Perceiver resampler for conditioning.
    Inspired by https://arxiv.org/abs/2103.03206
    """

    def __init__(
        self,
        pre_attention_query_token: int = 32,
        pre_attention_query_size: int = 1024,
        embedding_dim: int = 1024,
        num_attn_heads: int = 4
    ):
        """
        Initialize the perceiver module.

        Args:
            pre_attention_query_token: Number of query tokens
            pre_attention_query_size: Size of each query token
            embedding_dim: Dimension of the embedding space
            num_attn_heads: Number of attention heads
        """
        super().__init__()

        # Learnable query tokens
        # Initialize with uniform distribution
        query_variance = math.sqrt(3.0) * math.sqrt(
            2.0 / (pre_attention_query_token + pre_attention_query_token)
        )
        self.pre_attention_query = mx.random.uniform(
            low=-query_variance,
            high=query_variance,
            shape=(1, pre_attention_query_token, pre_attention_query_size)
        )

        # Attention block for cross and self attention
        self.attn = AttentionBlock(embedding_dim, num_attn_heads)

    def __call__(self, h: mx.array) -> mx.array:
        """
        Forward pass of the perceiver module.

        Args:
            h: Input tensor (B, T, C)

        Returns:
            Resampled output (B, num_queries, C)
        """
        # Expand query to batch size
        batch_size = h.shape[0]
        query = mx.broadcast_to(
            self.pre_attention_query,
            (batch_size, self.pre_attention_query.shape[1], self.pre_attention_query.shape[2])
        )

        # Cross-attention: query attends to input
        pre_att = self.attn(query, h)

        # Self-attention on the result
        attn_out = self.attn(pre_att, pre_att)

        return attn_out
