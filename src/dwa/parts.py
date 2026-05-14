"""PartA and PartB — causal Transformer halves of the DWA model."""

import jax
import jax.numpy as jnp
from flax import nnx

from .config import DWAConfig


class CausalSelfAttention(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, rngs: nnx.Rngs) -> None:
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        scale = d_model ** -0.5
        self.Wqkv = nnx.Linear(d_model, 3 * d_model, use_bias=False, rngs=rngs)
        self.Wo = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.scale = scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, T, C = x.shape
        H, Hd = self.n_heads, self.head_dim

        qkv = self.Wqkv(x)                           # [B, T, 3C]
        q, k, v = jnp.split(qkv, 3, axis=-1)         # each [B, T, C]

        # reshape to multi-head
        q = q.reshape(B, T, H, Hd).transpose(0, 2, 1, 3)  # [B, H, T, Hd]
        k = k.reshape(B, T, H, Hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, Hd).transpose(0, 2, 1, 3)

        # scaled dot-product attention with causal mask
        attn = jnp.einsum("bhid,bhjd->bhij", q, k) * self.scale  # [B, H, T, T]
        mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        attn = jnp.where(mask[None, None], attn, jnp.finfo(attn.dtype).min)
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.einsum("bhij,bhjd->bhid", attn, v)  # [B, H, T, Hd]
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.Wo(out)


class FFN(nnx.Module):
    def __init__(self, d_model: int, ffn_mult: int, rngs: nnx.Rngs) -> None:
        d_ff = d_model * ffn_mult
        self.W1 = nnx.Linear(d_model, d_ff, use_bias=False, rngs=rngs)
        self.W2 = nnx.Linear(d_ff, d_model, use_bias=False, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.W2(jax.nn.gelu(self.W1(x)))


class TransformerBlock(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, rngs: nnx.Rngs) -> None:
        self.norm1 = nnx.RMSNorm(d_model, rngs=rngs)
        self.attn = CausalSelfAttention(d_model, n_heads, rngs)
        self.norm2 = nnx.RMSNorm(d_model, rngs=rngs)
        self.ffn = FFN(d_model, ffn_mult, rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PartA(nnx.Module):
    """
    Transformer encoder (Part A).
    Input:  token embeddings [B, seq, d_A]
    Output: hidden state h_A [B, seq, d_A]
    """

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs) -> None:
        self.blocks = nnx.List([
            TransformerBlock(cfg.d_A, cfg.n_heads, cfg.ffn_mult, rngs)
            for _ in range(cfg.n_layers_A)
        ])
        self.norm = nnx.RMSNorm(cfg.d_A, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)   # [B, seq, d_A]


class PartB(nnx.Module):
    """
    Transformer decoder (Part B).
    Input:  h_mid [B, seq, d_B]
    Output: hidden states [B, seq, d_B]
    """

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs) -> None:
        self.blocks = nnx.List([
            TransformerBlock(cfg.d_B, cfg.n_heads, cfg.ffn_mult, rngs)
            for _ in range(cfg.n_layers_B)
        ])
        self.norm = nnx.RMSNorm(cfg.d_B, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)   # [B, seq, d_B]
