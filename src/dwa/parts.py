"""PartA and PartB — causal Transformer halves of the DWA model."""

import jax
import jax.numpy as jnp
from flax import nnx
try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from .config import DWAConfig

try:
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
    HAS_PALLAS_ATTN = True
except ImportError:
    HAS_PALLAS_ATTN = False


def shard_flash_attention(q, k, v, causal, sm_scale, mesh):
    # q, k, v: [B, H, T, Hd] sharded on B
    return shard_map(
        lambda _q, _k, _v: flash_attention(_q, _k, _v, causal=causal, sm_scale=sm_scale),
        mesh=mesh,
        in_specs=(P("data", None, None, None), P("data", None, None, None), P("data", None, None, None)),
        out_specs=P("data", None, None, None),
        check_rep=False,
    )(q, k, v)


def precompute_rope_freqs(seq_len: int, head_dim: int, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2) / head_dim))
    t = jnp.arange(seq_len)
    freqs = jnp.outer(t, inv_freq)
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray):
    # x: [B, H, T, Hd] or [B, T, H, Hd]
    # We assume x is [B, T, H, Hd] for RoPE application convenience
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    x_rotated = jnp.concatenate((-x2, x1), axis=-1)
    # cos/sin: [T, Hd] -> [1, T, 1, Hd]
    return (x * cos[None, :, None, :]) + (x_rotated * sin[None, :, None, :])


class CausalSelfAttention(nnx.Module):
    def __init__(self, cfg: DWAConfig, d_model: int, n_heads: int, n_kv_heads: int,
                 rngs: nnx.Rngs, compute_dtype=None) -> None:
        self.cfg = cfg
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self._cdtype = compute_dtype
        
        self.Wq = nnx.Linear(d_model, d_model, use_bias=False,
                             dtype=compute_dtype, rngs=rngs)
        self.Wk = nnx.Linear(d_model, n_kv_heads * self.head_dim, use_bias=False,
                             dtype=compute_dtype, rngs=rngs)
        self.Wv = nnx.Linear(d_model, n_kv_heads * self.head_dim, use_bias=False,
                             dtype=compute_dtype, rngs=rngs)
        self.Wo = nnx.Linear(d_model, d_model, use_bias=False,
                              dtype=compute_dtype, rngs=rngs)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: jnp.ndarray, cos=None, sin=None, mesh=None) -> jnp.ndarray:
        B, T, C = x.shape
        H, Hd = self.n_heads, self.head_dim
        Hk = self.n_kv_heads

        q = self.Wq(x).reshape(B, T, H, Hd)
        k = self.Wk(x).reshape(B, T, Hk, Hd)
        v = self.Wv(x).reshape(B, T, Hk, Hd)

        # Apply RoPE
        if self.cfg.use_rope and cos is not None and sin is not None:
            q = apply_rope(q, cos[:T], sin[:T])
            k = apply_rope(k, cos[:T], sin[:T])

        # Cast to compute dtype
        if self._cdtype is not None:
            q, k, v = q.astype(self._cdtype), k.astype(self._cdtype), v.astype(self._cdtype)

        # Broadcast KV if using GQA
        if Hk < H:
            # Repeat K and V heads to match Q heads
            k = jnp.repeat(k, H // Hk, axis=2)
            v = jnp.repeat(v, H // Hk, axis=2)

        is_tpu = any(d.platform == "tpu" for d in jax.devices())
        if self.cfg.use_flash_attn and HAS_PALLAS_ATTN and is_tpu and mesh is not None:
            # Pallas FlashAttention expects [Batch, Heads, Time, HeadDim]
            q, k, v = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
            out = shard_flash_attention(q, k, v, True, self.scale, mesh)
            out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        else:
            # Fallback
            q, k, v = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
            attn = jnp.einsum("bhid,bhjd->bhij", q, k) * self.scale
            mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
            attn = jnp.where(mask[None, None], attn, jnp.finfo(attn.dtype).min)
            attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(attn.dtype)
            out = jnp.einsum("bhij,bhjd->bhid", attn, v)
            out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
            
        return self.Wo(out)


class FFN(nnx.Module):
    def __init__(self, d_model: int, ffn_mult: int, rngs: nnx.Rngs,
                 compute_dtype=None) -> None:
        d_ff = d_model * ffn_mult
        self.W1 = nnx.Linear(d_model, d_ff, use_bias=False,
                              dtype=compute_dtype, rngs=rngs)
        self.W2 = nnx.Linear(d_ff, d_model, use_bias=False,
                              dtype=compute_dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.W2(jax.nn.gelu(self.W1(x)))


class TransformerBlock(nnx.Module):
    def __init__(self, cfg: DWAConfig, d_model: int, n_heads: int, n_kv_heads: int,
                 ffn_mult: int, rngs: nnx.Rngs,
                 compute_dtype=None, remat: bool = False) -> None:
        self.norm1 = nnx.RMSNorm(d_model, rngs=rngs)
        self.attn = CausalSelfAttention(cfg, d_model, n_heads, n_kv_heads, rngs, compute_dtype)
        self.norm2 = nnx.RMSNorm(d_model, rngs=rngs)
        self.ffn = FFN(d_model, ffn_mult, rngs, compute_dtype)
        self._remat = remat

    def __call__(self, x: jnp.ndarray, cos=None, sin=None, mesh=None) -> jnp.ndarray:
        if self._remat:
            attn_fn = jax.checkpoint(lambda h: self.attn(self.norm1(h), cos, sin, mesh))
            ffn_fn  = jax.checkpoint(lambda h: self.ffn(self.norm2(h)))
            x = x + attn_fn(x)
            x = x + ffn_fn(x)
        else:
            x = x + self.attn(self.norm1(x), cos, sin, mesh)
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
            TransformerBlock(cfg, cfg.d_A, cfg.n_heads, cfg.n_kv_heads,
                             cfg.ffn_mult, rngs, cfg.compute_dtype, cfg.remat)
            for _ in range(cfg.n_layers_A)
        ])
        self.norm = nnx.RMSNorm(cfg.d_A, rngs=rngs)

    def __call__(self, x: jnp.ndarray, cos=None, sin=None, mesh=None) -> jnp.ndarray:
        for block in self.blocks:
            x = block(x, cos, sin, mesh)
        return self.norm(x)   # [B, seq, d_A]


class PartB(nnx.Module):
    """
    Transformer decoder (Part B).
    Input:  h_mid [B, seq, d_B]
    Output: hidden states [B, seq, d_B]
    """

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs) -> None:
        self.blocks = nnx.List([
            TransformerBlock(cfg, cfg.d_B, cfg.n_heads, cfg.n_kv_heads,
                             cfg.ffn_mult, rngs, cfg.compute_dtype, cfg.remat)
            for _ in range(cfg.n_layers_B)
        ])
        self.norm = nnx.RMSNorm(cfg.d_B, rngs=rngs)

    def __call__(self, x: jnp.ndarray, cos=None, sin=None, mesh=None) -> jnp.ndarray:
        for block in self.blocks:
            x = block(x, cos, sin, mesh)
        return self.norm(x)   # [B, seq, d_B]
