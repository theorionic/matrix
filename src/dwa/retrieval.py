"""MultiAspectRetrieval — sigmoid-gated, multi-head cosine similarity retrieval."""

import jax
import jax.numpy as jnp
from flax import nnx

from .config import DWAConfig
from .utils import cosine_sim_batched


class CentroidEMA(nnx.Variable):
    """Non-trainable IVF centroids updated via EMA outside the optimizer."""
    pass


class MultiAspectRetrieval(nnx.Module):
    """
    Computes S-aspect cosine similarity between Part A hidden state queries
    and pool keys, then selects top-k via sigmoid gating or fixed top-k.

    Optionally uses IVF (Inverted File Index) to reduce bandwidth:
    1. Search centroids [S, C, d_k] to find top-m clusters.
    2. Search only keys in those clusters.

    Learnable parameters:
        W_Q           [S, d_k, d_A]  — aspect query projections
        aspect_weights [S]            — learnable aspect importance
        tau            scalar         — learnable selection threshold
    Non-trainable state:
        centroids      [S, C, d_k]    — IVF centroids (EMA of pool key partitions)
    """

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs) -> None:
        self.cfg = cfg
        scale = cfg.d_A ** -0.5
        self.W_Q = nnx.Param(
            jax.random.normal(rngs.params(), (cfg.S, cfg.d_k, cfg.d_A)) * scale
        )
        self.aspect_weights = nnx.Param(jnp.zeros(cfg.S))  # softmax'd
        self.tau = nnx.Param(jnp.array(0.0))               # learnable threshold

        # IVF centroids — non-trainable; updated via EMA of pool key partitions
        # in train_window so they always track actual pool key space, not routing bias.
        self.centroids = CentroidEMA(
            jax.random.normal(rngs.params(), (cfg.S, cfg.C, cfg.d_k)) * (cfg.d_k ** -0.5)
        )

    def __call__(
        self,
        z: jnp.ndarray,           # [B, d_A]
        pool_keys: jnp.ndarray,   # [S, N, d_k]  (may be N_local when model-sharded)
        lambda_val: float,        # sharpness
        is_warmup: bool,          # static
        mesh=None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns:
            alphas:  [B, k_max]
            indices: [B, k_max]
        """
        cfg = self.cfg
        B = z.shape[0]

        # Aspect queries: [B, S, d_k]
        queries = jnp.einsum("ska,ba->bsk", self.W_Q[...], z)
        q_norm = queries / (jnp.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)

        # Aspect weights (shared between IVF and full-search paths)
        w = jax.nn.softmax(self.aspect_weights[...], axis=0)  # [S]

        # IVF is only valid when pool_keys covers the full N (not model-sharded).
        # When pool is model-sharded each device has N_local < N keys, so the
        # global offset arithmetic (cluster * N_per_C) would index out of range.
        model_sharded = (
            mesh is not None
            and "model" in mesh.axis_names
            and mesh.shape["model"] > 1
        )
        use_ivf_now = cfg.use_ivf and not model_sharded

        if use_ivf_now:
            # ── Stage 1: centroid search (tiny; fits in L1 cache) ────────────
            # Centroids are EMA of pool key partitions (not gradient-trained),
            # so they track actual key-space positions without routing bias.
            c_norm = self.centroids[...] / (
                jnp.linalg.norm(self.centroids[...], axis=-1, keepdims=True) + 1e-8
            )
            c_sim   = jnp.einsum("bsk,sck->bsc", q_norm, c_norm)   # [B, S, C]
            c_score = jnp.einsum("s,bsc->bc", w, c_sim)             # [B, C]
            m_eff = min(cfg.m, cfg.C)
            _, top_clusters = jax.lax.top_k(c_score, m_eff)         # [B, m_eff]

            # ── Stage 2: exact search within selected clusters ────────────────
            N_per_C = cfg.N // cfg.C
            cand_idx = (
                top_clusters[:, :, None] * N_per_C
                + jnp.arange(N_per_C)[None, None, :]
            ).reshape(B, m_eff * N_per_C)                            # [B, K_refine]

            keys_refine = jax.vmap(lambda idx: pool_keys[:, idx, :])(cand_idx)
            # → [B, S, K_refine, d_k]
            k_norm  = keys_refine / (
                jnp.linalg.norm(keys_refine, axis=-1, keepdims=True) + 1e-8
            )
            sim_ref = jnp.einsum("bsk,bsnk->bsn", q_norm, k_norm)   # [B, S, K_refine]
            s_i     = jnp.einsum("s,bsn->bn", w, sim_ref)            # [B, K_refine]
            candidate_indices = cand_idx                              # [B, K_refine]

            # ── Full-pool soft scores for l_util ─────────────────────────────
            # Computing soft_full only over IVF candidates means l_util gradient
            # never reaches the 93%+ of pool vectors not searched this step —
            # the root cause of pool collapse.  We pay one extra full-pool
            # matmul here (cheap vs. TPU headroom) so every vector gets gradient.
            p_norm_full = pool_keys / (
                jnp.linalg.norm(pool_keys, axis=-1, keepdims=True) + 1e-8
            )
            sim_all  = jnp.einsum("bsk,snk->bsn", q_norm, p_norm_full)  # [B, S, N]
            s_i_full = jnp.einsum("s,bsn->bn", w, sim_all)               # [B, N]
            soft_full = jax.nn.softmax(s_i_full / cfg.T, axis=-1)        # [B, N]

        else:
            # ── Full search (used when model-sharded or IVF disabled) ────────
            p_norm = pool_keys / (jnp.linalg.norm(pool_keys, axis=-1, keepdims=True) + 1e-8)
            sim    = jnp.einsum("bsk,snk->bsn", q_norm, p_norm)      # [B, S, N_local]
            s_i    = jnp.einsum("s,bsn->bn", w, sim)                 # [B, N_local]

            # All-gather across model axis so top_k sees the full [B, N] scores.
            if model_sharded:
                from jax.sharding import NamedSharding, PartitionSpec as P
                s_i = jax.lax.with_sharding_constraint(
                    s_i, NamedSharding(mesh, P("data", None))
                )  # [B, N]

            N_full = s_i.shape[1]
            candidate_indices = jnp.broadcast_to(
                jnp.arange(N_full, dtype=jnp.int32), (B, N_full)
            )
            soft_full = jax.nn.softmax(s_i / cfg.T, axis=-1)         # [B, N]

        # ── Selection (warmup = softmax top-k; gate = sigmoid-gated) ─────────
        # soft_full [B, N] covers ALL pool vectors regardless of IVF path,
        # so l_util entropy gradient reaches every vector every step.
        def warmup_select(_):
            scores, local_idx = jax.lax.top_k(s_i, cfg.k_max)       # [B, k_max]
            global_idx = jnp.take_along_axis(candidate_indices, local_idx, axis=1)
            alpha = jax.nn.softmax(scores / cfg.T, axis=-1)
            return alpha, global_idx, soft_full

        def gate_select(_):
            g   = jax.nn.sigmoid(lambda_val * (s_i - self.tau[...]))
            raw = g * jnp.exp(s_i / cfg.T)
            raw = raw / (raw.sum(axis=-1, keepdims=True) + 1e-8)
            top_raw, local_idx = jax.lax.top_k(raw, cfg.k_max)
            global_idx = jnp.take_along_axis(candidate_indices, local_idx, axis=1)
            alpha = top_raw / (top_raw.sum(axis=-1, keepdims=True) + 1e-8)
            return alpha, global_idx, soft_full

        alphas, indices, soft_full_out = jax.lax.cond(is_warmup, warmup_select, gate_select, None)
        return alphas, indices, soft_full_out
