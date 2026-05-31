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
        self.W_input = nnx.Param(
            jax.random.normal(rngs.params(), (cfg.S, cfg.d_A, cfg.d_A)) * scale
        )
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
        gate_mix = 0.0,          # 0.0 = pure warmup, 1.0 = pure gate; JAX scalar or float
        mesh=None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Returns:
            alphas:  [B, k_max]
            indices: [B, k_max]
            soft_full: [B, N]
            l_z:     scalar, z-loss for score magnitude stabilization
        """
        cfg = self.cfg
        B = z.shape[0]

        # Per-aspect input projections diversify queries across aspects
        z_aspects = jnp.einsum("sda,ba->bsa", self.W_input[...], z)
        queries = jnp.einsum("ska,bsa->bsk", self.W_Q[...], z_aspects)
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
            if cfg.approx_soft_full:
                # Centroid approximation: ~0.5 MB instead of 34 MB per step.
                # Each pool vector inherits its cluster's softmax score uniformly.
                # c_score [B, C] already computed in Stage 1 — zero extra bandwidth.
                # Tradeoff: l_util/l_reuse gradient operates at cluster granularity
                # instead of per-vector.  Revival mechanism handles dead vectors.
                N_per_C  = cfg.N // cfg.C
                c_soft   = jax.nn.softmax(c_score / cfg.T, axis=-1)     # [B, C]
                soft_full = jnp.repeat(c_soft, N_per_C, axis=-1) / N_per_C  # [B, N]
                # s_i_full for z-loss: expand cluster scores to [B, N]
                s_i_full = jnp.repeat(c_score, N_per_C, axis=-1)        # [B, N]
            else:
                # Exact full-pool pass: reads [S,N,d_k]=32 MB per step.
                # Every vector gets precise per-vector gradient for l_util/l_reuse.
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

        # Z-loss: (log Σ exp(s_i / T))² per sample, averaged over batch.
        # Prevents score magnitude explosion which causes softmax concentration → collapse.
        # Use the full-pool scores for z-loss (s_i_full in IVF path, s_i otherwise).
        s_for_z = s_i_full if use_ivf_now else s_i
        l_z = (jax.nn.logsumexp(s_for_z / cfg.T, axis=-1) ** 2).mean()

        # ── Selection ────────────────────────────────────────────────────────
        # Three modes controlled by is_warmup and gate_mix:
        #   is_warmup=True  → pure warmup (top-k + softmax)
        #   gate_mix=0      → pure warmup (top-k + softmax)
        #   gate_mix=1      → pure gate (sigmoid-gated)
        #   0 < gate_mix < 1 → blended: indices from gate, α blended warmup/gate
        #
        # soft_full [B, N] covers ALL pool vectors regardless of IVF path,
        # so l_util entropy gradient reaches every vector every step.
        #
        # Warmup exploration: hard top-k is deterministic, which creates a
        # positive-feedback loop (selected vectors get strong assembly gradient,
        # gain advantage, get re-selected) → pool collapse before aux losses
        # can counter-balance.  We add Gumbel noise to scores before top-k so
        # every vector has non-zero selection probability during warmup.  Noise
        # scale anneals via (1 - gate_mix) → vanishes once the gate is fully on.
        # Seed is derived from z (varies per-batch) to avoid threading RNGs.
        # explore_sigma anneals with gate_mix; computed with JAX ops so gate_mix
        # can be a traced scalar (no Python float() needed).
        explore_sigma = jnp.maximum(
            jnp.float32(cfg.min_explore_noise),
            cfg.warmup_explore_noise * (1.0 - gate_mix),
        )

        def _explore_scores(scores):
            # Deterministic per-batch seed: hash z into a uint32 via bit-cast.
            # Stop-grad: we do not want the noise scale tied to the query
            # gradient path.  bitcast_convert_type avoids int-cast saturation
            # for arbitrary z magnitudes.
            seed_f = jax.lax.stop_gradient(jnp.sum(z.astype(jnp.float32)))
            seed_u = jax.lax.bitcast_convert_type(seed_f.astype(jnp.float32), jnp.uint32)
            key    = jax.random.PRNGKey(seed_u)
            gumbel = jax.random.gumbel(key, scores.shape).astype(scores.dtype)
            return scores + explore_sigma * gumbel

        def warmup_select(_):
            noisy = _explore_scores(s_i)
            _, local_idx = jax.lax.top_k(noisy, cfg.k_max)          # [B, k_max]
            # alpha weights use the CLEAN scores (not noise) so per-vector
            # contributions reflect retrieval quality, not exploration jitter.
            winner_scores = jnp.take_along_axis(s_i, local_idx, axis=1)
            global_idx = jnp.take_along_axis(candidate_indices, local_idx, axis=1)
            alpha = jax.nn.softmax(winner_scores / cfg.T, axis=-1)
            return alpha, global_idx, soft_full

        def gate_select(_):
            # Sigmoid gate provides soft selection; top-k on raw gated scores
            # (no pre-normalization or exponential — those cause hard winner-take-all).
            # Gumbel noise also applied here with the SAME (1-gate_mix) annealing,
            # so exploration continues through the gate ramp.  Without this,
            # blended_select's indices come from a fully-deterministic top-k and
            # the pool collapses as soon as the gate ramps in (the issue we saw
            # after the warmup fix: unique drops from 99% → 8% at gate_on entry).
            g   = jax.nn.sigmoid(lambda_val * (s_i - self.tau[...]))
            raw = g * s_i
            noisy = _explore_scores(raw)
            _, local_idx = jax.lax.top_k(noisy, cfg.k_max)
            global_idx = jnp.take_along_axis(candidate_indices, local_idx, axis=1)
            winner_scores = jnp.take_along_axis(s_i, local_idx, axis=1)
            alpha = jax.nn.softmax(winner_scores / cfg.T, axis=-1)
            return alpha, global_idx, soft_full

        def blended_select(_):
            wu_alpha, wu_idx, _ = warmup_select(None)
            g_alpha, g_idx, _ = gate_select(None)
            alpha = (1.0 - gate_mix) * wu_alpha + gate_mix * g_alpha
            return alpha, g_idx, soft_full

        if is_warmup:
            alphas, indices, soft_full_out = warmup_select(None)
        else:
            # gate_mix may be a JAX traced scalar — use lax.switch so gate_mix
            # can vary per-step without triggering recompilation.
            idx = jnp.where(gate_mix <= 0.0, 0,
                            jnp.where(gate_mix >= 1.0, 2, 1)).astype(jnp.int32)
            alphas, indices, soft_full_out = jax.lax.switch(
                idx, [warmup_select, blended_select, gate_select], None
            )
        return alphas, indices, soft_full_out, l_z
