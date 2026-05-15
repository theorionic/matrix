"""MultiAspectRetrieval — sigmoid-gated, multi-head cosine similarity retrieval."""

import jax
import jax.numpy as jnp
from flax import nnx

from .config import DWAConfig
from .utils import cosine_sim_batched


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
        centroids      [S, C, d_k]    — IVF centroids
    """

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs) -> None:
        self.cfg = cfg
        scale = cfg.d_A ** -0.5
        self.W_Q = nnx.Param(
            jax.random.normal(rngs.params(), (cfg.S, cfg.d_k, cfg.d_A)) * scale
        )
        self.aspect_weights = nnx.Param(jnp.zeros(cfg.S))  # softmax'd
        self.tau = nnx.Param(jnp.array(0.0))               # learnable threshold
        
        # IVF centroids
        self.centroids = nnx.Param(
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
            # Score all C centroids using the combined aspect-weighted similarity.
            # Select the same top-m clusters for every aspect — this keeps the
            # candidate index set uniform so the Stage-2 global-index map is
            # trivially correct (no per-aspect offset juggling needed).
            c_norm = self.centroids[...] / (
                jnp.linalg.norm(self.centroids[...], axis=-1, keepdims=True) + 1e-8
            )
            c_sim   = jnp.einsum("bsk,sck->bsc", q_norm, c_norm)   # [B, S, C]
            c_score = jnp.einsum("s,bsc->bc", w, c_sim)             # [B, C]
            _, top_clusters = jax.lax.top_k(c_score, cfg.m)         # [B, m]

            # ── Stage 2: exact search within selected clusters ────────────────
            # Build the flat candidate index list [B, K_refine].
            # Each cluster c owns exactly N_per_C consecutive pool indices:
            #   [c*N_per_C, ..., (c+1)*N_per_C - 1]
            N_per_C = cfg.N // cfg.C
            # [B, m, 1] + [N_per_C] → [B, m, N_per_C] → [B, K_refine]
            cand_idx = (
                top_clusters[:, :, None] * N_per_C
                + jnp.arange(N_per_C)[None, None, :]
            ).reshape(B, cfg.m * N_per_C)                            # [B, K_refine]

            # Gather keys for all S aspects at the candidate indices.
            # vmap over the batch dim: for each example b, gather pool_keys[:, cand_idx[b], :]
            keys_refine = jax.vmap(lambda idx: pool_keys[:, idx, :])(cand_idx)
            # → [B, S, K_refine, d_k]

            k_norm  = keys_refine / (
                jnp.linalg.norm(keys_refine, axis=-1, keepdims=True) + 1e-8
            )
            sim_ref = jnp.einsum("bsk,bsnk->bsn", q_norm, k_norm)   # [B, S, K_refine]
            s_i     = jnp.einsum("s,bsn->bn", w, sim_ref)            # [B, K_refine]
            candidate_indices = cand_idx                              # [B, K_refine]

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

        # ── Selection (warmup = softmax top-k; gate = sigmoid-gated) ─────────
        # Both branches also return soft_full: the normalised distribution over
        # ALL candidate vectors before top-k selection.  This is used by the
        # utilisation loss to compute retrieval entropy — giving real gradient
        # signal through W_Q / key_proj (unlike the old pool_ema-based formula
        # which was a constant w.r.t. nnx.Param and had zero gradient).
        def warmup_select(_):
            scores, local_idx = jax.lax.top_k(s_i, cfg.k_max)       # [B, k_max]
            global_idx = jnp.take_along_axis(candidate_indices, local_idx, axis=1)
            alpha = jax.nn.softmax(scores / cfg.T, axis=-1)
            soft_full = jax.nn.softmax(s_i / cfg.T, axis=-1)         # [B, N_cands]
            return alpha, global_idx, soft_full

        def gate_select(_):
            g   = jax.nn.sigmoid(lambda_val * (s_i - self.tau[...]))
            raw = g * jnp.exp(s_i / cfg.T)
            raw = raw / (raw.sum(axis=-1, keepdims=True) + 1e-8)
            top_raw, local_idx = jax.lax.top_k(raw, cfg.k_max)
            global_idx = jnp.take_along_axis(candidate_indices, local_idx, axis=1)
            alpha = top_raw / (top_raw.sum(axis=-1, keepdims=True) + 1e-8)
            return alpha, global_idx, raw                              # raw = full soft dist

        alphas, indices, soft_full = jax.lax.cond(is_warmup, warmup_select, gate_select, None)
        return alphas, indices, soft_full
