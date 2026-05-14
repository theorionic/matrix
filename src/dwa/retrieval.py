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

    Learnable parameters:
        W_Q           [S, d_k, d_A]  — aspect query projections
        aspect_weights [S]            — learnable aspect importance
        tau            scalar         — learnable selection threshold
    """

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs) -> None:
        self.cfg = cfg
        scale = cfg.d_A ** -0.5
        self.W_Q = nnx.Param(
            jax.random.normal(rngs.params(), (cfg.S, cfg.d_k, cfg.d_A)) * scale
        )
        self.aspect_weights = nnx.Param(jnp.zeros(cfg.S))  # softmax'd
        self.tau = nnx.Param(jnp.array(0.0))               # learnable threshold

    def __call__(
        self,
        z: jnp.ndarray,           # [B, d_A]
        pool_keys: jnp.ndarray,   # [S, N, d_k]
        lambda_val: float,        # sharpness (Python scalar or 0-d array)
        is_warmup: bool,          # static: True → fixed top-k softmax
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns:
            alphas:  [B, k_max] — normalized assembly weights
            indices: [B, k_max] — pool indices of selected vectors
        """
        cfg = self.cfg

        # Step 1 — compute S aspect queries from z
        # W_Q: [S, d_k, d_A], z: [B, d_A] → queries: [B, S, d_k]
        queries = jnp.einsum("ska,ba->bsk", self.W_Q[...], z)

        # Step 2 — multi-aspect cosine similarity → [B, S, N]
        # pool_keys: [S, N, d_k], queries: [B, S, d_k]
        sim = jnp.einsum(
            "bsk,snk->bsn",
            queries / (jnp.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8),
            pool_keys / (jnp.linalg.norm(pool_keys, axis=-1, keepdims=True) + 1e-8),
        )  # [B, S, N]

        # Step 3 — combine aspects with learned weights
        w = jax.nn.softmax(self.aspect_weights[...], axis=0)   # [S]
        s_i = jnp.einsum("s,bsn->bn", w, sim)                 # [B, N]

        # Step 4a — warmup: fixed top-k with softmax (no sigmoid gate)
        def warmup_select(_):
            scores, idx = jax.lax.top_k(s_i, cfg.k_max)   # [B, k_max]
            alpha = jax.nn.softmax(scores / cfg.T, axis=-1)
            return alpha, idx

        # Step 4b — gate: sigmoid-gated selection
        def gate_select(_):
            g = jax.nn.sigmoid(lambda_val * (s_i - self.tau[...]))  # [B, N]
            raw = g * jnp.exp(s_i / cfg.T)
            raw = raw / (raw.sum(axis=-1, keepdims=True) + 1e-8)
            # take top k_max from the already-normalized distribution
            top_raw, idx = jax.lax.top_k(raw, cfg.k_max)           # [B, k_max]
            alpha = top_raw / (top_raw.sum(axis=-1, keepdims=True) + 1e-8)
            return alpha, idx

        alphas, indices = jax.lax.cond(is_warmup, warmup_select, gate_select, None)
        return alphas, indices
