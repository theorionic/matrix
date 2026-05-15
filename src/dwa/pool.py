"""VectorPool — stores N×D parameter matrix and S key projection heads."""

import jax
import jax.numpy as jnp
from flax import nnx

from .config import DWAConfig


class VectorPool(nnx.Module):
    """
    Stores the N pool vectors (each of dimension D) and the S key-projection
    matrices used for multi-aspect retrieval.

    Both the retrieval and assembly gradients flow through ``vectors``.
    Key projections live here so the gradient path "who should retrieve you?"
    stays coupled to the parameters that store the matrix factors.
    """

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs, pool_vectors=None) -> None:
        self.cfg = cfg
        pool_dtype = jnp.bfloat16 if cfg.bf16_pool else jnp.float32
        # Pool vectors: [N, D] — if pre-initialized (e.g. sharded), use that;
        # otherwise initialize randomly (safe only when N×D fits on one device).
        if pool_vectors is not None:
            self.vectors = nnx.Param(pool_vectors)
        else:
            self.vectors = nnx.Param(
                (jax.random.normal(rngs.params(), (cfg.N, cfg.D)) * 0.02).astype(pool_dtype)
            )
        # Key projections per aspect: [S, D, d_k]
        self.key_proj = nnx.Param(
            (jax.random.normal(rngs.params(), (cfg.S, cfg.D, cfg.d_k))
             * (cfg.D ** -0.5)).astype(pool_dtype)
        )

    def compute_keys(self) -> jnp.ndarray:
        """
        Project all pool vectors through each aspect key projection.
        Returns: [S, N, d_k] in float32 regardless of storage dtype.
        """
        vecs = self.vectors[...].astype(jnp.float32)
        kp   = self.key_proj[...].astype(jnp.float32)
        return jnp.einsum("nd,sda->sna", vecs, kp)

    def get_factors(self, indices: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Gather vectors for given indices and split into low-rank factors.

        indices: [k] integer indices into pool
        Returns: U [k, d_B, r], V [k, r, d_A], b [k, d_B]
        """
        cfg = self.cfg
        s1, s2, s3 = cfg.factor_split
        vecs = self.vectors[...][indices]          # [k, D]
        U = vecs[:, :s1].reshape(-1, cfg.d_B, cfg.r)        # [k, d_B, r]
        V = vecs[:, s1:s2].reshape(-1, cfg.r, cfg.d_A)      # [k, r, d_A]
        b = vecs[:, s2:s3]                                   # [k, d_B]
        return U, V, b
