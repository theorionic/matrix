"""ProductQuantizer — fast approximate pool key retrieval via lookup tables."""

import jax
import jax.numpy as jnp
from flax import nnx

from .config import DWAConfig


class PQVariable(nnx.Variable):
    """Non-trainable PQ state (codebook, codes) — updated offline, not by optimizer."""
    pass


class ProductQuantizer(nnx.Module):
    """
    Splits d_k-dimensional pool keys into M subspaces of d_sub = d_k // M dims,
    quantizes each subspace to K codewords (uint8), and scores queries via LUT lookup.

    Approximate score cost: O(M·K + N·M) vs O(N·d_k) for exact cosine.
    Codebook and codes are updated offline on the host; not gradient-trained.

    Scaling:
        N=65K  → ~16× faster than exact (with d_k=64, M=8)
        N=1M   → ~100× faster than exact
        N=10M  → only PQ (or IVF+PQ) is feasible on TPU HBM
    """

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs) -> None:
        assert cfg.d_k % cfg.pq_M == 0, (
            f"d_k={cfg.d_k} must be divisible by pq_M={cfg.pq_M}"
        )
        self.cfg = cfg
        M, K, d_sub = cfg.pq_M, cfg.pq_K, cfg.d_k // cfg.pq_M
        # codebook[s, m, k, d_sub] — K codewords per subspace per aspect
        self.codebook = PQVariable(jnp.zeros((cfg.S, M, K, d_sub), dtype=jnp.float32))
        # codes[s, n, m] uint8 — nearest codeword index for each pool key
        self.codes = PQVariable(jnp.zeros((cfg.S, cfg.N, M), dtype=jnp.uint8))

    def approx_scores(
        self,
        q_norm: jnp.ndarray,          # [B, S, d_k] normalized queries
        aspect_weights: jnp.ndarray,  # [S] softmax aspect weights
    ) -> jnp.ndarray:                 # [B, N] approximate cosine scores
        """
        Fast approximate similarity via lookup table (ADC — asymmetric distance computation).

        For each batch item and aspect, builds a [M, K] table of subspace dot products,
        then accumulates scores for all N vectors by table lookup on their uint8 codes.
        Memory peak: [B, N, M] per aspect via vmap (one batch item at a time).
        """
        cfg = self.cfg
        M, K, d_sub = cfg.pq_M, cfg.pq_K, cfg.d_k // cfg.pq_M
        B = q_norm.shape[0]

        # Split queries into subspaces: [B, S, M, d_sub]
        q_split = q_norm.reshape(B, cfg.S, M, d_sub)

        # Build lookup table: lut[b, s, m, k] = dot(q_split[b,s,m], codebook[s,m,k])
        lut = jnp.einsum("bsmd,smkd->bsmk", q_split, self.codebook[...])  # [B, S, M, K]

        codes = self.codes[...].astype(jnp.int32)  # [S, N, M]
        m_idx = jnp.arange(M)                      # [M]

        # Accumulate per-aspect scores; loop over S (≤4, unrolled at trace time).
        approx = jnp.zeros((B, cfg.N), dtype=jnp.float32)
        for s in range(cfg.S):
            lut_s   = lut[:, s]   # [B, M, K]
            codes_s = codes[s]    # [N, M]
            # gathered[b, n, m] = lut_s[b, m, codes_s[n, m]]
            # vmap over batch to keep peak memory at [N, M] per item
            def score_b(lut_b):              # [M, K]
                return lut_b[m_idx[None, :], codes_s].sum(-1)   # [N]
            approx = approx + aspect_weights[s] * jax.vmap(score_b)(lut_s)

        return approx  # [B, N]

    def update(self, pool_keys: jnp.ndarray) -> None:
        """
        Refit codebook via mini-batch k-means and re-encode all pool keys.
        Called on host (not inside JIT) every pq_update_interval steps.

        pool_keys: [S, N, d_k] float32 — L2-normalized pool keys
        """
        import numpy as np

        cfg = self.cfg
        M, K, d_sub = cfg.pq_M, cfg.pq_K, cfg.d_k // cfg.pq_M

        keys_np = np.array(pool_keys)  # host copy [S, N, d_k]
        new_codebook = np.zeros((cfg.S, M, K, d_sub), dtype=np.float32)
        new_codes    = np.zeros((cfg.S, cfg.N, M),    dtype=np.uint8)

        for s in range(cfg.S):
            for m in range(M):
                sub = keys_np[s, :, m * d_sub : (m + 1) * d_sub]  # [N, d_sub]
                centroids, assign = _kmeans(sub, K, seed=s * M + m)
                new_codebook[s, m] = centroids
                new_codes[s, :, m] = assign.astype(np.uint8)

        self.codebook.value = jnp.array(new_codebook)
        self.codes.value    = jnp.array(new_codes)


def _kmeans(data: "np.ndarray", K: int, seed: int = 0, n_iter: int = 20):
    """K-means on [N, d] data. Returns (centroids [K, d], assignments [N]).

    Uses sklearn MiniBatchKMeans with BLAS thread limit (avoids OpenBLAS crash
    on TPU machines with >128 CPU cores). Falls back to pure-numpy Lloyd's if
    sklearn or threadpoolctl unavailable.
    """
    import numpy as np

    try:
        from sklearn.cluster import MiniBatchKMeans

        # Limit BLAS threads at runtime — OpenBLAS crashes when thread count
        # exceeds its compile-time MAX_THREADS (128) on large TPU host machines.
        # threadpoolctl works after numpy is already loaded; env vars do not.
        try:
            from threadpoolctl import threadpool_limits
            blas_ctx = threadpool_limits(limits=32, user_api="blas")
        except ImportError:
            import contextlib
            blas_ctx = contextlib.nullcontext()

        with blas_ctx:
            km = MiniBatchKMeans(n_clusters=K, n_init=3, max_iter=n_iter, random_state=seed)
            km.fit(data)
        centroids = km.cluster_centers_.astype(np.float32)

    except Exception:
        # Pure-numpy fallback: no BLAS calls, safe on any machine.
        # Uses ||a-b||² = ||a||² + ||b||² - 2·a·b to avoid [N,K,d] broadcast.
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(data), K, replace=False)
        centroids = data[idx].copy().astype(np.float32)
        for _ in range(n_iter):
            dists  = _sq_dists(data, centroids)
            assign = np.argmin(dists, axis=-1)
            for k in range(K):
                mask = assign == k
                if mask.any():
                    centroids[k] = data[mask].mean(0)

    assign = np.argmin(_sq_dists(data, centroids), axis=-1)
    return centroids, assign


def _sq_dists(a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    """Squared L2 distances [N, K] between rows of a [N,d] and b [K,d].
    Uses a²+b²-2ab form to avoid materialising [N,K,d] intermediate."""
    import numpy as np
    return (
        (a ** 2).sum(-1, keepdims=True)   # [N, 1]
        + (b ** 2).sum(-1)                # [K]
        - 2.0 * (a @ b.T)                 # [N, K]  — small d, safe BLAS call
    ).clip(0)
