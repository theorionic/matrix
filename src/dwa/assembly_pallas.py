"""
Optimized DWA assembly via Pallas kernel + jax.custom_vjp.

Two optimizations over the baseline:
  1. Key-cache: pool keys [S, N, d_k] are pre-computed once per window
     (not per step), turning a 2 GB pool read into a 32 MB cache read.
  2. Pallas assembly: the gather → factorize → accumulate W → apply residual
     chain runs in VMEM without materialising the intermediate W matrix
     back to HBM.  The backward pass is written by hand (custom_vjp) to
     avoid Mosaic autodiff limitations.

Usage:
    from src.dwa.assembly_pallas import pallas_assemble, assemble_jax

pallas_assemble is the optimized path; assemble_jax is the pure-JAX
fallback used if the Pallas kernel cannot be compiled (e.g., on GPU / CPU).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from flax import nnx


# ---------------------------------------------------------------------------
# Pure-JAX reference (used as fallback and for gradient testing)
# ---------------------------------------------------------------------------

def assemble_jax(
    gathered: jnp.ndarray,   # [B, k, D]
    alphas: jnp.ndarray,     # [B, k]
    h_A: jnp.ndarray,        # [B, T, d_A]
    W_base: jnp.ndarray,     # [d_B, d_A]
    b_base: jnp.ndarray,     # [d_B]
    gamma: jnp.ndarray,      # scalar
    d_B: int,
    r: int,
    d_A: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pure-JAX assembly + residual application. Returns (h_mid, W)."""
    B, k, D = gathered.shape
    s1 = d_B * r
    s2 = s1 + r * d_A

    U = gathered[:, :, :s1].reshape(B, k, d_B, r)    # [B,k,d_B,r]
    V = gathered[:, :, s1:s2].reshape(B, k, r, d_A)  # [B,k,r,d_A]
    b_vec = gathered[:, :, s2:s2 + d_B]               # [B,k,d_B]

    # Assemble W per batch item
    aU = alphas[:, :, None, None] * U                 # [B,k,d_B,r]
    aU_flat = aU.reshape(B * k, d_B, r)
    V_flat  = V.reshape(B * k, r, d_A)
    delta_W = jnp.matmul(aU_flat, V_flat).reshape(B, k, d_B, d_A).sum(1)  # [B,d_B,d_A]
    W = W_base[None, ...] + delta_W                    # [B,d_B,d_A]

    delta_b = (alphas[:, :, None] * b_vec).sum(1)     # [B,d_B]
    bias = b_base[None, :] + delta_b                   # [B,d_B]

    # Residual: h_mid = h_A + gamma * h_A @ W^T + bias
    h_res = jnp.matmul(h_A, W.transpose(0, 2, 1))     # [B,T,d_B]
    h_mid = h_A + gamma * h_res + bias[:, None, :]
    return h_mid, W


# ---------------------------------------------------------------------------
# Pallas assembly kernel (forward only — backward handled by custom_vjp)
# ---------------------------------------------------------------------------

def _make_pallas_kernel(B: int, k: int, T: int, d_B: int, r: int, d_A: int):
    """Return a compiled pallas_call for the given static shapes."""
    s1 = d_B * r
    s2 = s1 + r * d_A
    s3 = s2 + d_B

    # gamma is a scalar — reshape to [1] to satisfy Pallas rank >= 1 requirement
    def _kernel(gathered_ref, alphas_ref, h_A_ref, W_base_ref, b_base_ref, gamma_ref, out_ref):
        gathered = gathered_ref[...]          # [B,k,D]
        alphas   = alphas_ref[...]            # [B,k]
        h_A      = h_A_ref[...]              # [B,T,d_A]
        W_base   = W_base_ref[...]           # [d_B,d_A]
        b_base   = b_base_ref[...]           # [d_B]
        gamma    = gamma_ref[0]              # scalar from [1] array

        U = gathered[:, :, :s1].reshape(B, k, d_B, r)
        V = gathered[:, :, s1:s2].reshape(B, k, r, d_A)
        b_vec = gathered[:, :, s2:s3]        # [B,k,d_B]

        # Weighted low-rank accumulation (decomposed to avoid 3-tensor einsum)
        aU = alphas[:, :, None, None] * U    # [B,k,d_B,r]
        delta_W = jnp.matmul(
            aU.reshape(B * k, d_B, r), V.reshape(B * k, r, d_A)
        ).reshape(B, k, d_B, d_A).sum(axis=1)               # [B,d_B,d_A]

        W = W_base[None, ...] + delta_W                       # [B,d_B,d_A]
        delta_b = (alphas[:, :, None] * b_vec).sum(axis=1)   # [B,d_B]
        bias = b_base[None, :] + delta_b                      # [B,d_B]

        h_res = jnp.matmul(h_A, W.transpose(0, 2, 1))        # [B,T,d_B]
        out_ref[...] = h_A + gamma * h_res + bias[:, None, :]

    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((B, T, d_A), jnp.float32),
    )


# Cache compiled kernels by shape to avoid recompilation
_kernel_cache: dict = {}


def _pallas_assemble_forward(
    gathered: jnp.ndarray,
    alphas: jnp.ndarray,
    h_A: jnp.ndarray,
    W_base: jnp.ndarray,
    b_base: jnp.ndarray,
    gamma: jnp.ndarray,
    d_B: int,
    r: int,
    d_A: int,
) -> jnp.ndarray:
    """Run the Pallas assembly forward kernel (returns h_mid only)."""
    B, k, _ = gathered.shape
    T = h_A.shape[1]
    key = (B, k, T, d_B, r, d_A)
    if key not in _kernel_cache:
        _kernel_cache[key] = _make_pallas_kernel(B, k, T, d_B, r, d_A)
    # Pallas requires rank >= 1 — reshape scalar gamma to [1]
    return _kernel_cache[key](gathered, alphas, h_A, W_base, b_base, gamma.reshape(1))


# ---------------------------------------------------------------------------
# custom_vjp wrapper: forward saves residuals, backward uses analytic grads
# ---------------------------------------------------------------------------

@functools.partial(jax.custom_jvp, nondiff_argnums=(6, 7, 8))
def pallas_assemble(
    gathered: jnp.ndarray,
    alphas: jnp.ndarray,
    h_A: jnp.ndarray,
    W_base: jnp.ndarray,
    b_base: jnp.ndarray,
    gamma: jnp.ndarray,
    d_B: int,
    r: int,
    d_A: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Pallas-accelerated assembly. Returns (h_mid [B,T,d_B], W [B,d_B,d_A]).

    Uses custom_jvp so JAX can compute a JVP without calling the Pallas kernel
    in tangent space (which would OOM on VMEM inside jax.lax.scan).
    The VJP is derived automatically by JAX from the analytic JVP below.
    """
    h_mid = _pallas_assemble_forward(gathered, alphas, h_A, W_base, b_base, gamma, d_B, r, d_A)
    # W needed for aux losses — compute in pure JAX
    B, k, _ = gathered.shape
    s1 = d_B * r
    s2 = s1 + r * d_A
    U = gathered[:, :, :s1].reshape(B, k, d_B, r)
    V = gathered[:, :, s1:s2].reshape(B, k, r, d_A)
    aU = alphas[:, :, None, None] * U
    delta_W = jnp.matmul(aU.reshape(B * k, d_B, r), V.reshape(B * k, r, d_A)
                         ).reshape(B, k, d_B, d_A).sum(1)
    W = W_base[None] + delta_W
    return h_mid, W


@pallas_assemble.defjvp
def _pallas_assemble_jvp(d_B, r, d_A, primals, tangents):
    """
    Analytic forward-mode JVP — pure JAX, no Pallas kernel in tangent path.

    Forward:
        W    = W_base + Σ_k α_k (U_k @ V_k)
        bias = b_base + Σ_k α_k b_k
        h_mid = h_A + γ (h_A @ W^T) + bias

    Tangent (linear in dot-inputs):
        Ẇ    = Ẇ_base + Σ_k [α̇_k (U_k@V_k) + α_k (U̇_k@V_k + U_k@V̇_k)]
        ḃias = ḃ_base + Σ_k [α̇_k b_k + α_k ḃ_k]
        ḣ_mid = ḣ_A + γ̇(h_A@W^T) + γ(ḣ_A@W^T + h_A@Ẇ^T) + ḃias
    """
    gathered, alphas, h_A, W_base, b_base, gamma = primals
    tg, ta, th_A, tW_base, tb_base, tgamma = tangents

    B, k, D = gathered.shape
    s1 = d_B * r
    s2 = s1 + r * d_A
    s3 = s2 + d_B

    U  = gathered[:, :, :s1].reshape(B, k, d_B, r)
    V  = gathered[:, :, s1:s2].reshape(B, k, r, d_A)
    bv = gathered[:, :, s2:s3]                         # [B,k,d_B]

    tU  = tg[:, :, :s1].reshape(B, k, d_B, r)
    tV  = tg[:, :, s1:s2].reshape(B, k, r, d_A)
    tbv = tg[:, :, s2:s3]

    # Primal W
    UV = jnp.matmul(U.reshape(B * k, d_B, r),
                    V.reshape(B * k, r, d_A)).reshape(B, k, d_B, d_A)
    delta_W = (alphas[:, :, None, None] * UV).sum(1)
    W = W_base[None] + delta_W                          # [B,d_B,d_A]
    bias = b_base[None] + (alphas[:, :, None] * bv).sum(1)  # [B,d_B]

    # Primal h_mid (use Pallas kernel)
    h_mid = _pallas_assemble_forward(gathered, alphas, h_A, W_base, b_base, gamma,
                                     d_B, r, d_A)

    # Tangent of W
    tUV = jnp.matmul(tU.reshape(B * k, d_B, r),
                     V.reshape(B * k, r, d_A)).reshape(B, k, d_B, d_A) + \
          jnp.matmul(U.reshape(B * k, d_B, r),
                     tV.reshape(B * k, r, d_A)).reshape(B, k, d_B, d_A)
    tdelta_W = ((ta[:, :, None, None] * UV) + (alphas[:, :, None, None] * tUV)).sum(1)
    tW = tW_base[None] + tdelta_W                       # [B,d_B,d_A]

    # Tangent of bias
    tbias = tb_base[None] + ((ta[:, :, None] * bv) + (alphas[:, :, None] * tbv)).sum(1)

    # Tangent of h_mid
    h_Wt = jnp.matmul(h_A, W.transpose(0, 2, 1))      # [B,T,d_B]  (primal h_A@W^T)
    th_mid = (th_A
              + tgamma * h_Wt
              + gamma * (jnp.matmul(th_A, W.transpose(0, 2, 1)) +
                         jnp.matmul(h_A, tW.transpose(0, 2, 1)))
              + tbias[:, None, :])

    return (h_mid, W), (th_mid, tW)


# ---------------------------------------------------------------------------
# Multi-device wrapper: pallas_assemble via shard_map
# ---------------------------------------------------------------------------

def shard_pallas_assemble(
    gathered: jnp.ndarray,   # [B, k, D]   — sharded on batch
    alphas: jnp.ndarray,     # [B, k]       — sharded on batch
    h_A: jnp.ndarray,        # [B, T, d_A] — sharded on batch
    W_base: jnp.ndarray,     # [d_B, d_A]  — replicated
    b_base: jnp.ndarray,     # [d_B]        — replicated
    gamma: jnp.ndarray,      # scalar        — replicated
    d_B: int,
    r: int,
    d_A: int,
    mesh,                    # jax.sharding.Mesh
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    pallas_assemble wrapped in shard_map so each device runs the Pallas
    kernel on its local batch slice independently.

    Mosaic kernels cannot be auto-partitioned by GSPMD, but they work fine
    when each device owns its own data via shard_map.  The custom_vjp
    backward is also mapped per-shard; gradients for replicated inputs
    (W_base, b_base, gamma) are all-reduced by shard_map automatically.
    """
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    def _fn(g, a, h, wb, bb, gm):
        return pallas_assemble(g, a, h, wb, bb, gm, d_B, r, d_A)

    return shard_map(
        _fn,
        mesh=mesh,
        in_specs=(
            P("batch", None, None),  # gathered — batch-sharded
            P("batch", None),        # alphas   — batch-sharded
            P("batch", None, None),  # h_A      — batch-sharded
            P(),                     # W_base   — replicated
            P(),                     # b_base   — replicated
            P(),                     # gamma    — replicated
        ),
        out_specs=(
            P("batch", None, None),  # h_mid — batch-sharded
            P("batch", None, None),  # W     — batch-sharded
        ),
        check_rep=False,
    )(gathered, alphas, h_A, W_base, b_base, gamma)


# ---------------------------------------------------------------------------
# Key-cache helper (precompute S × N × d_k projection once per window)
# ---------------------------------------------------------------------------

def compute_key_cache(
    pool_vectors: jnp.ndarray,   # [N, D]
    key_proj: jnp.ndarray,        # [S, D, d_k]
) -> jnp.ndarray:                 # [S, N, d_k]
    """
    Precompute the key cache for all pool vectors.
    Call once per training window; use result for all retrieval steps inside.
    """
    return jnp.einsum("nd,sda->sna", pool_vectors, key_proj)
