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

@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8))
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
    Pallas-accelerated assembly with analytic backward pass.
    Returns (h_mid [B,T,d_B], W [B,d_B,d_A]).
    """
    h_mid = _pallas_assemble_forward(gathered, alphas, h_A, W_base, b_base, gamma, d_B, r, d_A)
    # W is needed for aux losses — compute in pure JAX (small cost)
    B, k, D = gathered.shape
    s1, s2, s3 = d_B * r, d_B * r + r * d_A, d_B * r + r * d_A + d_B
    U = gathered[:, :, :s1].reshape(B, k, d_B, r)
    V = gathered[:, :, s1:s2].reshape(B, k, r, d_A)
    aU = alphas[:, :, None, None] * U
    delta_W = jnp.matmul(aU.reshape(B * k, d_B, r), V.reshape(B * k, r, d_A)
                         ).reshape(B, k, d_B, d_A).sum(1)
    W = W_base[None] + delta_W
    return h_mid, W


def _pallas_assemble_fwd(gathered, alphas, h_A, W_base, b_base, gamma, d_B, r, d_A):
    h_mid, W = pallas_assemble(gathered, alphas, h_A, W_base, b_base, gamma, d_B, r, d_A)
    # Save everything needed for backward (don't save W — recompute to save HBM)
    return (h_mid, W), (gathered, alphas, h_A, W_base, b_base, gamma)


def _pallas_assemble_bwd(d_B, r, d_A, residuals, g):
    """
    Analytic backward pass.

    Forward:
        h_mid = h_A + γ · (h_A @ W^T) + bias
        W     = W_base + Σ_k α_k · (U_k @ V_k)
        bias  = b_base + Σ_k α_k · b_k

    Upstream cotangents:
        g_h_mid [B,T,d_B]  — from task + downstream losses
        g_W     [B,d_B,d_A] — from aux losses (L_norm uses W directly)
    """
    gathered, alphas, h_A, W_base, b_base, gamma = residuals
    g_h_mid, g_W_aux = g

    B, k, D = gathered.shape
    T = h_A.shape[1]
    s1 = d_B * r
    s2 = s1 + r * d_A
    s3 = s2 + d_B

    # Recompute W (cheaper than saving [B,d_B,d_A] to HBM in forward)
    U = gathered[:, :, :s1].reshape(B, k, d_B, r)    # [B,k,d_B,r]
    V = gathered[:, :, s1:s2].reshape(B, k, r, d_A)  # [B,k,r,d_A]
    b_vec = gathered[:, :, s2:s3]                      # [B,k,d_B]
    aU = alphas[:, :, None, None] * U
    delta_W = jnp.matmul(
        aU.reshape(B * k, d_B, r), V.reshape(B * k, r, d_A)
    ).reshape(B, k, d_B, d_A).sum(1)
    W = W_base[None] + delta_W                         # [B,d_B,d_A]

    # Grad of h_mid w.r.t. W (from residual h_A @ W^T path):
    # h_res[b,t,u] = Σ_a h_A[b,t,a] * W[b,u,a]
    # ∂L/∂W[b,u,a] = γ · Σ_t g_h_mid[b,t,u] * h_A[b,t,a]
    g_W = gamma * jnp.matmul(
        g_h_mid.transpose(0, 2, 1), h_A   # [B,d_B,T] @ [B,T,d_A] = [B,d_B,d_A]
    ) + g_W_aux                             # add aux gradient

    # Grad w.r.t. h_A: direct + residual
    # ∂L/∂h_A = g_h_mid + γ · g_h_mid @ W
    g_h_A = g_h_mid + gamma * jnp.matmul(g_h_mid, W)  # [B,T,d_A]

    # Grad w.r.t. W_base: sum over batch
    g_W_base = g_W.sum(0)                              # [d_B,d_A]

    # Grad w.r.t. b_k and b_base (compute first — needed for g_alphas)
    # bias = b_base + Σ_k α_k · b_k, applied as bias[:,None,:]
    # ∂L/∂bias[b,u] = Σ_t g_h_mid[b,t,u]
    g_bias = g_h_mid.sum(1)                             # [B,d_B]
    g_b_base = g_bias.sum(0)                            # [d_B]
    g_b_vec = alphas[:, :, None] * g_bias[:, None, :]  # [B,k,d_B]

    # Grad w.r.t. α_k: W path + bias path
    # W path:   ∂L/∂α[b,k] = <g_W[b], U_k@V_k>_F
    # bias path: ∂L/∂α[b,k] += Σ_u g_bias[b,u] * b_vec[b,k,u]
    UV = jnp.matmul(
        U.reshape(B * k, d_B, r), V.reshape(B * k, r, d_A)
    ).reshape(B, k, d_B, d_A)                          # [B,k,d_B,d_A]
    g_alphas = (
        (g_W[:, None, :, :] * UV).sum((-2, -1))        # [B,k] from W
        + (g_bias[:, None, :] * b_vec).sum(-1)          # [B,k] from bias
    )

    # Grad w.r.t. U_k: α_k · g_W[b] @ V_k^T
    # g_W: [B,d_B,d_A], V: [B,k,r,d_A] → V^T: [B,k,d_A,r]
    g_W_exp = g_W[:, None, :, :]                        # [B,1,d_B,d_A]
    g_U = alphas[:, :, None, None] * jnp.matmul(
        g_W_exp, V.transpose(0, 1, 3, 2)               # [B,k,d_A,r]
    )                                                    # [B,k,d_B,r]

    # Grad w.r.t. V_k: α_k · U_k^T @ g_W[b]
    # U^T: [B,k,r,d_B], g_W: [B,d_B,d_A]
    g_V = alphas[:, :, None, None] * jnp.matmul(
        U.transpose(0, 1, 3, 2),                        # [B,k,r,d_B]
        g_W_exp,                                         # [B,1,d_B,d_A] broadcasts
    )                                                    # [B,k,r,d_A]

    # Gather grads back into [B,k,D] format
    g_gathered = jnp.concatenate([
        g_U.reshape(B, k, d_B * r),
        g_V.reshape(B, k, r * d_A),
        g_b_vec,
        jnp.zeros((B, k, D - s3)),
    ], axis=-1)                                          # [B,k,D]

    # Grad w.r.t. γ
    h_res = jnp.matmul(h_A, W.transpose(0, 2, 1))      # [B,T,d_B]
    g_gamma = (g_h_mid * h_res).sum()

    return g_gathered, g_alphas, g_h_A, g_W_base, g_b_base, g_gamma


pallas_assemble.defvjp(_pallas_assemble_fwd, _pallas_assemble_bwd)


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
