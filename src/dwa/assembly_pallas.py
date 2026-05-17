"""
Optimized DWA assembly via Pallas kernel + jax.custom_vjp.

Two optimizations over the baseline:
  1. Key-cache: pool keys [S, N, d_k] are pre-computed once per window
     (not per step), turning a 2 GB pool read into a 32 MB cache read.
  2. Pallas assembly: the gather → factorize → accumulate W → apply residual
     chain runs in VMEM without materialising the intermediate W matrix
     back to HBM.  The backward pass is written by hand (custom_vjp) to
     avoid Mosaic autodiff limitations and VMEM contention inside scan.

Why custom_vjp (not custom_jvp):
  custom_jvp causes VMEM explosion inside jax.lax.scan + value_and_grad
  because scan's backward re-runs the JVP rule, which calls the Pallas
  kernel in the primal slot of the JVP.  Both the primal Pallas kernel and
  the tangent computations compete for VMEM simultaneously.

  custom_vjp avoids this: the backward rule is pure JAX (no Pallas), so
  scan's backward pass has zero VMEM pressure.  The forward Pallas kernel
  runs once per step inside the scan body as normal.

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

_B_BLOCK = 8  # batch tile; must be divisible by 8 (Mosaic second-to-last dim rule)


def _make_pallas_kernel(B: int, T: int, d_B: int, kr: int, d_A: int, dtype=jnp.float32):
    """Pallas assembly kernel: two batch matmuls, no Python loops.

    Inputs are pre-factored by the caller so all BlockSpec dimensions satisfy
    Mosaic's alignment constraints (second-to-last div-by-8, last div-by-128):
        V_scaled [B, kr, d_A]  — alpha-scaled V factors; kr=k*r (div by 8)
        h_A      [B, T,  d_A]  — Part A hidden states
        U_flat   [B, kr, d_B]  — reshaped U factors
        W_base   [d_B, d_A]    — replicated base weight
        pb       [B, d_B]      — pre-computed alpha-weighted bias (b_base included)
        gamma    [1]            — residual scale (scalar reshaped to rank-1)

    VMEM per B_BLOCK=8 block (medium config d=128, kr=192, T=256): ~7 MB.
    dtype matches the activation dtype (float32 or bfloat16).
    """
    Bb = min(_B_BLOCK, B)

    def _kernel(Vs_ref, hA_ref, Uf_ref, Wb_ref, pb_ref, gm_ref, out_ref):
        Vs    = Vs_ref[...]        # [Bb, kr, d_A]
        hA    = hA_ref[...]       # [Bb, T, d_A]
        Uf    = Uf_ref[...]       # [Bb, kr, d_B]
        Wb    = Wb_ref[...]       # [d_B, d_A]
        pb    = pb_ref[...]       # [Bb, d_B]
        gamma = gm_ref[0]

        # Two batch matmuls replace the k-loop:
        #   h_A @ V_scaled^T → [Bb, T, kr]: captures alpha-scaled V projections
        #   (h_A@Vs^T) @ U_flat → [Bb, T, d_B]: assembles weighted residual
        h_V         = jnp.matmul(hA, Vs.transpose(0, 2, 1))  # [Bb, T, kr]
        h_res_delta = jnp.matmul(h_V, Uf)                     # [Bb, T, d_B]
        h_base      = jnp.matmul(hA, Wb.T)                    # [Bb, T, d_B]

        out_ref[...] = hA + gamma * (h_base + h_res_delta) + pb[:, None, :]

    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((B, T, d_A), dtype),
        in_specs=[
            pl.BlockSpec((Bb, kr, d_A), lambda i: (i, 0, 0)),  # V_scaled
            pl.BlockSpec((Bb, T,  d_A), lambda i: (i, 0, 0)),  # h_A
            pl.BlockSpec((Bb, kr, d_B), lambda i: (i, 0, 0)),  # U_flat
            pl.BlockSpec((d_B, d_A),    lambda i: (0, 0)),      # W_base (full)
            pl.BlockSpec((Bb, d_B),     lambda i: (i, 0)),      # pb
            pl.BlockSpec((1,),          lambda i: (0,)),         # gamma
        ],
        out_specs=pl.BlockSpec((Bb, T, d_A), lambda i: (i, 0, 0)),
        grid=(B // Bb,),
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
    """Run the Pallas assembly forward kernel (returns h_mid only).

    Pre-computes V_scaled and U_flat in JAX so the Pallas kernel only receives
    aligned 3D tensors (all dims divisible by 8/128 as Mosaic requires).
    """
    B, k, _ = gathered.shape
    T = h_A.shape[1]
    s1 = d_B * r
    s2 = s1 + r * d_A
    s3 = s2 + d_B
    kr = k * r

    U     = gathered[:, :, :s1].reshape(B, k, d_B, r)
    V     = gathered[:, :, s1:s2].reshape(B, k, r, d_A)
    b_vec = gathered[:, :, s2:s3]                               # [B, k, d_B]

    # Factor outside the kernel: small pure-JAX ops, no HBM spill risk
    compute_dtype = h_A.dtype  # match kernel dtype to activation dtype (float32 or bfloat16)
    V_scaled = (alphas[:, :, None, None] * V).reshape(B, kr, d_A)  # [B, kr, d_A]
    U_flat   = U.transpose(0, 1, 3, 2).reshape(B, kr, d_B)         # [B, kr, d_B]
    pb       = (b_base.astype(compute_dtype) +
                jnp.einsum("bk,bkd->bd", alphas, b_vec))            # [B, d_B]

    key = (B, T, d_B, kr, d_A, compute_dtype)
    if key not in _kernel_cache:
        _kernel_cache[key] = _make_pallas_kernel(B, T, d_B, kr, d_A, dtype=compute_dtype)
    return _kernel_cache[key](
        V_scaled, h_A, U_flat,
        W_base.astype(compute_dtype),
        pb,
        gamma.reshape(1).astype(compute_dtype),
    )


# ---------------------------------------------------------------------------
# custom_vjp wrapper: forward saves residuals, backward uses analytic grads
#
# Why custom_vjp instead of custom_jvp:
#   custom_jvp re-runs the Pallas kernel inside scan's backward JVP context,
#   creating simultaneous VMEM pressure for both primal and tangent.
#   custom_vjp backward is pure JAX — scan's backward has zero VMEM pressure.
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
    Pallas-accelerated assembly. Returns (h_mid [B,T,d_B], W [B,d_B,d_A]).

    Forward runs in VMEM via Pallas (no HBM writes for delta_W, U, V).
    Backward is analytic pure-JAX (no Pallas) — safe inside scan+value_and_grad.
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


def _pallas_assemble_fwd(
    gathered: jnp.ndarray,
    alphas: jnp.ndarray,
    h_A: jnp.ndarray,
    W_base: jnp.ndarray,
    b_base: jnp.ndarray,
    gamma: jnp.ndarray,
    d_B: int,
    r: int,
    d_A: int,
):
    """Forward pass: run Pallas kernel, save residuals for backward."""
    primals_out = pallas_assemble(gathered, alphas, h_A, W_base, b_base, gamma, d_B, r, d_A)
    h_mid, W = primals_out
    # Save residuals needed for backward: gathered (for U,V,b_vec grads),
    # alphas, h_A (for g_W), W (for g_hA from h_res), gamma (for g_gamma).
    # h_res is recomputed in backward from h_A and W to save HBM vs storing it.
    residuals = (gathered, alphas, h_A, W, gamma)
    return primals_out, residuals


def _pallas_assemble_bwd(d_B: int, r: int, d_A: int, residuals, g):
    """
    Backward pass: pure-JAX analytic VJP. No Pallas → no VMEM pressure.

    Forward:
        U [B,k,d_B,r], V [B,k,r,d_A], b_vec [B,k,d_B] = split(gathered)
        W = W_base + sum_k alpha_k * U_k @ V_k         [B,d_B,d_A]
        bias = b_base + sum_k alpha_k * b_vec_k         [B,d_B]
        h_res = h_A @ W^T                               [B,T,d_B]
        h_mid = h_A + gamma * h_res + bias              [B,T,d_B]

    Backward from (g_hmid [B,T,d_B], g_W [B,d_B,d_A]):
    """
    gathered, alphas, h_A, W, gamma = residuals
    g_hmid, g_W_out = g  # cotangents from both outputs (h_mid, W)

    B, k, D = gathered.shape
    s1 = d_B * r
    s2 = s1 + r * d_A
    s3 = s2 + d_B

    U     = gathered[:, :, :s1].reshape(B, k, d_B, r)  # [B,k,d_B,r]
    V     = gathered[:, :, s1:s2].reshape(B, k, r, d_A) # [B,k,r,d_A]
    b_vec = gathered[:, :, s2:s3]                        # [B,k,d_B]

    # h_mid = h_A + gamma * h_res + bias  where h_res = h_A @ W^T
    h_res = jnp.matmul(h_A, W.transpose(0, 2, 1))  # [B,T,d_B]

    # Gradients from h_mid output
    g_gamma  = jnp.sum(g_hmid * h_res)               # scalar
    g_bias   = g_hmid.sum(axis=1)                     # [B,d_B]
    g_h_res  = gamma * g_hmid                         # [B,T,d_B]

    # h_res = h_A @ W^T
    # g_W from h_res path: g_W[b,j,i] = sum_t g_h_res[b,t,j] * h_A[b,t,i]
    g_W_hres = jnp.einsum("btj,bti->bji", g_h_res, h_A)  # [B,d_B,d_A]
    # g_hA from h_res path: g_hA[b,t,i] = sum_j g_h_res[b,t,j] * W[b,j,i]
    g_hA_hres = jnp.einsum("btj,bji->bti", g_h_res, W)   # [B,T,d_A]

    g_hA = g_hmid + g_hA_hres                            # [B,T,d_A]  (direct + through h_res)

    # Total grad_W = from h_res path + from aux-loss W output
    g_W_total  = g_W_hres + g_W_out                      # [B,d_B,d_A]
    g_W_base   = g_W_total.sum(0)                        # [d_B,d_A]
    g_delta_W  = g_W_total                               # [B,d_B,d_A]

    # delta_W = sum_k alpha_k * (U_k @ V_k)
    UV = jnp.matmul(
        U.reshape(B * k, d_B, r), V.reshape(B * k, r, d_A)
    ).reshape(B, k, d_B, d_A)                           # [B,k,d_B,d_A]

    # g_alpha from delta_W: g_alpha[b,k] = sum_{j,i} UV[b,k,j,i] * g_delta_W[b,j,i]
    g_alpha_dW = jnp.einsum("bkji,bji->bk", UV, g_delta_W)   # [B,k]

    # g_U[b,k,j,r] = alpha[b,k] * sum_i g_delta_W[b,j,i] * V[b,k,r,i]
    #              = alpha * (g_delta_W @ V^T_per_k)
    # g_delta_W[:,None]: [B,1,d_B,d_A]; V.T: [B,k,d_A,r]
    g_U = (alphas[:, :, None, None] *
           jnp.matmul(g_delta_W[:, None, :, :],
                      V.transpose(0, 1, 3, 2)))          # [B,k,d_B,r]

    # g_V[b,k,r,i] = alpha[b,k] * sum_j U[b,k,j,r] * g_delta_W[b,j,i]
    #              = alpha * (U^T @ g_delta_W)
    # U.T: [B,k,r,d_B]; g_delta_W[:,None]: [B,1,d_B,d_A]
    g_V = (alphas[:, :, None, None] *
           jnp.matmul(U.transpose(0, 1, 3, 2),
                      g_delta_W[:, None, :, :]))         # [B,k,r,d_A]

    # bias = b_base + sum_k alpha_k * b_vec_k
    g_b_base    = g_bias.sum(0)                          # [d_B]
    g_alpha_bias = (b_vec * g_bias[:, None, :]).sum(-1)  # [B,k]
    g_b_vec     = alphas[:, :, None] * g_bias[:, None, :]  # [B,k,d_B]

    g_alpha = g_alpha_dW + g_alpha_bias                  # [B,k]

    # Pack grad_gathered: reassemble U,V,b_vec grads into [B,k,D] layout
    g_gathered = jnp.concatenate([
        g_U.reshape(B, k, s1),
        g_V.reshape(B, k, s2 - s1),
        g_b_vec,                                         # [B,k,d_B]
        jnp.zeros((B, k, D - s3), dtype=gathered.dtype),
    ], axis=-1)

    return g_gathered, g_alpha, g_hA, g_W_base, g_b_base, g_gamma


pallas_assemble.defvjp(_pallas_assemble_fwd, _pallas_assemble_bwd)


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
    try:
        from jax import shard_map
    except ImportError:
        from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    def _fn(g, a, h, wb, bb, gm):
        return pallas_assemble(g, a, h, wb, bb, gm, d_B, r, d_A)

    return shard_map(
        _fn,
        mesh=mesh,
        in_specs=(
            P("data", None, None),  # gathered — batch-sharded
            P("data", None),        # alphas   — batch-sharded
            P("data", None, None),  # h_A      — batch-sharded
            P(),                    # W_base   — replicated
            P(),                    # b_base   — replicated
            P(),                    # gamma    — replicated
        ),
        out_specs=(
            P("data", None, None),  # h_mid — batch-sharded
            P("data", None, None),  # W     — batch-sharded
        ),
        check_vma=False,
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
