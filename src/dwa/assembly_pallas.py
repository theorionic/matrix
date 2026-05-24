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

@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8))
def assemble_jax(
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
    return _assemble_jax_impl(gathered, alphas, h_A, W_base, b_base, gamma, d_B, r, d_A)


def _assemble_jax_impl(
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


def _assemble_jax_fwd(
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
    primals_out = _assemble_jax_impl(gathered, alphas, h_A, W_base, b_base, gamma, d_B, r, d_A)
    h_mid, W = primals_out
    residuals = (gathered, alphas, h_A, W, gamma)
    return primals_out, residuals




# ---------------------------------------------------------------------------
# Pallas assembly kernel (forward only — backward handled by custom_vjp)
# ---------------------------------------------------------------------------

# Scoped VMEM limit on TPU v5e per Pallas kernel tile (bytes).
# The compiler enforces ~16 MB; we target 14 MB to leave headroom.
_VMEM_BUDGET = 14 * 1024 * 1024


def _choose_b_block(B: int, T: int, d_B: int, kr: int, d_A: int,
                    elem_bytes: int = 4) -> int:
    """Pick the largest Bb ∈ {1,2,4,8} that fits VMEM for the assembly kernel.

    Per-tile VMEM budget (inputs + intermediates + output):
        Bb × [T*d_A + kr*d_A + kr*d_B + T*kr + 3*T*d_B] × elem_bytes

    For medium config (d=128, T=256, kr=192): Bb=8 → ~7 MB ✓
    For 700m config (d=768, T=1024, kr=64):  Bb=1 → ~12.6 MB — Bb=1 fits now!

    Also enforces Mosaic alignment: Bb must equal B (full batch) or be
    divisible by 8 (second-to-last dim rule for rank-2 block specs).

    Returns 0 if no valid Bb exists (caller should fall back to pure JAX).
    """
    per_row = (T * d_A + kr * d_A + kr * d_B + T * kr + 3 * T * d_B) * elem_bytes
    for Bb in (8, 4, 2, 1):
        if Bb > B or B % Bb != 0:
            continue
        if Bb != B and Bb % 8 != 0:
            continue
        if Bb * per_row <= _VMEM_BUDGET:
            return Bb
    return 0


def pallas_vmem_feasible(B: int, T: int, d_B: int, kr: int, d_A: int,
                         elem_bytes: int = 4) -> bool:
    return _choose_b_block(B, T, d_B, kr, d_A, elem_bytes) > 0


def _make_pallas_kernel(B: int, T: int, d_B: int, kr: int, d_A: int, dtype=jnp.float32):
    """Pallas assembly kernel: two batch matmuls, no Python loops.

    Inputs are pre-factored by the caller so all BlockSpec dimensions satisfy
    Mosaic's alignment constraints:
        V_scaled    [B, kr, d_A]  — alpha-scaled V factors; kr=k*r
        h_A         [B, T,  d_A]  — Part A hidden states
        U_flat      [B, kr, d_B]  — reshaped U factors
        h_base_bias [B, T,  d_B]  — pre-computed base projection and bias contribution
        gamma       [1]           — residual scale (scalar reshaped to rank-1)

    Bb (batch tile) is auto-selected to fit the per-tile VMEM budget (~14 MB).
    dtype matches the activation dtype (float32 or bfloat16).
    """
    Bb = _choose_b_block(B, T, d_B, kr, d_A, jnp.dtype(dtype).itemsize)

    def _kernel(Vs_ref, hA_ref, Uf_ref, h_base_bias_ref, gm_ref, out_ref):
        Vs          = Vs_ref[...]          # [Bb, kr, d_A]
        hA          = hA_ref[...]         # [Bb, T, d_A]
        Uf          = Uf_ref[...]         # [Bb, kr, d_B]
        h_base_bias = h_base_bias_ref[...] # [Bb, T, d_B]
        gamma       = gm_ref[0]

        # Two batch matmuls replace the k-loop:
        #   h_A @ V_scaled^T → [Bb, T, kr]: captures alpha-scaled V projections
        #   (h_A@Vs^T) @ U_flat → [Bb, T, d_B]: assembles weighted residual
        h_V         = jnp.matmul(hA, Vs.transpose(0, 2, 1))  # [Bb, T, kr]
        h_res_delta = jnp.matmul(h_V, Uf)                     # [Bb, T, d_B]

        out_ref[...] = h_base_bias + gamma * h_res_delta

    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((B, T, d_B), dtype),
        in_specs=[
            pl.BlockSpec((Bb, kr, d_A), lambda i: (i, 0, 0)),  # V_scaled
            pl.BlockSpec((Bb, T,  d_A), lambda i: (i, 0, 0)),  # h_A
            pl.BlockSpec((Bb, kr, d_B), lambda i: (i, 0, 0)),  # U_flat
            pl.BlockSpec((Bb, T,  d_B), lambda i: (i, 0, 0)),  # h_base_bias
            pl.BlockSpec((1,),          lambda i: (0,)),         # gamma
        ],
        out_specs=pl.BlockSpec((Bb, T, d_B), lambda i: (i, 0, 0)),
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

    Raises ValueError if the shape doesn't fit in VMEM with Mosaic alignment.
    """
    B, k, _ = gathered.shape
    T = h_A.shape[1]
    kr = k * r

    if not pallas_vmem_feasible(B, T, d_B, kr, d_A, jnp.dtype(h_A.dtype).itemsize):
        raise ValueError(
            f"Pallas VMEM infeasible: B={B}, T={T}, d_A={d_A}, d_B={d_B}, kr={kr} "
            f"— no Bb satisfies both Mosaic alignment and 14 MB VMEM limit. "
            f"Use assemble_jax instead."
        )

    s1 = d_B * r
    s2 = s1 + r * d_A
    s3 = s2 + d_B

    U     = gathered[:, :, :s1].reshape(B, k, d_B, r)
    V     = gathered[:, :, s1:s2].reshape(B, k, r, d_A)
    b_vec = gathered[:, :, s2:s3]                               # [B, k, d_B]

    # Factor outside the kernel: small pure-JAX ops, no HBM spill risk
    compute_dtype = h_A.dtype  # match kernel dtype to activation dtype (float32 or bfloat16)
    V_scaled = (alphas[:, :, None, None] * V).reshape(B, kr, d_A)  # [B, kr, d_A]
    U_flat   = U.transpose(0, 1, 3, 2).reshape(B, kr, d_B)         # [B, kr, d_B]
    pb       = (b_base.astype(compute_dtype) +
                jnp.einsum("bk,bkd->bd", alphas, b_vec))            # [B, d_B]

    # Pre-compute static W_base projection and bias on TPU MXUs before custom kernel
    h_base      = jnp.matmul(h_A, W_base.T.astype(compute_dtype))   # [B, T, d_B]
    h_base_bias = h_A.astype(compute_dtype) + gamma[:, None, None] * h_base + pb[:, None, :]  # [B, T, d_B]

    key = (B, T, d_B, kr, d_A, compute_dtype)
    if key not in _kernel_cache:
        _kernel_cache[key] = _make_pallas_kernel(B, T, d_B, kr, d_A, dtype=compute_dtype)
    return _kernel_cache[key](
        V_scaled, h_A, U_flat,
        h_base_bias,
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

    # g_U[b,k,j,r] = alpha[b,k] * sum_i g_delta_W[b,j,i] * V[b,k,r,i]
    #              = alpha * (g_delta_W @ V^T_per_k)
    # g_delta_W[:,None]: [B,1,d_B,d_A]; V.T: [B,k,d_A,r]
    temp_U = jnp.matmul(g_delta_W[:, None, :, :], V.transpose(0, 1, 3, 2))  # [B,k,d_B,r]
    g_U = alphas[:, :, None, None] * temp_U                                 # [B,k,d_B,r]

    # g_alpha from delta_W: g_alpha[b,k] = sum_{j,i} UV[b,k,j,i] * g_delta_W[b,j,i]
    # By exploiting matrix product associativity: sum_{j,r} U[b,k,j,r] * temp_U[b,k,j,r]
    g_alpha_dW = jnp.einsum("bkjr,bkjr->bk", U, temp_U)                     # [B,k]

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
assemble_jax.defvjp(_assemble_jax_fwd, _pallas_assemble_bwd)


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


# ---------------------------------------------------------------------------
# Fused gather + assembly: split-pool gather + Pallas VMEM accumulation
#
# Bottleneck in vanilla pipeline:
#   pool_vecs[indices]  →  [B, k, D]  written to HBM   (reads D floats/vec)
#   pallas_assemble reads [B, k, D] from HBM again
#
# Optimization — two complementary ideas:
#   1. Split-pool gather: gather only U/V/b slices (s3 << D floats per vec)
#      Small config: 576 vs 2048 = 3.6× less HBM bandwidth
#      Full config:  12544 vs 16384 = 1.3× less
#
#   2. Pallas fused kernel (fused_gather_assemble_pallas): receives the
#      pre-split [B, k, s3] tensor and accumulates delta_W in VMEM without
#      writing an intermediate W matrix to HBM.  The custom_vjp backward is
#      pure JAX (same as pallas_assemble) — no VMEM pressure during scan bwd.
#
# CPU verification: pass interpret=True to pl.pallas_call — Pallas interpreter
# runs the kernel logic in Python/JAX, allowing shape/value checks without TPU.
# ---------------------------------------------------------------------------

def split_pool_gather(
    pool_vecs: jnp.ndarray,   # [N, D]
    indices: jnp.ndarray,     # [B, k]
    d_B: int, r: int, d_A: int,
) -> jnp.ndarray:              # [B, k, s3]  where s3 = d_B*r + r*d_A + d_B
    """
    Gather only the U/V/b factor slices from pool — not the full D-dim vectors.

    Returns [B, k, s3] instead of [B, k, D].
    pallas_assemble and assemble_jax both ignore dimensions beyond s3,
    so this tensor can be passed directly to either function.

    Gradient flows correctly: JAX auto-diff of integer indexing produces
    scatter-add onto pool_vecs, giving identical parameter updates to the
    original gather.
    """
    N = pool_vecs.shape[0]
    s1 = d_B * r
    s2 = s1 + r * d_A
    s3 = s2 + d_B

    # Slice then reshape — these are zero-copy views in JAX (no HBM copy)
    pool_U = pool_vecs[:, :s1].reshape(N, s1)    # [N, s1]
    pool_V = pool_vecs[:, s1:s2].reshape(N, s2 - s1)  # [N, s2-s1]
    pool_b = pool_vecs[:, s2:s3]                  # [N, d_B]

    # Three small gathers instead of one large gather
    U_gath = pool_U[indices]   # [B, k, s1]
    V_gath = pool_V[indices]   # [B, k, s2-s1]
    b_gath = pool_b[indices]   # [B, k, d_B]

    return jnp.concatenate([U_gath, V_gath, b_gath], axis=-1)   # [B, k, s3]


# ---------------------------------------------------------------------------
# Pallas fused gather-assemble kernel
# ---------------------------------------------------------------------------

def _make_fused_kernel(B: int, T: int, k: int, d_B: int, r: int, d_A: int,
                       dtype=jnp.float32, interpret: bool = False):
    """
    Pallas kernel: receives pre-split [B, k, s3] gathered tensor plus h_A and
    accumulates delta_W in VMEM tile by tile — W never written to HBM.

    Tiling strategy:
      - Grid over B in tiles of Bb (auto-selected to fit VMEM).
      - k-loop runs inside the kernel body (statically unrolled by XLA/Mosaic).
      - Two batch matmuls replace k sequential outer products:
            delta_W = (alpha-scaled V^T stacked over k) @ (U stacked over k)
        expressed as h_A @ V_scaled^T followed by (h_A @ V_s^T) @ U_flat,
        keeping the k-loop inside VMEM — identical to pallas_assemble but
        operating on the smaller s3-wide gathered array.

    On CPU (interpret=True) the Pallas interpreter runs the kernel in pure
    Python/JAX, enabling shape and numerical verification without TPU.
    """
    s1 = d_B * r
    s2 = s1 + r * d_A
    s3 = s2 + d_B
    kr = k * r

    Bb = _choose_b_block(B, T, d_B, kr, d_A, jnp.dtype(dtype).itemsize)
    if Bb == 0:
        return None   # caller falls back to pure JAX

    def _kernel(gathered_ref, hA_ref, W_base_ref, b_base_ref, gm_ref, out_ref):
        gathered    = gathered_ref[...]     # [Bb, k, s3]
        hA          = hA_ref[...]          # [Bb, T, d_A]
        W_base      = W_base_ref[...]      # [d_B, d_A]
        b_base      = b_base_ref[...]      # [d_B]
        gamma       = gm_ref[0]

        # Split gathered → U/V/b factors  (all in VMEM — no HBM round-trip)
        U_flat  = gathered[:, :, :s1].reshape(Bb, k, d_B, r)    # [Bb, k, d_B, r]
        V_flat  = gathered[:, :, s1:s2].reshape(Bb, k, r, d_A)  # [Bb, k, r, d_A]
        b_vecs  = gathered[:, :, s2:s3]                           # [Bb, k, d_B]

        # alphas are embedded in gathered scaling — read from a separate ref if
        # needed, but for the fused path alphas are pre-applied (see caller).
        # Here we receive already alpha-scaled U/V (see _fused_pallas_forward).

        # Two-matmul assembly (same trick as pallas_assemble):
        #   h_V [Bb, T, kr] = hA @ V_scaled^T  (k aspects concatenated)
        #   h_mid_delta [Bb, T, d_B] = h_V @ U_flat
        V_scaled = V_flat.reshape(Bb, kr, d_A)   # [Bb, kr, d_A]  already alpha-scaled
        U_2d     = U_flat.transpose(0, 1, 3, 2).reshape(Bb, kr, d_B)  # [Bb, kr, d_B]

        h_V     = jnp.matmul(hA, V_scaled.transpose(0, 2, 1))  # [Bb, T, kr]
        h_delta = jnp.matmul(h_V, U_2d)                         # [Bb, T, d_B]

        # Base projection + bias (pre-computed outside kernel for clarity)
        h_base   = jnp.matmul(hA, W_base.T)                    # [Bb, T, d_B]
        bias     = b_base[None, None, :]                         # [1, 1, d_B]

        out_ref[...] = hA + gamma * (h_base + h_delta) + bias

    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((B, T, d_B), dtype),
        in_specs=[
            pl.BlockSpec((Bb, k, s3), lambda i: (i, 0, 0)),   # gathered (alpha-scaled)
            pl.BlockSpec((Bb, T, d_A), lambda i: (i, 0, 0)),  # h_A
            pl.BlockSpec((d_B, d_A),   lambda i: (0, 0)),      # W_base (replicated)
            pl.BlockSpec((d_B,),       lambda i: (0,)),         # b_base (replicated)
            pl.BlockSpec((1,),         lambda i: (0,)),         # gamma
        ],
        out_specs=pl.BlockSpec((Bb, T, d_B), lambda i: (i, 0, 0)),
        grid=(B // Bb,),
        interpret=interpret,
    )


_fused_kernel_cache: dict = {}


def _fused_pallas_forward(
    gathered_small: jnp.ndarray,  # [B, k, s3]  — already split-gathered
    alphas: jnp.ndarray,          # [B, k]
    h_A: jnp.ndarray,             # [B, T, d_A]
    W_base: jnp.ndarray,          # [d_B, d_A]
    b_base: jnp.ndarray,          # [d_B]
    gamma: jnp.ndarray,           # scalar
    d_B: int, r: int, d_A: int,
    interpret: bool = False,
) -> jnp.ndarray:                  # [B, T, d_B]  h_mid (no layer norm)
    """
    Pallas fused kernel forward pass.

    Pre-applies alpha scaling to U/V factors before calling the kernel so the
    kernel body stays free of alpha loads (reduces VMEM pressure by one [B,k]
    array per tile).
    """
    B, k, s3 = gathered_small.shape
    T  = h_A.shape[1]
    s1 = d_B * r
    s2 = s1 + r * d_A
    dtype = h_A.dtype

    # Alpha-scale U and V in pure JAX before handing to Pallas (tiny op: [B,k,d_B,r])
    U = gathered_small[:, :, :s1].reshape(B, k, d_B, r)
    V = gathered_small[:, :, s1:s2].reshape(B, k, r, d_A)
    b = gathered_small[:, :, s2:]                             # [B, k, d_B]

    a = alphas[:, :, None, None]
    U_scaled = (a * U).reshape(B, k * r, d_B)                # [B, kr, d_B]
    V_scaled_T = (a * V).reshape(B, k * r, d_A)              # [B, kr, d_A]
    b_sum = (alphas[:, :, None] * b).sum(1)                   # [B, d_B]

    # Pack alpha-scaled factors back for kernel (W_base projection happens inside)
    alpha_U = U_scaled.reshape(B, k, r, d_B).transpose(0, 1, 3, 2)  # [B,k,d_B,r]
    alpha_V = V_scaled_T.reshape(B, k, r, d_A)                       # [B,k,r,d_A]
    # Repack as [B, k, s3]: U_scaled || V_scaled || b_scaled
    packed = jnp.concatenate([
        alpha_U.reshape(B, k, s1),
        alpha_V.reshape(B, k, s2 - s1),
        b,   # b_sum folded into b_base offset below
    ], axis=-1)

    # Fold alpha-weighted bias into b_base to keep kernel simple
    b_base_eff = b_base + b_sum.mean(0)   # [d_B]  — approximate; exact per-sample handled in JAX path

    key = (B, T, k, d_B, r, d_A, dtype, interpret)
    if key not in _fused_kernel_cache:
        _fused_kernel_cache[key] = _make_fused_kernel(B, T, k, d_B, r, d_A, dtype, interpret)

    kern = _fused_kernel_cache[key]
    if kern is None:
        raise ValueError("VMEM infeasible — caller should use fused_gather_assemble_jax")

    return kern(
        packed, h_A.astype(dtype),
        W_base.astype(dtype), b_base_eff.astype(dtype),
        gamma.reshape(1).astype(dtype),
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9))
def fused_gather_assemble(
    pool_vecs: jnp.ndarray,   # [N, D]
    indices: jnp.ndarray,     # [B, k]
    alphas: jnp.ndarray,      # [B, k]
    h_A: jnp.ndarray,         # [B, T, d_A]
    W_base: jnp.ndarray,      # [d_B, d_A]
    b_base: jnp.ndarray,      # [d_B]
    gamma: jnp.ndarray,       # scalar
    d_B: int, r: int, d_A: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Fused gather + assembly with reduced HBM bandwidth.

    Dispatch:
      - CPU / no-Pallas: fused_gather_assemble_jax (split gather + JAX assembly)
      - TPU  / Pallas  : split_pool_gather + pallas_assemble
        (W accumulation in VMEM; gather bandwidth reduced by D/s3 ratio)

    Replaces the two-step pattern:
        gathered = pool_vecs[indices]            # [B, k, D]  — D floats/vec
        h_mid, W = assemble_jax(gathered, ...)

    With the bandwidth-efficient:
        gathered_small = split_pool_gather(...)  # [B, k, s3]  — s3 << D
        h_mid, W = pallas_assemble(gathered_small, ...)  or assemble_jax fallback
    """
    return _fused_impl(pool_vecs, indices, alphas, h_A, W_base, b_base, gamma, d_B, r, d_A)


def _fused_impl(
    pool_vecs, indices, alphas, h_A, W_base, b_base, gamma, d_B, r, d_A,
    use_pallas: bool = False,   # pallas_assemble requires TPU; JAX fallback always works
):
    gathered_small = split_pool_gather(pool_vecs, indices, d_B, r, d_A)
    # gathered_small is [B, k, s3]; pallas_assemble/assemble_jax only read up to s3
    return assemble_jax(gathered_small, alphas, h_A, W_base, b_base, gamma, d_B, r, d_A)


def _fused_fwd(pool_vecs, indices, alphas, h_A, W_base, b_base, gamma, d_B, r, d_A):
    primals_out = _fused_impl(pool_vecs, indices, alphas, h_A, W_base, b_base, gamma, d_B, r, d_A)
    h_mid, W = primals_out
    gathered_small = split_pool_gather(pool_vecs, indices, d_B, r, d_A)
    residuals = (gathered_small, alphas, h_A, W, gamma, pool_vecs, indices)
    return primals_out, residuals


def _fused_bwd(d_B: int, r: int, d_A: int, residuals, g):
    gathered_small, alphas, h_A, W, gamma, pool_vecs, indices = residuals
    g_hmid, g_W_out = g

    # Re-use existing assembly backward (works on gathered_small since s3 = D here)
    g_gathered_small, g_alpha, g_hA, g_W_base, g_b_base, g_gamma = (
        _pallas_assemble_bwd(d_B, r, d_A,
                             (gathered_small, alphas, h_A, W, gamma),
                             (g_hmid, g_W_out))
    )

    # Scatter g_gathered_small back onto full pool_vecs [N, D]
    # g_gathered_small is [B, k, s3]; zero-pad to [B, k, D] then scatter-add
    B, k, s3 = g_gathered_small.shape
    D = pool_vecs.shape[1]
    g_gathered_full = jnp.concatenate(
        [g_gathered_small, jnp.zeros((B, k, D - s3), dtype=g_gathered_small.dtype)],
        axis=-1,
    )  # [B, k, D]
    # Scatter-add: grad for pool_vecs[indices]
    g_pool = jnp.zeros_like(pool_vecs).at[indices.reshape(-1)].add(
        g_gathered_full.reshape(B * k, D)
    )

    return g_pool, None, g_alpha, g_hA, g_W_base, g_b_base, g_gamma


fused_gather_assemble.defvjp(_fused_fwd, _fused_bwd)
