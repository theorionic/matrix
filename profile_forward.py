"""
Profile each DWA forward-pass component individually.

Runs each part (embed, Part A, retrieval, assembly, Part B, lm_head)
as a separate JIT-compiled function and times with block_until_ready().
This gives real wall-clock breakdown rather than theory.

Usage:
    python profile_forward.py
"""

import time

# Patch JAX config before any import (fixes optax/JAX version mismatch)
import jax._src.config as _jax_cfg
_orig_update = _jax_cfg.config.update
def _safe_update(name, val):
    try:
        _orig_update(name, val)
    except AttributeError:
        pass
_jax_cfg.config.update = _safe_update

import jax
import jax.numpy as jnp
from flax import nnx

from src.dwa.model import DWAModel
from src.dwa.parts import precompute_rope_freqs
from src.dwa.assembly_pallas import assemble_jax, compute_key_cache
from src.dwa.run_config import load_config

# ── load config ──────────────────────────────────────────────────────────────
import sys
cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/large.yaml"
run_cfg = load_config(cfg_path)
cfg  = run_cfg.model
tcfg = run_cfg.train

print(f"Config: {cfg_path}")
print(f"  B={tcfg.batch_size}  T={cfg.seq_len}  d_A={cfg.d_A}  n_heads={cfg.n_heads}  "
      f"d_head={cfg.d_A//cfg.n_heads}  N={cfg.N}  D={cfg.D}")
print(f"  n_layers_A={cfg.n_layers_A}  n_layers_B={cfg.n_layers_B}")
print()

B = tcfg.batch_size
T = cfg.seq_len

# ── build model ───────────────────────────────────────────────────────────────
rngs = nnx.Rngs(params=0, dropout=1)
model = DWAModel(cfg, rngs)

# Dummy inputs
key = jax.random.PRNGKey(0)
input_ids = jax.random.randint(key, (B, T), 0, cfg.vocab_size)
cos, sin = precompute_rope_freqs(T, cfg.d_A // cfg.n_heads)


def _time(name, fn, *args, n_warmup=3, n_runs=5, flops=0):
    """JIT-compile, warm up, then time with block_until_ready."""
    jit_fn = jax.jit(fn)
    # warmup — forces compilation
    for _ in range(n_warmup):
        out = jit_fn(*args)
        jax.block_until_ready(out)
    # timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = jit_fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    ms = min(times) * 1e3          # best-case (eliminates scheduling noise)
    ms_avg = sum(times) / len(times) * 1e3
    tflops = (flops / 1e12) / (min(times)) if flops else 0
    pct = tflops / 197.0 * 100 if tflops else 0
    print(f"  {name:<28s}  best={ms:7.1f} ms  avg={ms_avg:7.1f} ms"
          + (f"  {tflops:5.1f} TFLOP/s  {pct:4.1f}% MFU" if flops else ""))
    return out


# ── 1. Embedding ──────────────────────────────────────────────────────────────
print("=== Forward pass components (forward only, no backward) ===")
print()

embed_fn = lambda ids: model.embed(ids)
x = _time("Embed", embed_fn, input_ids,
          flops=2 * B * T * cfg.vocab_size * cfg.d_A)  # embedding lookup is gather; tiny

# ── 2. Part A ─────────────────────────────────────────────────────────────────
x_val = model.embed(input_ids)
x_val = jax.block_until_ready(x_val)

part_a_flops = (
    B * T * ((4 + 2 * cfg.ffn_mult) * cfg.d_A ** 2 + 2 * T * cfg.d_A)
    * cfg.n_layers_A
)
h_A = _time("Part A", lambda x: model.part_a(x, cos, sin, None), x_val,
            flops=part_a_flops)

# ── 3. Key cache (pool→keys projection, done once per window) ─────────────────
pool_vecs = model.pool.vectors[...].astype(jnp.float32)
key_proj  = model.pool.key_proj[...].astype(jnp.float32)
key_cache_flops = 2 * cfg.N * cfg.D * cfg.d_k * cfg.S  # [N,D] x [D,d_k] for S aspects
_time("Key cache (per-window)", compute_key_cache, pool_vecs, key_proj,
      flops=key_cache_flops)

pool_keys = jax.jit(compute_key_cache)(pool_vecs, key_proj)
pool_keys = jax.block_until_ready(pool_keys)

# ── 4. Retrieval ──────────────────────────────────────────────────────────────
z = jax.block_until_ready(h_A).mean(axis=1)  # [B, d_A]
retrieval_flops = cfg.S * B * cfg.N * cfg.d_k  # cosine sim [B,d_k] x [N,d_k] for each aspect
alphas, indices, soft_full = _time(
    "Retrieval",
    lambda _z, _pk: model.retrieval(_z, _pk, 1.0, True, mesh=None),
    z, pool_keys,
    flops=retrieval_flops,
)

# ── 5. Gather pool vectors ────────────────────────────────────────────────────
gathered = _time(
    "Pool gather [B,k,D]",
    lambda idx: model.pool.vectors[...][idx],
    indices,
)

# ── 6. Assembly (pure JAX — Pallas needs mesh, skip for profiling) ────────────
gathered_f32 = jax.block_until_ready(gathered).astype(jnp.float32)
W_base = model.assembler.W_base[...]
b_base = model.assembler.b_base[...]
gamma  = model.assembler.gamma[...]

from src.dwa.assembly_pallas import assemble_jax
assembly_flops = (
    B * cfg.k_max * 2 * cfg.r * cfg.d_A ** 2   # k low-rank outer products
    + B * T * cfg.d_A ** 2                       # apply W to h_A
)
h_mid_no_ln, W = _time(
    "Assembly (JAX)",
    lambda g, a, h: assemble_jax(g, a, h, W_base, b_base, gamma, cfg.d_B, cfg.r, cfg.d_A),
    gathered_f32, alphas, jax.block_until_ready(h_A),
    flops=assembly_flops,
)

h_mid = jax.jit(model.assembler.layer_norm)(jax.block_until_ready(h_mid_no_ln))
h_mid = jax.block_until_ready(h_mid)

# ── 7. Part B ─────────────────────────────────────────────────────────────────
part_b_flops = (
    B * T * ((4 + 2 * cfg.ffn_mult) * cfg.d_B ** 2 + 2 * T * cfg.d_B)
    * cfg.n_layers_B
)
h_out = _time("Part B", lambda h: model.part_b(h, cos, sin, None), h_mid,
              flops=part_b_flops)

# ── 8. LM head ────────────────────────────────────────────────────────────────
h_out_val = jax.block_until_ready(h_out)
lm_head_flops = 2 * B * T * cfg.d_B * cfg.vocab_size
logits = _time("LM head [B,T,V]", model.lm_head, h_out_val,
               flops=lm_head_flops)

# ── 9. Full forward (end-to-end, forward only) ────────────────────────────────
print()
total_fwd_flops = part_a_flops + retrieval_flops + assembly_flops + part_b_flops + lm_head_flops

@jax.jit
def full_forward(ids):
    x = model.embed(ids)
    h_A = model.part_a(x, cos, sin, None)
    z   = h_A.mean(axis=1)
    alphas, indices, _ = model.retrieval(z, pool_keys, 1.0, True, mesh=None)
    g = model.pool.vectors[...][indices].astype(jnp.float32)
    h_mn, W = assemble_jax(g, alphas, h_A, W_base, b_base, gamma, cfg.d_B, cfg.r, cfg.d_A)
    h_mid = model.assembler.layer_norm(h_mn)
    h_out = model.part_b(h_mid, cos, sin, None)
    return model.lm_head(h_out)

_time("=== Full forward (no backward)", full_forward, input_ids,
      n_warmup=3, n_runs=5, flops=total_fwd_flops)

# ── 10. Full forward+backward (matches training) ──────────────────────────────
from src.dwa.losses import task_loss
from src.dwa.model import forward_and_loss

@jax.jit
def fwd_bwd(ids):
    def loss_fn(m):
        return forward_and_loss(
            m, ids, 1.0, True, tcfg, False,
            key_cache=pool_keys, use_pallas=False, mesh=None,
        )
    (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    return loss

total_step_flops = 3 * total_fwd_flops  # fwd + bwd ≈ 3x fwd
_time("=== Full fwd+bwd (training step)", fwd_bwd, input_ids,
      n_warmup=2, n_runs=3, flops=total_step_flops)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("FLOPs breakdown (forward only):")
components = {
    "Part A": part_a_flops,
    "Retrieval": retrieval_flops,
    "Assembly": assembly_flops,
    "Part B": part_b_flops,
    "LM head": lm_head_flops,
}
for name, f in components.items():
    print(f"  {name:<12s}  {f/1e9:8.1f} GFLOPs  {f/total_fwd_flops*100:5.1f}%")
print(f"  {'Total':<12s}  {total_fwd_flops/1e9:8.1f} GFLOPs")
