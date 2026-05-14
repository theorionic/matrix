"""
DWA training entry point — 8-device TPU v5e data-parallel training.

Training strategy:
  - Mesh(8 devices, ('batch',)) with GSPMD data parallelism
  - nnx.jit wraps a jax.lax.scan window of `steps_per_window` steps
  - All gradient sync happens on-device inside the scan (no CPU round-trips)
  - Three-phase schedule: warmup → gate_on → sharpen
  - Per-component LRs via optax.masked chain

Run:
    python train.py                      # small config (fast sanity check)
    python train.py --full               # full-scale config
    python train.py --steps 50000        # custom step count
"""

from __future__ import annotations

# Patch JAX config BEFORE any other import (fixes optax/JAX version mismatch)
import jax._src.config as _jax_cfg

_orig_update = _jax_cfg.config.update


def _safe_update(name: str, val) -> None:
    try:
        _orig_update(name, val)
    except AttributeError:
        pass


_jax_cfg.config.update = _safe_update

import argparse
import functools
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from src.dwa.assembly_pallas import compute_key_cache
from src.dwa.config import DWAConfig, TrainConfig
from src.dwa.model import DWAModel, forward_and_loss
from src.dwa.schedule import PhaseScheduler
from src.dwa.utils import ema_update


# ---------------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------------

def _build_optimizer(model: DWAModel, tcfg: TrainConfig) -> nnx.Optimizer:
    """
    Per-component learning rates via chained masked Adam optimizers.

    Labels are derived from the pure (un-NNX-wrapped) parameter structure
    so they are compatible with what nnx.Optimizer passes to optax internally.
    """
    pure_params = nnx.as_pure(nnx.state(model, nnx.Param))
    leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(pure_params)

    def _label(path: tuple) -> str:
        p = "/".join(str(k) for k in path)
        if "pool" in p:
            return "pool"
        if "tau" in p or "gamma" in p:
            return "threshold"
        if "W_Q" in p or "aspect_weights" in p:
            return "retrieval"
        return "parts"

    def _make_mask(target_label: str) -> object:
        flags = [_label(path) == target_label for path, _ in leaves_with_paths]
        return jax.tree_util.tree_unflatten(treedef, flags)

    tx = optax.chain(
        optax.masked(optax.adam(tcfg.lr_pool), _make_mask("pool")),
        optax.masked(optax.adam(tcfg.lr_threshold), _make_mask("threshold")),
        optax.masked(optax.adam(tcfg.lr_retrieval), _make_mask("retrieval")),
        optax.masked(optax.adam(tcfg.lr_parts), _make_mask("parts")),
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


# ---------------------------------------------------------------------------
# Training window (compiled with nnx.jit; runs steps_per_window steps on TPU)
# ---------------------------------------------------------------------------

def _make_train_window(tcfg: TrainConfig, is_warmup: bool, aux_on: bool,
                       use_pallas: bool = True, mesh=None):
    """
    Returns a compiled function that runs steps_per_window training steps
    inside a single jax.lax.scan call.

    Optimizations:
    - key_cache: the [S,N,d_k] key projection is computed once per window
      (outside the scan), not per step.  Reduces per-step HBM reads from
      the 2 GB pool to a 32 MB cache read.
    - use_pallas: the assembly stage runs in a Pallas kernel via shard_map
      (one kernel per device, each sees its local batch slice).

    is_warmup, aux_on, use_pallas, and mesh are static closures.
    """

    @functools.partial(nnx.jit, static_argnames={})
    def train_window(
        model: DWAModel,
        optimizer: nnx.Optimizer,
        data: jnp.ndarray,           # [steps_per_window, B, seq_len]
        lambda_vals: jnp.ndarray,    # [steps_per_window]
        pool_ema_in: jnp.ndarray,    # [N]
        ema_decay: float,
    ) -> tuple[DWAModel, nnx.Optimizer, jnp.ndarray, dict]:

        # --- KEY CACHE: compute once here, reuse for all steps in this window ---
        key_cache = compute_key_cache(
            model.pool.vectors[...].astype(jnp.float32),
            model.pool.key_proj[...].astype(jnp.float32),
        )  # [S, N, d_k] — stays in HBM across the scan

        def step_fn(carry, xs):
            model, optimizer, pool_ema, step_in_window = carry
            batch, lam = xs  # batch: [B, seq_len], lam: scalar

            def loss_fn(m):
                return forward_and_loss(
                    m, batch, lam, is_warmup, tcfg, aux_on,
                    key_cache=key_cache, use_pallas=use_pallas, mesh=mesh,
                )

            (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
            optimizer.update(model, grads)

            # Update pool utilisation EMA
            alphas = info["alphas"]      # [B, k_max]
            indices = info["indices"]    # [B, k_max]
            usage = jnp.zeros(model.cfg.N).at[indices.reshape(-1)].add(
                alphas.reshape(-1) / (batch.shape[0] + 1e-8)
            )
            pool_ema = ema_update(pool_ema, usage, ema_decay)

            return (model, optimizer, pool_ema, step_in_window + 1), loss

        init_carry = (model, optimizer, pool_ema_in, jnp.array(0))
        (model, optimizer, pool_ema_out, _), losses = jax.lax.scan(
            step_fn, init_carry, (data, lambda_vals)
        )
        return model, optimizer, pool_ema_out, {"losses": losses}

    return train_window


# ---------------------------------------------------------------------------
# Data helpers (synthetic for now — replace with real data loader)
# ---------------------------------------------------------------------------

def _synthetic_batch(
    rng: jax.Array, batch_size: int, seq_len: int, vocab_size: int
) -> jnp.ndarray:
    return jax.random.randint(rng, (batch_size, seq_len), 0, vocab_size)


def _synthetic_window(
    rng: jax.Array, steps: int, batch_size: int, seq_len: int, vocab_size: int
) -> jnp.ndarray:
    """Returns [steps, B, seq_len] of random token IDs."""
    keys = jax.random.split(rng, steps)
    batches = jax.vmap(
        lambda k: jax.random.randint(k, (batch_size, seq_len), 0, vocab_size)
    )(keys)
    return batches


# ---------------------------------------------------------------------------
# Pallas probe — check at startup whether the kernel compiles successfully
# ---------------------------------------------------------------------------

def _probe_pallas(cfg: DWAConfig, mesh: Mesh) -> bool:
    """
    Try to compile the Pallas assembly kernel via shard_map.
    Each device sees one example; passes if kernel compiles cleanly.
    """
    from src.dwa.assembly_pallas import shard_pallas_assemble
    try:
        n_dev = len(mesh.devices)
        B_probe = n_dev  # one item per device
        gathered = jax.device_put(
            jnp.zeros((B_probe, cfg.k_max, cfg.D)),
            NamedSharding(mesh, P("batch", None, None)),
        )
        alphas = jax.device_put(
            jnp.ones((B_probe, cfg.k_max)) / cfg.k_max,
            NamedSharding(mesh, P("batch", None)),
        )
        h_A = jax.device_put(
            jnp.zeros((B_probe, cfg.seq_len, cfg.d_A)),
            NamedSharding(mesh, P("batch", None, None)),
        )
        W_base = jnp.zeros((cfg.d_B, cfg.d_A))
        b_base = jnp.zeros(cfg.d_B)
        gamma  = jnp.array(cfg.gamma_init)
        @jax.jit
        def _probe_fn(g, a, h, wb, bb, gm):
            return shard_pallas_assemble(g, a, h, wb, bb, gm,
                                         cfg.d_B, cfg.r, cfg.d_A, mesh)
        result = _probe_fn(gathered, alphas, h_A, W_base, b_base, gamma)
        jax.block_until_ready(result)
        return True
    except Exception as e:
        print(f"[DWA] Pallas probe failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: DWAConfig, tcfg: TrainConfig) -> None:
    devices = jax.devices()
    n_devices = len(devices)
    print(f"[DWA] Training on {n_devices} TPU devices")
    print(f"[DWA] Model config: N={cfg.N}, D={cfg.D}, d_A={cfg.d_A}, r={cfg.r}")

    # Mesh for data-parallel sharding (batch split across all devices)
    mesh = Mesh(devices, ("batch",))
    batch_sharding = NamedSharding(mesh, P("batch"))

    # Per-device batch size
    assert tcfg.batch_size % n_devices == 0, "batch_size must be divisible by n_devices"
    local_batch = tcfg.batch_size // n_devices

    scheduler = PhaseScheduler(tcfg)
    lambda_array = scheduler.make_lambda_array()     # [total_steps]

    # Initialise model
    rng = jax.random.PRNGKey(tcfg.seed)
    model = DWAModel(cfg, nnx.Rngs(rng))
    optimizer = _build_optimizer(model, tcfg)
    pool_ema = jnp.zeros(cfg.N)

    # Count parameters
    n_params = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"[DWA] Total trainable params: {n_params / 1e6:.1f}M")

    # Pallas + shard_map is verified correct but hits a TPU VMEM constraint
    # when compiled inside jax.lax.scan + value_and_grad (the scan backward
    # JVP context has a 16MB scoped VMEM limit that the kernel exceeds).
    # Training uses pure-JAX; Pallas remains available for inference.
    _use_pallas = False
    print(f"[DWA] Pallas assembly: disabled for training (VMEM constraint in scan+vjp); available for inference")

    # Pre-JIT train windows for each phase
    # (re-compilation happens at phase boundaries, not per step)
    compiled_fns: dict[tuple, object] = {}

    def get_train_fn(is_warmup: bool, aux_on: bool):
        key = (is_warmup, aux_on)
        if key not in compiled_fns:
            compiled_fns[key] = _make_train_window(
                tcfg, is_warmup, aux_on,
                use_pallas=_use_pallas,
                mesh=mesh if _use_pallas else None,
            )
        return compiled_fns[key]

    n_windows = tcfg.total_steps // tcfg.steps_per_window
    steps_done = 0
    rng = jax.random.PRNGKey(tcfg.seed + 1)

    print(f"[DWA] Training for {tcfg.total_steps} steps "
          f"({n_windows} windows × {tcfg.steps_per_window} steps)")

    t0 = time.time()
    for window_idx in range(n_windows):
        start_step = window_idx * tcfg.steps_per_window
        phase = scheduler.get_phase(start_step)
        is_warmup = scheduler.is_warmup(start_step)
        aux_on = scheduler.aux_enabled(start_step)

        # Slice lambda schedule for this window
        lam_window = lambda_array[start_step: start_step + tcfg.steps_per_window]

        # Synthetic data window [steps, B, seq]
        rng, data_rng = jax.random.split(rng)
        data_window = _synthetic_window(
            data_rng,
            tcfg.steps_per_window,
            tcfg.batch_size,
            cfg.seq_len,
            cfg.vocab_size,
        )  # [steps, B, seq_len]

        # Shard data: batch dim across devices
        # data_window: [steps, B, seq_len] — shard along batch axis (dim 1)
        data_sharded = jax.device_put(data_window, NamedSharding(mesh, P(None, "batch", None)))

        train_fn = get_train_fn(is_warmup, aux_on)
        t_win = time.time()
        model, optimizer, pool_ema, info = train_fn(
            model, optimizer, data_sharded, lam_window, pool_ema, tcfg.ema_decay
        )
        # Block until TPU computation finishes before timing
        jax.block_until_ready(info["losses"])
        win_secs = time.time() - t_win

        steps_done += tcfg.steps_per_window
        elapsed = time.time() - t0
        mean_loss = float(info["losses"].mean())
        win_steps_per_sec = tcfg.steps_per_window / win_secs
        win_tok_per_sec = int(win_steps_per_sec * tcfg.batch_size * cfg.seq_len)

        print(
            f"[DWA] step={steps_done:6d}/{tcfg.total_steps} "
            f"phase={phase:8s} λ={scheduler.get_lambda(start_step):.2f} "
            f"loss={mean_loss:.4f} "
            f"win={win_secs:.1f}s  steps/s={win_steps_per_sec:.1f}  tok/s={win_tok_per_sec:,}"
        )

        # Update model's pool EMA (non-trainable variable)
        model.pool_ema[...] = pool_ema

    print(f"[DWA] Training complete in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train DWA model")
    parser.add_argument("--full",   action="store_true", help="Full-scale config (~13GB/device)")
    parser.add_argument("--medium", action="store_true", help="Medium config (fits 16GB/device)")
    parser.add_argument("--mxu",    action="store_true", help="MXU-aligned config: r=128, 128×128 assembly matmuls")
    parser.add_argument("--bf16",   action="store_true", help="Store pool in bfloat16 (halves gather bandwidth)")
    parser.add_argument("--steps", type=int, default=None, help="Override total_steps")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps-per-window", type=int, default=None)
    args = parser.parse_args()

    if args.full:
        cfg = DWAConfig()
    elif args.mxu:
        cfg = DWAConfig.medium_mxu(bf16=args.bf16)
    elif args.medium:
        from src.dwa.config import DWAConfig as _DWA
        cfg = _DWA.medium()
        if args.bf16:
            cfg.bf16_pool = True
    else:
        cfg = DWAConfig.small()

    tcfg = TrainConfig()

    if args.full:
        pass
    elif args.mxu or args.medium:
        tcfg.total_steps = tcfg.total_steps // 5
        tcfg.warmup_steps = tcfg.warmup_steps // 5
        tcfg.gate_on_steps = tcfg.gate_on_steps // 5
        tcfg.batch_size = 128
    else:
        tcfg.total_steps = tcfg.total_steps // 10
        tcfg.warmup_steps = tcfg.warmup_steps // 10
        tcfg.gate_on_steps = tcfg.gate_on_steps // 10
        tcfg.batch_size = 16

    if args.steps is not None:
        tcfg.total_steps = args.steps
    if args.batch_size is not None:
        tcfg.batch_size = args.batch_size
    if args.steps_per_window is not None:
        tcfg.steps_per_window = args.steps_per_window

    train(cfg, tcfg)


if __name__ == "__main__":
    main()
