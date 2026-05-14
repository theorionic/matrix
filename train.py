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

def _make_train_window(tcfg: TrainConfig, is_warmup: bool, aux_on: bool):
    """
    Returns a compiled function that runs steps_per_window training steps
    inside a single jax.lax.scan call.

    is_warmup and aux_on are static (they change only ~twice over training).
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

        def step_fn(carry, xs):
            model, optimizer, pool_ema, step_in_window = carry
            batch, lam = xs  # batch: [B, seq_len], lam: scalar

            def loss_fn(m):
                return forward_and_loss(m, batch, lam, is_warmup, tcfg, aux_on)

            (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
            optimizer.update(model, grads)

            # Update pool utilisation EMA using the current batch's alphas
            alphas = info["alphas"]      # [B, k_max]
            indices = info["indices"]    # [B, k_max]
            # Scatter mean alpha per vector (coarse but sufficient for L_util)
            usage = jnp.zeros(model.cfg.N).at[indices.reshape(-1)].add(
                alphas.reshape(-1) / (batch.shape[0] + 1e-8)
            )
            pool_ema = ema_update(pool_ema, usage, ema_decay)

            jax.debug.print(
                "  step_in_window={s} loss={l:.4f} l_task={lt:.4f} l_aux={la:.4f}",
                s=step_in_window, l=loss,
                lt=info["l_task"], la=info["total_aux"],
                ordered=False,
            )
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

    # Pre-JIT train windows for each phase
    # (re-compilation happens at phase boundaries, not per step)
    compiled_fns: dict[tuple, object] = {}

    def get_train_fn(is_warmup: bool, aux_on: bool):
        key = (is_warmup, aux_on)
        if key not in compiled_fns:
            compiled_fns[key] = _make_train_window(tcfg, is_warmup, aux_on)
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
        model, optimizer, pool_ema, info = train_fn(
            model, optimizer, data_sharded, lam_window, pool_ema, tcfg.ema_decay
        )

        steps_done += tcfg.steps_per_window
        elapsed = time.time() - t0
        steps_per_sec = steps_done / elapsed
        mean_loss = float(info["losses"].mean())

        print(
            f"[DWA] step={steps_done:6d}/{tcfg.total_steps} "
            f"phase={phase:8s} λ={scheduler.get_lambda(start_step):.2f} "
            f"loss={mean_loss:.4f} "
            f"steps/s={steps_per_sec:.1f}"
        )

        # Update model's pool EMA (non-trainable variable)
        model.pool_ema[...] = pool_ema

    print(f"[DWA] Training complete in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train DWA model")
    parser.add_argument("--full", action="store_true", help="Use full-scale config")
    parser.add_argument("--steps", type=int, default=None, help="Override total_steps")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps-per-window", type=int, default=None)
    args = parser.parse_args()

    cfg = DWAConfig() if args.full else DWAConfig.small()
    tcfg = TrainConfig()

    if not args.full:
        # Small config: shrink training too
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
