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

import numpy as np

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
import os
import time
from collections import deque
from typing import NamedTuple

import orbax.checkpoint as ocp

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from src.dwa.assembly_pallas import compute_key_cache
from src.dwa.config import DWAConfig, TrainConfig
from src.dwa.model import DWAModel, forward_and_loss
from src.dwa.run_config import (
    CheckpointConfig, DataConfig, RunConfig, ShardingConfig,
    load_config, save_config,
)
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
        if "W_Q" in p or "aspect_weights" in p or "centroids" in p:
            return "retrieval"
        return "parts"

    def _make_mask(target_label: str) -> object:
        flags = [_label(path) == target_label for path, _ in leaves_with_paths]
        return jax.tree_util.tree_unflatten(treedef, flags)

    tx = optax.chain(
        optax.clip_by_global_norm(tcfg.grad_clip_norm),   # must be first
        optax.masked(optax.adam(tcfg.lr_pool), _make_mask("pool")),
        optax.masked(optax.adam(tcfg.lr_threshold), _make_mask("threshold")),
        optax.masked(optax.adam(tcfg.lr_retrieval), _make_mask("retrieval")),
        optax.masked(optax.adam(tcfg.lr_parts), _make_mask("parts")),
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


# ---------------------------------------------------------------------------
# Multi-device mesh helpers
# ---------------------------------------------------------------------------

def _select_n_model(cfg: DWAConfig, n_devices: int, override: int | str = "auto") -> int:
    """
    Choose model-parallel sharding degree so pool+Adam fits in ~4 GB/device.

    override: "auto" → automatic; int → use that value (must divide n_devices).
    """
    if override != "auto":
        n = int(override)
        assert n_devices % n == 0, f"n_model={n} must divide n_devices={n_devices}"
        return n
    pool_and_adam_bytes = cfg.N * cfg.D * 4 * 3  # float32 params + m + v
    target_bytes = 4 * 1024 ** 3  # 4 GB threshold
    for n_model in [1, 2, 4, 8]:
        if n_model > n_devices:
            break
        if n_devices % n_model != 0:
            continue
        if pool_and_adam_bytes / n_model <= target_bytes:
            return n_model
    return n_devices


def _build_mesh(devices, n_model: int) -> "Mesh":
    """
    2D Mesh with axes ('data', 'model').

    n_data = n_devices // n_model devices replicate the batch;
    n_model devices share the pool parameter matrix.
    """
    n_devices = len(devices)
    n_data = n_devices // n_model
    devices_2d = np.array(devices).reshape(n_data, n_model)
    return Mesh(devices_2d, ("data", "model"))


def _make_sharded_pool_vectors(cfg: "DWAConfig", mesh: "Mesh", rng) -> "jnp.ndarray | None":
    """
    Create pool vectors directly on each device's HBM in sharded form.

    Returns a globally-sharded [N, D] JAX array with P('model', None) sharding,
    or None if n_model == 1 (no model parallelism needed).

    Each device generates only its local [N_local, D] shard — no device ever
    allocates the full pool, avoiding the OOM that would occur if we initialized
    everything on device 0 and tried to shard afterwards.
    """
    n_model = mesh.shape.get("model", 1)
    if n_model <= 1:
        return None

    N_local = cfg.N // n_model
    pool_dtype = jnp.bfloat16 if cfg.bf16_pool else jnp.float32
    pool_sharding = NamedSharding(mesh, P("model", None))
    global_shape = (cfg.N, cfg.D)

    per_device_arrays = []
    idx_map = pool_sharding.addressable_devices_indices_map(global_shape)
    for device in pool_sharding.addressable_devices:
        idx_tuple = idx_map[device]
        row_start = idx_tuple[0].start or 0
        m_idx = row_start // N_local          # which model shard this device owns
        # Compute the shard directly on the target device — no cross-device traffic
        with jax.default_device(device):
            rng_dev = jax.device_put(rng, device)
            key = jax.random.fold_in(rng_dev, m_idx)
            shard = (jax.random.normal(key, (N_local, cfg.D)) * 0.02).astype(pool_dtype)
        per_device_arrays.append(shard)

    return jax.make_array_from_single_device_arrays(
        global_shape, pool_sharding, per_device_arrays
    )


# ---------------------------------------------------------------------------
# MFU helpers
# ---------------------------------------------------------------------------

PATTERN_PERIOD = 4  # period for repeat-pattern learning verification


def compute_step_flops(cfg: DWAConfig, tcfg: TrainConfig) -> int:
    """Approximate forward+backward FLOPs for one training step."""
    B, T, d = tcfg.batch_size, cfg.seq_len, cfg.d_A
    # Transformer layers: QKV+O projections + self-attention + FFN
    layer = B * T * ((4 + 2 * cfg.ffn_mult) * d ** 2 + 2 * T * d)
    parts = (cfg.n_layers_A + cfg.n_layers_B) * layer
    # Retrieval similarity scores
    retrieval = cfg.S * B * cfg.N * cfg.d_k
    # Assembly: k low-rank outer products + apply W
    assembly = B * cfg.k_max * 2 * cfg.r * d ** 2 + B * T * d ** 2
    # LM head (dominates at large vocab)
    head = B * T * d * cfg.vocab_size
    return int(3 * (parts + retrieval + assembly + head))  # ×3 for fwd+bwd


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

            # --- NaN / Inf guard ---
            # Compute the sum of squared gradient values in one pass.
            # If any gradient leaf contains NaN/Inf the sum propagates it,
            # so a single isnan(grad_sq_sum) check covers all parameters.
            grad_state  = nnx.state(grads, nnx.Param)
            grad_leaves = jax.tree_util.tree_leaves(grad_state)
            grad_sq_sum = sum(
                jnp.sum(g.astype(jnp.float32) ** 2) for g in grad_leaves
            )
            has_bad = (
                jnp.isnan(loss) | jnp.isinf(loss)
                | jnp.isnan(grad_sq_sum) | jnp.isinf(grad_sq_sum)
            )
            # Zero out gradients for the poisoned step so Adam moments stay clean.
            safe_state = jax.tree_util.tree_map(
                lambda g: jnp.where(has_bad, jnp.zeros_like(g), g), grad_state
            )
            nnx.update(grads, safe_state)
            # Pre-clip grad norm (NaN replaced with 0 so downstream metrics are clean).
            grad_norm = jnp.sqrt(jnp.where(has_bad, jnp.zeros_like(grad_sq_sum), grad_sq_sum))

            optimizer.update(model, grads)
            safe_loss = jnp.where(has_bad, jnp.zeros_like(loss), loss)

            # Update pool utilisation EMA
            alphas  = info["alphas"]   # [B, k_max]
            indices = info["indices"]  # [B, k_max]
            usage = jnp.zeros(model.cfg.N).at[indices.reshape(-1)].add(
                alphas.reshape(-1) / (batch.shape[0] + 1e-8)
            )
            pool_ema = ema_update(pool_ema, usage, ema_decay)

            return (
                (model, optimizer, pool_ema, step_in_window + 1),
                (safe_loss, indices, grad_norm, has_bad.astype(jnp.int32)),
            )

        init_carry = (model, optimizer, pool_ema_in, jnp.array(0))
        (model, optimizer, pool_ema_out, _), (losses, all_indices, grad_norms, nan_flags) = (
            jax.lax.scan(step_fn, init_carry, (data, lambda_vals))
        )
        return model, optimizer, pool_ema_out, {
            "losses":       losses,           # [steps_per_window]
            "last_indices": all_indices[-1],  # [B, k_max] — last step only
            "grad_norms":   grad_norms,       # [steps_per_window]
            "nan_flags":    nan_flags,        # [steps_per_window] int32
        }

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
# TinyStories real data loader — streaming, chunked iterator
# ---------------------------------------------------------------------------

class TinyStoriesLoader:
    """
    Streams roneneldan/TinyStories, tokenises with GPT-2, and packs tokens
    into fixed-length windows.  Uses the chunked-iterator pattern:

        iterator = dataset.iter(batch_size=CHUNK_SIZE)
        chunk = next(iterator)   # dict {"text": [str, ...]}

    A token buffer accumulates across chunks so we never download more than
    needed.  get_window() refills automatically when the buffer runs low.
    """

    CHUNK_SIZE = 2000  # stories per HuggingFace fetch

    def __init__(self, tokenizer, seq_len: int):
        from datasets import load_dataset
        self.tokenizer      = tokenizer
        self.seq_len        = seq_len
        self._chunks_fetched = 0
        dataset             = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        self._iterator      = dataset.iter(batch_size=self.CHUNK_SIZE)
        self._buf: list[int] = []
        print(f"[DWA] TinyStories loader ready (chunk={self.CHUNK_SIZE} stories, seq_len={seq_len})")

    def _refill(self) -> None:
        chunk = next(self._iterator)          # {"text": ["Once upon a time ...", ...]}
        self._chunks_fetched += 1
        eos   = self.tokenizer.eos_token_id
        for text in chunk["text"]:
            self._buf.extend(self.tokenizer.encode(text))
            if eos is not None:
                self._buf.append(eos)

    def state_dict(self) -> dict:
        """Return serialisable state so the loader can be restored exactly."""
        return {"chunks_fetched": self._chunks_fetched, "buf": list(self._buf)}

    def load_state_dict(self, state: dict) -> None:
        """Fast-forward the HuggingFace iterator to the saved position."""
        target = int(state["chunks_fetched"])
        if target > 0:
            print(f"[DWA] Loader fast-forward: skipping {target} chunks...")
            for i in range(target):
                try:
                    next(self._iterator)      # discard — just advance iterator
                except StopIteration:
                    from datasets import load_dataset
                    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
                    self._iterator = dataset.iter(batch_size=self.CHUNK_SIZE)
                    self._chunks_fetched = 0
                    break
                if (i + 1) % 500 == 0:
                    print(f"[DWA] Loader fast-forward: {i + 1}/{target} chunks...")
        self._chunks_fetched = target
        self._buf = list(state["buf"])
        print(f"[DWA] Loader restored at chunk {target} ({len(self._buf)} tokens buffered).")

    def get_window(self, steps: int, batch_size: int) -> np.ndarray:
        """Return int32 array [steps, batch_size, seq_len] of real token IDs."""
        needed = steps * batch_size * self.seq_len
        while len(self._buf) < needed:
            self._refill()
        tokens    = np.array(self._buf[:needed], dtype=np.int32)
        self._buf = self._buf[needed:]
        return tokens.reshape(steps, batch_size, self.seq_len)


# ---------------------------------------------------------------------------
# Pattern data — learnable repeat-period sequences
# ---------------------------------------------------------------------------

def _make_pattern_window(
    rng: jax.Array, steps: int, batch_size: int, seq_len: int, vocab_size: int
) -> jnp.ndarray:
    """
    [steps, B, seq_len] where each row repeats a random PATTERN_PERIOD-length
    pattern.  After seeing the first PATTERN_PERIOD tokens the model can
    predict all remaining positions deterministically → loss → ~0.
    """
    rng_p, rng_o = jax.random.split(rng)
    n = steps * batch_size
    patterns = jax.random.randint(rng_p, (n, PATTERN_PERIOD), 0, vocab_size)
    offsets  = jax.random.randint(rng_o, (n,), 0, PATTERN_PERIOD)

    def make_seq(pattern, offset):
        idx = (offset + jnp.arange(seq_len)) % PATTERN_PERIOD
        return pattern[idx]

    seqs = jax.vmap(make_seq)(patterns, offsets)   # [n, seq_len]
    return seqs.reshape(steps, batch_size, seq_len)


# ---------------------------------------------------------------------------
# Generation (greedy, for learning verification)
# ---------------------------------------------------------------------------

def generate(
    model: DWAModel,
    prefix_ids,               # list[int] or 1-D array
    n_new: int,
    tcfg: TrainConfig,
    temperature: float = 0.8,
    repetition_penalty: float = 1.3,
) -> list[int]:
    """Temperature-sampled generation with repetition penalty on recent tokens."""
    tokens = list(map(int, prefix_ids))
    cfg = model.cfg
    lam = tcfg.lambda_sharpen_end
    rng = jax.random.PRNGKey(0)
    for _ in range(n_new):
        ids = jnp.array(tokens[-cfg.seq_len:], dtype=jnp.int32)[None]   # [1, T]
        logits, _ = model(ids, lam, is_warmup=False, use_pallas=False)
        logit_vec = np.array(logits[0, -1], dtype=np.float32)           # host copy

        # Down-weight tokens seen in the recent context window
        for t in set(tokens[-64:]):
            if 0 <= t < len(logit_vec):
                logit_vec[t] /= repetition_penalty

        logit_vec /= temperature
        rng, k = jax.random.split(rng)
        tokens.append(int(jax.random.categorical(k, jnp.array(logit_vec))))
    return tokens


def _verify_learning(
    model: DWAModel, cfg: DWAConfig, tcfg: TrainConfig
) -> None:
    """Print pattern-learning accuracy for 3 random repeat-period test cases."""
    print(f"\n[DWA] Learning verification (period-{PATTERN_PERIOD} repeat pattern, "
          f"vocab={cfg.vocab_size}):")
    print(f"  Theoretical min loss ≈ "
          f"{PATTERN_PERIOD / cfg.seq_len * jnp.log(cfg.vocab_size).item():.3f} nats")

    rng = jax.random.PRNGKey(9999)
    for trial in range(3):
        rng, k = jax.random.split(rng)
        pattern = list(map(int, jax.random.randint(k, (PATTERN_PERIOD,), 0, cfg.vocab_size)))
        # Two full periods as prefix so model can identify the pattern
        prefix  = (pattern * 4)[:2 * PATTERN_PERIOD]
        n_gen   = 2 * PATTERN_PERIOD          # generate two more periods
        out     = generate(model, prefix, n_gen, tcfg)
        gen_new = out[len(prefix):]
        exp_new = (pattern * 16)[len(prefix): len(prefix) + n_gen]
        acc     = sum(g == e for g, e in zip(gen_new, exp_new)) / n_gen
        ok      = "✓" if acc >= 0.875 else "✗"
        print(f"  [{ok}] pattern={pattern}  gen={gen_new}  exp={exp_new}  acc={acc:.0%}")


# ---------------------------------------------------------------------------
# Text generation helper for TinyStories learning verification
# ---------------------------------------------------------------------------

def _generate_text_sample(model: DWAModel, tokenizer, tcfg: TrainConfig, step: int) -> None:
    """Generate ~60 tokens from a fixed prompt and print decoded text."""
    prompt = "Once upon a time"
    ids    = tokenizer.encode(prompt)
    out    = generate(model, ids, n_new=60, tcfg=tcfg)
    text   = tokenizer.decode(out, skip_special_tokens=True)
    print(f"\n[DWA] step={step} sample: {text}\n")


# ---------------------------------------------------------------------------
# Safety helpers (run on host, outside JIT)
# ---------------------------------------------------------------------------

def _check_nan_params(model: DWAModel) -> tuple[bool, str]:
    """Scan all trainable parameters for NaN / Inf values.

    Returns (has_bad, first_bad_path).  Called every few windows so we catch
    corruption early instead of discovering it at the end of a run.
    """
    params = nnx.state(model, nnx.Param)
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        arr = np.array(leaf)
        if np.any(~np.isfinite(arr)):
            name = "/".join(str(k) for k in path)
            return True, name
    return False, ""


def _revive_dead_vectors(
    model: DWAModel,
    pool_ema: jnp.ndarray,
    cfg: DWAConfig,
    step: int,
) -> int:
    """Replace pool vectors that have near-zero EMA usage.

    Dead vectors are re-seeded as perturbed copies of high-usage vectors so
    they start with meaningful geometry rather than drifting randomly.
    The existing sharding of model.pool.vectors is preserved.

    Returns the number of vectors revived.
    """
    ema_np = np.array(pool_ema, dtype=np.float32)
    dead_mask = ema_np < cfg.dead_vector_threshold
    n_dead = int(dead_mask.sum())
    if n_dead == 0:
        return 0

    # Donors: top-50% by EMA usage (avoids picking just-revived vectors)
    median_ema = float(np.median(ema_np[ema_np > 0])) if (ema_np > 0).any() else 0.0
    donor_idx = np.where(ema_np >= median_ema)[0]
    if len(donor_idx) == 0:
        return 0

    dead_idx = np.where(dead_mask)[0]
    pool_np  = np.array(model.pool.vectors[...], dtype=np.float32)  # gather to host
    rng      = np.random.default_rng(step)

    chosen_donors = rng.choice(donor_idx, size=n_dead, replace=True)
    noise = rng.normal(0.0, 0.01, (n_dead, cfg.D)).astype(pool_np.dtype)
    pool_np[dead_idx] = pool_np[chosen_donors] + noise

    # Put back — honour the existing sharding (model-parallel or replicated)
    orig_arr = model.pool.vectors[...]
    new_jax  = jnp.array(pool_np, dtype=orig_arr.dtype)
    orig_sharding = getattr(orig_arr, "sharding", None)
    if orig_sharding is not None:
        new_jax = jax.device_put(new_jax, orig_sharding)
    model.pool.vectors[...] = new_jax
    return n_dead


# ---------------------------------------------------------------------------
# Checkpoint save / restore (orbax-based)
# ---------------------------------------------------------------------------

def _get_ckpt_manager(ckpt_dir: str, keep: int = 3) -> "ocp.CheckpointManager":
    os.makedirs(ckpt_dir, exist_ok=True)
    return ocp.CheckpointManager(
        ckpt_dir,
        options=ocp.CheckpointManagerOptions(max_to_keep=keep),
    )


def save_checkpoint(
    ckpt_dir: str,
    model: DWAModel,
    optimizer: nnx.Optimizer,
    pool_ema: jnp.ndarray,
    steps_done: int,
    rng: jax.Array,
    loader: "TinyStoriesLoader | None",
    keep: int = 3,
) -> None:
    """
    Save a full training checkpoint.

    JAX arrays (model params, opt state, pool EMA, RNG) go through orbax
    StandardCheckpointer — which handles sharded arrays correctly.
    The token buffer (variable-length Python list) is saved as a .npy file
    alongside the orbax directory at ckpt_dir/{steps_done}/buf.npy.
    """
    mngr = _get_ckpt_manager(ckpt_dir, keep)

    # Model parameters: State pytree of numpy arrays
    model_np = jax.tree_util.tree_map(np.array, nnx.state(model, nnx.Param))

    # Optimizer state: flatten to an indexed dict so we avoid optax custom types
    # (MaskedState, etc.) that orbax cannot serialize as-is.
    opt_leaves, _ = jax.tree_util.tree_flatten(optimizer.opt_state)
    opt_dict = {f"{i:04d}": np.array(leaf) for i, leaf in enumerate(opt_leaves)}

    save_item = {
        "model":          model_np,
        "opt":            opt_dict,
        "meta": {
            "opt_step":       np.array(int(optimizer.step[...]), dtype=np.int32),
            "pool_ema":       np.array(pool_ema, dtype=np.float32),
            "steps_done":     np.array(steps_done, dtype=np.int32),
            "rng":            np.array(rng),
            "chunks_fetched": np.array(loader._chunks_fetched if loader else 0, dtype=np.int32),
        },
    }

    mngr.save(steps_done, args=ocp.args.StandardSave(save_item))
    mngr.wait_until_finished()

    # Save variable-length token buffer alongside the orbax step directory
    if loader is not None and loader._buf:
        buf_path = os.path.join(ckpt_dir, str(steps_done), "buf.npy")
        np.save(buf_path, np.array(loader._buf, dtype=np.int32))

    print(f"[Ckpt] Saved step {steps_done} → {ckpt_dir}/{steps_done}/")


def _abstract_like(item):
    """Recursively build an abstract (shape+dtype) version of a numpy pytree."""
    return jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, item)


def load_checkpoint(
    ckpt_dir: str,
    steps_done_target: int,
    model: DWAModel,
    optimizer: nnx.Optimizer,
    cfg: DWAConfig,
    mesh,
) -> tuple[int, jax.Array, dict | None]:
    """
    Restore model, optimizer, and metadata from a checkpoint.

    Handles model-parallel sharding: pool vectors are re-sharded to
    P('model', None) when mesh has a 'model' dimension > 1.

    Returns (steps_done, rng, loader_state_dict).
    """
    mngr = _get_ckpt_manager(ckpt_dir)

    # Build abstract reference from the current (freshly-initialised) model
    model_np   = jax.tree_util.tree_map(np.array, nnx.state(model, nnx.Param))
    opt_leaves, opt_treedef = jax.tree_util.tree_flatten(optimizer.opt_state)
    opt_dict_ref = {f"{i:04d}": np.array(leaf) for i, leaf in enumerate(opt_leaves)}

    abstract = {
        "model": _abstract_like(model_np),
        "opt":   _abstract_like(opt_dict_ref),
        "meta": {
            "opt_step":       jax.ShapeDtypeStruct((), np.int32),
            "pool_ema":       jax.ShapeDtypeStruct((cfg.N,), np.float32),
            "steps_done":     jax.ShapeDtypeStruct((), np.int32),
            "rng":            _abstract_like(np.array(jax.random.PRNGKey(0))),
            "chunks_fetched": jax.ShapeDtypeStruct((), np.int32),
        },
    }

    restored = mngr.restore(steps_done_target, args=ocp.args.StandardRestore(abstract))

    # --- Restore model parameters ---
    n_model = mesh.shape.get("model", 1) if mesh is not None else 1
    pool_sharding = NamedSharding(mesh, P("model", None)) if (mesh is not None and n_model > 1) else None

    if pool_sharding is not None:
        # Re-shard pool vectors without materialising the full array on one device:
        # slice each device's rows and put them directly.
        pool_np  = np.array(restored["model"]["pool"]["vectors"])  # host numpy [N, D]
        N_local  = pool_np.shape[0] // n_model
        idx_map  = pool_sharding.addressable_devices_indices_map(pool_np.shape)
        per_dev  = []
        for dev in pool_sharding.addressable_devices:
            rows = idx_map[dev][0]
            shard = jax.device_put(
                jnp.array(pool_np[rows], dtype=model.pool.vectors[...].dtype), dev
            )
            per_dev.append(shard)
        sharded_pool = jax.make_array_from_single_device_arrays(
            pool_np.shape, pool_sharding, per_dev
        )
        # Update model params without pool first, then fix pool
        nnx.update(model, restored["model"])
        model.pool.vectors[...] = sharded_pool
    else:
        nnx.update(model, restored["model"])

    # --- Restore optimizer state ---
    restored_opt_leaves = [
        jnp.array(restored["opt"][f"{i:04d}"]) for i in range(len(opt_leaves))
    ]
    optimizer.opt_state = jax.tree_util.tree_unflatten(opt_treedef, restored_opt_leaves)
    optimizer.step[...] = jnp.array(restored["meta"]["opt_step"], dtype=jnp.uint32)

    # --- Pool EMA ---
    pool_ema = jnp.array(restored["meta"]["pool_ema"])

    # --- RNG + steps ---
    rng         = jnp.array(restored["meta"]["rng"])
    steps_done  = int(restored["meta"]["steps_done"])

    # --- Loader state (buf + chunks_fetched) ---
    chunks_fetched = int(restored["meta"]["chunks_fetched"])
    buf_path = os.path.join(ckpt_dir, str(steps_done_target), "buf.npy")
    buf = list(np.load(buf_path).tolist()) if os.path.exists(buf_path) else []
    loader_state = {"chunks_fetched": chunks_fetched, "buf": buf}

    print(f"[Ckpt] Loaded step {steps_done} from {ckpt_dir}/{steps_done_target}/")
    return steps_done, rng, loader_state, pool_ema


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
# Parameter breakdown table
# ---------------------------------------------------------------------------

def _print_param_table(model: DWAModel) -> None:
    """Print a per-component parameter count and memory table at training start."""
    GROUPS = [
        ("Embedding",          lambda p: p.startswith("embed")),
        ("Part A · Attention", lambda p: p.startswith("part_a") and "attn" in p),
        ("Part A · FFN",       lambda p: p.startswith("part_a") and "ffn"  in p),
        ("Part A · Norms",     lambda p: p.startswith("part_a") and "norm" in p),
        ("Part B · Attention", lambda p: p.startswith("part_b") and "attn" in p),
        ("Part B · FFN",       lambda p: p.startswith("part_b") and "ffn"  in p),
        ("Part B · Norms",     lambda p: p.startswith("part_b") and "norm" in p),
        ("Pool · Vectors",     lambda p: p.startswith("pool")   and "vectors" in p),
        ("Pool · Key Proj",    lambda p: p.startswith("pool")   and "key_proj" in p),
        ("Retrieval",          lambda p: p.startswith("retrieval")),
        ("Assembler",          lambda p: p.startswith("assembler")),
        ("LM Head",            lambda p: p.startswith("lm_head")),
    ]

    pure = nnx.as_pure(nnx.state(model, nnx.Param))
    lpaths, _ = jax.tree_util.tree_flatten_with_path(pure)

    def _pstr(path) -> str:
        # DictKey(key='foo') → 'foo'; fallback to str() for other key types
        return "/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)

    counts: dict[str, int] = {label: 0 for label, _ in GROUPS}
    mbytes: dict[str, int] = {label: 0 for label, _ in GROUPS}
    other_n = 0

    for path, leaf in lpaths:
        p = _pstr(path)
        matched = False
        for label, fn in GROUPS:
            if fn(p):
                counts[label] += leaf.size
                mbytes[label] += leaf.size * leaf.dtype.itemsize
                matched = True
                break
        if not matched:
            other_n += leaf.size

    total_n = sum(counts.values()) + other_n
    total_b = sum(mbytes.values())
    W = 24

    print(f"\n[DWA] Parameter breakdown — {total_n / 1e6:.1f}M params, "
          f"{total_b / 1e6:.0f} MB storage (dtype-aware):")
    print(f"  {'Component':<{W}}  {'Params':>10}  {'Storage':>10}  {'Share':>6}")
    print(f"  {'─' * W}  {'─' * 10}  {'─' * 10}  {'─' * 6}")
    for label, _ in GROUPS:
        n, b = counts[label], mbytes[label]
        if n == 0:
            continue
        print(f"  {label:<{W}}  {n / 1e6:>8.3f} M  {b / 1e6:>7.1f} MB  {100 * n / total_n:>5.1f}%")
    if other_n > 0:
        print(f"  {'(unmatched)':<{W}}  {other_n / 1e6:>8.3f} M")
    print(f"  {'─' * W}  {'─' * 10}  {'─' * 10}  {'─' * 6}")
    print(f"  {'TOTAL':<{W}}  {total_n / 1e6:>8.3f} M  {total_b / 1e6:>7.1f} MB  100.0%\n")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(run_cfg: RunConfig) -> None:
    """
    Main training loop.  Accepts a fully-resolved RunConfig.

    All hyperparameters, sharding strategy, data source, and checkpoint
    settings come from run_cfg.  Build one with load_config() or assemble
    it manually from DWAConfig / TrainConfig / ShardingConfig etc.
    """
    cfg  = run_cfg.model
    tcfg = run_cfg.train

    # --- Data source setup (before model build so vocab_size is set) ---
    tokenizer   = None
    use_pattern = False
    if run_cfg.data.source == "tiny_stories":
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        # Pad to the nearest multiple of 64 ≥ 50257 so vocab_parallel sharding
        # divides evenly for any n_model in {1,2,4,8}.
        cfg.vocab_size = ((tokenizer.vocab_size + 63) // 64) * 64   # 50304
        print(f"[DWA] TinyStories mode: GPT-2 tokenizer, padded vocab_size={cfg.vocab_size}")
    elif run_cfg.data.source == "pattern":
        use_pattern = True

    gen_every  = run_cfg.data.gen_every
    ckpt_dir   = run_cfg.checkpoint.dir
    ckpt_every = run_cfg.checkpoint.every
    resume     = run_cfg.checkpoint.resume

    devices = jax.devices()
    n_devices = len(devices)
    device_kind = devices[0].device_kind

    # --- Mesh: 2D (data × model) ---
    n_model = _select_n_model(cfg, n_devices, run_cfg.sharding.n_model)
    n_data = n_devices // n_model
    mesh = _build_mesh(devices, n_model)

    sharding_src = ("auto" if run_cfg.sharding.n_model == "auto"
                    else f"explicit n_model={run_cfg.sharding.n_model}")
    print(f"[DWA] Training on {n_devices}× {device_kind}")
    print(f"[DWA] Mesh: {n_data}×data  {n_model}×model  "
          f"(pool+Adam/device ≈ {cfg.N * cfg.D * 4 * 3 / n_model / 1e9:.1f} GB)"
          f"  [{sharding_src}]")
    print(f"[DWA] Model config: N={cfg.N}, D={cfg.D}, d_A={cfg.d_A}, r={cfg.r}, "
          f"layers={cfg.n_layers_A}+{cfg.n_layers_B}, vocab={cfg.vocab_size}")

    # Per data-replica batch size
    assert tcfg.batch_size % n_data == 0, "batch_size must be divisible by n_data"
    local_batch = tcfg.batch_size // n_data

    scheduler = PhaseScheduler(tcfg)
    lambda_array = scheduler.make_lambda_array()     # [total_steps]

    # Initialise model.  For configs needing model parallelism (n_model > 1),
    # create the pool directly on each device's HBM before building the model
    # — this avoids materialising the full ~4 GB pool on device 0 first.
    # The optimizer is built after so Adam states inherit the pool's sharding.
    rng = jax.random.PRNGKey(tcfg.seed)
    sharded_pool = _make_sharded_pool_vectors(cfg, mesh, rng)  # None if n_model==1
    model = DWAModel(cfg, nnx.Rngs(rng), pool_vectors=sharded_pool)
    optimizer = _build_optimizer(model, tcfg)
    pool_ema = jnp.zeros(cfg.N)

    _print_param_table(model)

    step_flops = compute_step_flops(cfg, tcfg)
    data_label = "tiny_stories" if tokenizer is not None else ("pattern" if use_pattern else "random")
    print(f"[DWA] FLOPs/step (fwd+bwd, approx): {step_flops / 1e9:.1f}G"
          f"  (data={data_label})")

    # TinyStories streaming loader (created once; maintains iterator state across windows)
    loader = TinyStoriesLoader(tokenizer, cfg.seq_len) if tokenizer is not None else None

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
                mesh=mesh,  # always pass mesh — needed for model-parallel sharding
            )
        return compiled_fns[key]

    n_windows  = tcfg.total_steps // tcfg.steps_per_window
    steps_done = 0
    start_window = 0
    rng = jax.random.PRNGKey(tcfg.seed + 1)

    # --- Resume from checkpoint ---
    if resume and ckpt_dir:
        mngr_probe = _get_ckpt_manager(ckpt_dir)
        latest = mngr_probe.latest_step()
        if latest is not None:
            steps_done, rng, loader_state, pool_ema = load_checkpoint(
                ckpt_dir, latest, model, optimizer, cfg, mesh
            )
            model.pool_ema[...] = pool_ema
            start_window = steps_done // tcfg.steps_per_window
            if loader is not None and loader_state is not None:
                loader.load_state_dict(loader_state)
            print(f"[Ckpt] Resuming from step {steps_done} (window {start_window}/{n_windows})")
        else:
            print(f"[Ckpt] No checkpoint in '{ckpt_dir}', starting fresh.")

    print(f"[DWA] Training for {tcfg.total_steps} steps "
          f"({n_windows} windows × {tcfg.steps_per_window} steps)")
    if ckpt_dir:
        print(f"[DWA] Checkpointing: dir='{ckpt_dir}'  every={ckpt_every} steps  keep=3")
    print(f"[DWA] Safety: grad_clip={tcfg.grad_clip_norm}  "
          f"nan_stop={tcfg.nan_emergency_stop}w  "
          f"spike_sigma={tcfg.loss_spike_sigma}σ  "
          f"revival_every={tcfg.revival_interval_steps}s")

    # Steady-state tracking
    _ss_steps, _ss_time = 0, 0.0
    _win_min = float("inf")

    # Safety state
    _consecutive_nan = 0
    _loss_window: deque = deque(maxlen=50)   # rolling buffer for spike detection

    t0 = time.time()
    for window_idx in range(start_window, n_windows):
        start_step = window_idx * tcfg.steps_per_window
        phase = scheduler.get_phase(start_step)
        is_warmup = scheduler.is_warmup(start_step)
        aux_on = scheduler.aux_enabled(start_step)

        # Slice lambda schedule for this window
        lam_window = lambda_array[start_step: start_step + tcfg.steps_per_window]

        # Data window [steps, B, seq_len]
        rng, data_rng = jax.random.split(rng)
        if loader is not None:
            data_window = jnp.array(loader.get_window(tcfg.steps_per_window, tcfg.batch_size))
        elif use_pattern:
            data_window = _make_pattern_window(
                data_rng, tcfg.steps_per_window, tcfg.batch_size,
                cfg.seq_len, cfg.vocab_size,
            )
        else:
            data_window = _synthetic_window(
                data_rng, tcfg.steps_per_window, tcfg.batch_size,
                cfg.seq_len, cfg.vocab_size,
            )

        # Shard data: batch dim across data-parallel replicas (dim 1)
        data_sharded = jax.device_put(data_window, NamedSharding(mesh, P(None, "data", None)))

        train_fn = get_train_fn(is_warmup, aux_on)
        t_win = time.time()
        model, optimizer, pool_ema, info = train_fn(
            model, optimizer, data_sharded, lam_window, pool_ema, tcfg.ema_decay
        )
        # Block until TPU computation finishes before timing
        jax.block_until_ready(
            (info["losses"], info["last_indices"], info["grad_norms"], info["nan_flags"])
        )
        win_secs = time.time() - t_win

        steps_done += tcfg.steps_per_window
        elapsed = time.time() - t0
        mean_loss = float(info["losses"].mean())
        win_steps_per_sec = tcfg.steps_per_window / win_secs
        win_tok_per_sec = int(win_steps_per_sec * tcfg.batch_size * cfg.seq_len)

        achieved_tflops = step_flops * win_steps_per_sec / 1e12
        compile_thresh = (max(3.0, 4.0 * _win_min) if _win_min < float("inf") else 3.0)
        is_compile_win = win_secs > compile_thresh
        if not is_compile_win:
            _ss_steps += tcfg.steps_per_window
            _ss_time  += win_secs
            _win_min   = min(_win_min, win_secs)
        print(
            f"[DWA] step={steps_done:6d}/{tcfg.total_steps} "
            f"phase={phase:8s} λ={scheduler.get_lambda(start_step):.2f} "
            f"loss={mean_loss:.4f} "
            f"win={win_secs:.1f}s  steps/s={win_steps_per_sec:.1f}  "
            f"tok/s={win_tok_per_sec:,}  TFLOP/s={achieved_tflops:.1f}"
            + ("  [compile]" if is_compile_win else "")
        )

        # ── Safety checks ────────────────────────────────────────────────────

        # 1. NaN / Inf in losses / gradients
        nan_count  = int(info["nan_flags"].sum())
        mean_gnorm = float(info["grad_norms"].mean())
        max_gnorm  = float(info["grad_norms"].max())
        if nan_count > 0:
            _consecutive_nan += 1
            print(
                f"[Safety] NaN/Inf: {nan_count}/{tcfg.steps_per_window} steps "
                f"(grads zeroed) — consecutive bad windows: "
                f"{_consecutive_nan}/{tcfg.nan_emergency_stop}"
            )
            if _consecutive_nan >= tcfg.nan_emergency_stop:
                print("[Safety] EMERGENCY STOP: too many consecutive NaN windows.")
                break
        else:
            _consecutive_nan = 0

        # 2. Loss spike detector (rolling mean ± σ over last 50 windows)
        if len(_loss_window) >= 10:
            mu    = float(np.mean(_loss_window))
            sigma = float(np.std(_loss_window)) + 1e-8
            if mean_loss > mu + tcfg.loss_spike_sigma * sigma:
                print(
                    f"[Safety] Loss spike: {mean_loss:.4f} vs "
                    f"rolling {mu:.4f} ± {sigma:.4f} "
                    f"({(mean_loss - mu) / sigma:.1f}σ)"
                )
        _loss_window.append(mean_loss)

        # 3. Periodic parameter NaN check (every 10 windows — host-side scan)
        if window_idx % 10 == 0:
            has_bad, bad_name = _check_nan_params(model)
            if has_bad:
                print(f"[Safety] CRITICAL: NaN/Inf in parameter '{bad_name}'. Stopping.")
                break

        # 4. Pool-collapse check: unique vectors retrieved + EMA entropy
        last_idx     = np.array(info["last_indices"])   # [B, k_max] on host
        n_unique     = int(np.unique(last_idx).shape[0])
        max_possible = min(tcfg.batch_size * cfg.k_max, cfg.N)
        ema_np   = np.array(pool_ema)
        ema_norm = ema_np / (ema_np.sum() + 1e-8)
        entropy  = float(-np.sum(ema_norm * np.log(ema_norm + 1e-8)))
        n_active = int((ema_np > 0.01 * ema_np.mean()).sum())
        collapse_flag = n_unique < cfg.k_max * 2
        print(
            f"[Pool] unique={n_unique}/{max_possible}  "
            f"active={n_active}/{cfg.N}  "
            f"entropy={entropy / np.log(cfg.N):.3f}  "
            f"gnorm_mean={mean_gnorm:.3f} max={max_gnorm:.3f}"
            + ("  [COLLAPSE RISK]" if collapse_flag else "")
        )

        # 5. Dead vector revival (every revival_interval_steps steps)
        if steps_done % tcfg.revival_interval_steps == 0:
            n_revived = _revive_dead_vectors(model, pool_ema, cfg, steps_done)
            if n_revived > 0:
                print(f"[Safety] Revived {n_revived}/{cfg.N} dead pool vectors.")

        # Text generation check every gen_every steps
        prev_steps = steps_done - tcfg.steps_per_window
        if tokenizer is not None and (steps_done // gen_every) > (prev_steps // gen_every):
            _generate_text_sample(model, tokenizer, tcfg, steps_done)

        # Checkpoint save (boundary crossing check avoids double-saves at resume)
        if ckpt_dir and ckpt_every > 0 and (steps_done // ckpt_every) > (prev_steps // ckpt_every):
            save_checkpoint(ckpt_dir, model, optimizer, pool_ema, steps_done, rng, loader)
            # Dump effective config once, alongside the first checkpoint
            cfg_out = os.path.join(ckpt_dir, "effective_config.yaml")
            if not os.path.exists(cfg_out):
                save_config(run_cfg, cfg_out)

        # Update model's pool EMA (non-trainable variable)
        model.pool_ema[...] = pool_ema

    elapsed_total = time.time() - t0
    print(f"[DWA] Training complete in {elapsed_total:.1f}s")
    if _ss_time > 0:
        ss_sps  = _ss_steps / _ss_time
        ss_tflops = step_flops * ss_sps / 1e12
        ss_tok   = int(ss_sps * tcfg.batch_size * cfg.seq_len)
        # TPU v5 lite BF16 MXU peak: 197 TFLOP/s per chip
        peak = 197.0 * n_devices if "v5" in device_kind.lower() else None
        mfu_str = (f"  MFU={ss_tflops/peak*100:.2f}% (vs {peak:.0f} TFLOP/s BF16 peak)"
                   if peak else "")
        print(f"[DWA] Steady-state throughput: {ss_sps:.0f} steps/s  "
              f"{ss_tok:,} tok/s  {ss_tflops:.1f} TFLOP/s{mfu_str}")
        compile_secs = elapsed_total - _ss_time
        print(f"[DWA] Time breakdown: {_ss_time:.1f}s training + {compile_secs:.1f}s XLA compilation")

    if use_pattern:
        _verify_learning(model, cfg, tcfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_run_config_from_args(args) -> RunConfig:
    """
    Build a RunConfig from parsed CLI arguments.

    --config PATH is the base; every other flag is an override on top.
    Old preset flags (--full, --medium, …) still work when --config is absent.
    """
    # --- Base config ---
    if args.config:
        run_cfg = load_config(args.config)
        print(f"[DWA] Loaded config from '{args.config}' (name={run_cfg.name!r})")
    else:
        # Build from preset flags (backward compat)
        if args.verify:
            cfg_model = DWAConfig.pattern_test()
        elif args.wide:
            cfg_model = DWAConfig.full_wide()
        elif args.full:
            cfg_model = DWAConfig()
        elif args.large:
            cfg_model = DWAConfig.large()
        elif args.mxu:
            cfg_model = DWAConfig.medium_mxu(bf16=args.bf16)
        elif args.medium:
            cfg_model = DWAConfig.medium()
        else:
            cfg_model = DWAConfig.small()

        tcfg = TrainConfig()
        if args.verify:
            tcfg.total_steps, tcfg.warmup_steps, tcfg.gate_on_steps = 6000, 300, 1500
            tcfg.batch_size = 64
        elif args.wide:
            tcfg.batch_size, tcfg.steps_per_window = (64 if args.remat else 32), 128
        elif args.full:
            tcfg.batch_size, tcfg.steps_per_window = 32, 128
        elif args.large:
            tcfg.total_steps   //= 5
            tcfg.warmup_steps  //= 5
            tcfg.gate_on_steps //= 5
            tcfg.batch_size     = 128
        elif args.mxu or args.medium:
            tcfg.total_steps   //= 5
            tcfg.warmup_steps  //= 5
            tcfg.gate_on_steps //= 5
            tcfg.batch_size     = 128
        else:
            tcfg.total_steps   //= 10
            tcfg.warmup_steps  //= 10
            tcfg.gate_on_steps //= 10
            tcfg.batch_size     = 16

        source = "pattern" if args.verify else ("tiny_stories" if args.tiny_stories else "random")
        run_cfg = RunConfig(
            model=cfg_model,
            train=tcfg,
            sharding=ShardingConfig(n_model="auto"),
            data=DataConfig(source=source, gen_every=args.gen_every),
            checkpoint=CheckpointConfig(
                dir=args.ckpt_dir, every=args.ckpt_every, resume=args.resume,
            ),
        )

    # --- Per-flag overrides (apply regardless of --config or preset) ---
    if args.bf16:
        run_cfg.model.bf16_pool = True
    if args.bf16_compute or (not args.config and (args.full or args.wide)):
        run_cfg.model.compute_dtype = jnp.bfloat16
        print("[DWA] BF16 compute enabled: linear layers will run in bfloat16")
    if args.remat:
        run_cfg.model.remat = True
        print("[DWA] Gradient checkpointing enabled: ~4× less activation memory, ~33% more FLOPs")
    if args.tiny_stories:
        run_cfg.data.source = "tiny_stories"
    if args.n_model is not None:
        run_cfg.sharding.n_model = args.n_model if args.n_model == "auto" else int(args.n_model)

    # Scalar overrides (highest priority)
    if args.steps is not None:
        run_cfg.train.total_steps = args.steps
    if args.batch_size is not None:
        run_cfg.train.batch_size = args.batch_size
    if args.steps_per_window is not None:
        run_cfg.train.steps_per_window = args.steps_per_window
    if args.gen_every != 100:            # non-default means user explicitly set it
        run_cfg.data.gen_every = args.gen_every
    if args.ckpt_dir:
        run_cfg.checkpoint.dir = args.ckpt_dir
    if args.ckpt_every != 1000:
        run_cfg.checkpoint.every = args.ckpt_every
    if args.resume:
        run_cfg.checkpoint.resume = True

    return run_cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DWA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --config configs/full.yaml
  python train.py --config configs/medium.yaml --steps 50000 --ckpt-dir ckpts
  python train.py --config configs/small.yaml --n-model 1
  python train.py --full --bf16                         (legacy preset flags)
  python train.py --tiny-stories --ckpt-dir ckpts --resume
        """,
    )

    # --- Config file ---
    parser.add_argument("--config", type=str, default="",
                        help="Path to a YAML config file (base for all settings)")

    # --- Preset flags (backward compat; ignored when --config is given) ---
    presets = parser.add_argument_group("preset configs (ignored when --config is used)")
    presets.add_argument("--full",   action="store_true", help="Full-scale config")
    presets.add_argument("--wide",   action="store_true", help="Full pool + d=512")
    presets.add_argument("--large",  action="store_true", help="2× medium pool, 8 layers")
    presets.add_argument("--medium", action="store_true", help="Medium config (16GB/device)")
    presets.add_argument("--mxu",    action="store_true", help="MXU-aligned: r=128")
    presets.add_argument("--verify", action="store_true", help="Pattern-learning verification run")

    # --- Model overrides ---
    model_g = parser.add_argument_group("model overrides")
    model_g.add_argument("--bf16",         action="store_true", help="Pool in bfloat16")
    model_g.add_argument("--bf16-compute", action="store_true", default=False,
                         help="Linear layers in bfloat16 (~4× MXU throughput)")
    model_g.add_argument("--remat",        action="store_true",
                         help="Gradient checkpointing (~4× less activation memory)")

    # --- Sharding override ---
    parser.add_argument("--n-model", type=str, default=None, metavar="N|auto",
                        help="Model-parallel degree: integer (1/2/4/8) or 'auto' (default)")

    # --- Training overrides ---
    train_g = parser.add_argument_group("training overrides")
    train_g.add_argument("--steps",            type=int, default=None)
    train_g.add_argument("--batch-size",       type=int, default=None)
    train_g.add_argument("--steps-per-window", type=int, default=None)

    # --- Data ---
    data_g = parser.add_argument_group("data")
    data_g.add_argument("--tiny-stories", action="store_true",
                        help="Train on roneneldan/TinyStories with GPT-2 tokenizer")
    data_g.add_argument("--gen-every", type=int, default=100,
                        help="Generate text sample every N steps (default: 100)")

    # --- Checkpoint ---
    ckpt_g = parser.add_argument_group("checkpointing")
    ckpt_g.add_argument("--ckpt-dir",   type=str, default="",
                        help="Directory to save checkpoints")
    ckpt_g.add_argument("--ckpt-every", type=int, default=1000,
                        help="Save checkpoint every N steps (default: 1000)")
    ckpt_g.add_argument("--resume",     action="store_true",
                        help="Resume from latest checkpoint in --ckpt-dir")

    args = parser.parse_args()
    run_cfg = _build_run_config_from_args(args)
    train(run_cfg)


if __name__ == "__main__":
    main()
