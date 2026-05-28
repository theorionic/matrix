"""
DWA training entry point — multi-host TPU training with exact preemption resume.

Training strategy:
  - Multi-host: set JAX_COORDINATOR_ADDRESS / JAX_NUM_PROCESSES / JAX_PROCESS_ID env vars
  - Mesh(N devices, ('data','model')) with GSPMD data + model parallelism
  - nnx.jit wraps a jax.lax.scan window of `steps_per_window` steps
  - All gradient sync happens on-device inside the scan (no CPU round-trips)
  - Three-phase schedule: warmup → gate_on → sharpen
  - Per-component LRs via optax.masked chain
  - Grain data loader: per-host sharding + exact resume after preemption

Run (single host):
    python train.py                      # small config (fast sanity check)
    python train.py --full               # full-scale config

Run (multi-host, launch on every host):
    JAX_COORDINATOR_ADDRESS=host0:1234 JAX_NUM_PROCESSES=4 JAX_PROCESS_ID=<rank> python train.py ...
"""

from __future__ import annotations

import os

# XLA persistent compilation cache — avoids recompiling across runs.
# Must be set before any JAX import.  Falls back gracefully if the flag
# isn't recognized by this JAX version.
_CACHE_DIR = os.environ.get("JAX_COMPILATION_CACHE_DIR",
                             os.path.join(os.path.dirname(__file__), ".jax_cache"))
os.environ["JAX_COMPILATION_CACHE_DIR"] = _CACHE_DIR
os.makedirs(_CACHE_DIR, exist_ok=True)

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

# Enable XLA persistent cache (set before other JAX imports)
_jax_cfg.config.update("jax_compilation_cache_dir", _CACHE_DIR)

import argparse
import functools
import os
import time
import threading
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
from src.dwa.losses import task_loss
from src.dwa.model import DWAModel, forward_and_loss
from src.dwa.run_config import (
    CheckpointConfig, DataConfig, RunConfig, ShardingConfig, WandbConfig,
    load_config, save_config, to_dict,
)
from src.dwa.monitor import LossAdaptiveLRController, PoolCollapseDetector
from src.dwa.parquet_grain_loader import RollingParquetLoader
from src.dwa.schedule import PhaseScheduler
from src.dwa.utils import ema_update
from jax.experimental.multihost_utils import host_local_array_to_global_array


# ---------------------------------------------------------------------------
# Distributed init + host-0-gated logging
# ---------------------------------------------------------------------------

_IS_HOST0: bool = True                # set to False on workers after _init_distributed()
_DISTRIBUTED_INITIALIZED: bool = False
_ckpt_bg_thread: "threading.Thread | None" = None


def _init_distributed() -> tuple[int, int]:
    """
    Initialize JAX distributed if multi-host env vars are set.

    Set these before launching on every host:
        JAX_COORDINATOR_ADDRESS  e.g. "192.168.1.10:1234"  (host-0 IP:port)
        JAX_NUM_PROCESSES        total number of hosts in the pod slice
        JAX_PROCESS_ID           this host's rank (0-based)

    Safe to call multiple times; no-op after first call.
    Returns (process_index, process_count).
    """
    global _DISTRIBUTED_INITIALIZED
    if not _DISTRIBUTED_INITIALIZED:
        coord = os.environ.get("JAX_COORDINATOR_ADDRESS", "")
        if coord:
            n_proc  = int(os.environ.get("JAX_NUM_PROCESSES", 1))
            proc_id = int(os.environ.get("JAX_PROCESS_ID", 0))
            jax.distributed.initialize(
                coordinator_address=coord,
                num_processes=n_proc,
                process_id=proc_id,
            )
        _DISTRIBUTED_INITIALIZED = True
    return jax.process_index(), jax.process_count()


def _log(*args, **kwargs) -> None:
    """print() gated to host-0 only — avoids duplicate output in multi-host runs."""
    if _IS_HOST0:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------------

def _build_tx(model: DWAModel, tcfg: TrainConfig,
              scheduler: "PhaseScheduler",
              lr_scale: float = 1.0) -> "optax.GradientTransformation":
    """
    Build the optax chain with per-component Adam optimisers.

    lr_scale multiplies the entire LR schedule (all components equally).
    Used by the adaptive LR controller: when a plateau is detected, we
    rebuild only the tx (keeping Adam M/V moments) with a reduced scale.
    Adam moments are LR-independent, so they remain valid after an LR change.
    """
    pure_params = nnx.as_pure(nnx.state(model, nnx.Param))
    leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(pure_params)

    def _label(path: tuple) -> str:
        p = "/".join(str(k) for k in path)
        if "pool" in p:
            return "pool"
        if "tau" in p or "gamma" in p:
            return "threshold"
        if "W_Q" in p or "W_input" in p or "aspect_weights" in p:
            return "retrieval"
        return "parts"

    def _make_mask(target_label: str) -> object:
        flags = [_label(path) == target_label for path, _ in leaves_with_paths]
        return jax.tree_util.tree_unflatten(treedef, flags)

    def _sched(base_lr: float):
        return scheduler.make_optax_schedule(base_lr * lr_scale)

    return optax.chain(
        optax.clip_by_global_norm(tcfg.grad_clip_norm),
        optax.masked(optax.adam(_sched(tcfg.lr_pool)),       _make_mask("pool")),
        optax.masked(optax.adam(_sched(tcfg.lr_threshold)),  _make_mask("threshold")),
        optax.masked(optax.adam(_sched(tcfg.lr_retrieval)),  _make_mask("retrieval")),
        optax.masked(optax.adam(_sched(tcfg.lr_parts)),      _make_mask("parts")),
    )


def _build_optimizer(model: DWAModel, tcfg: TrainConfig,
                     scheduler: "PhaseScheduler") -> nnx.Optimizer:
    """
    Per-component learning rates via chained masked Adam optimizers,
    each with an auto-scheduled LR (linear warmup → constant → cosine decay).

    The schedule is a JAX array lookup keyed by optax's internal step counter,
    so it stays on-device and works correctly inside jax.lax.scan and after
    checkpoint resume (optax step counter is restored with the opt_state).
    """
    tx = _build_tx(model, tcfg, scheduler, lr_scale=1.0)
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

def _make_train_window(cfg: DWAConfig, tcfg: TrainConfig, is_warmup: bool, aux_on: bool,
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
    gate_mix is passed as a per-step JAX array through the scan so it can
    vary each window without triggering recompilation.
    """

    @functools.partial(nnx.jit, static_argnames={})
    def train_window(
        model: DWAModel,
        optimizer: nnx.Optimizer,
        data: jnp.ndarray,           # [steps_per_window, B, seq_len]
        lambda_vals: jnp.ndarray,    # [steps_per_window]
        gate_mix_vals: jnp.ndarray,  # [steps_per_window]
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
            batch, lam, gate_mix = xs  # batch: [B, seq_len], lam/gate_mix: scalars

            def loss_fn(m):
                return forward_and_loss(
                    m, batch, lam, is_warmup, tcfg, aux_on,
                    key_cache=key_cache, use_pallas=use_pallas, mesh=mesh,
                    gate_mix=gate_mix,
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
            jax.lax.scan(step_fn, init_carry, (data, lambda_vals, gate_mix_vals))
        )

        # ── EMA centroid update ───────────────────────────────────────────────
        # Only update when IVF is actually in use (model-sharded pools disable
        # IVF because each device has N_local keys, so centroid update with
        # positional partitions would corrupt centroids with partial data).
        model_sharded = (
            mesh is not None
            and "model" in mesh.axis_names
            and mesh.shape["model"] > 1
        )
        if cfg.use_ivf and not model_sharded:
            key_cache_new = compute_key_cache(
                model.pool.vectors[...].astype(jnp.float32),
                model.pool.key_proj[...].astype(jnp.float32),
            )  # [S, N, d_k]
            N_per_C = cfg.N // cfg.C
            new_c = key_cache_new.reshape(
                cfg.S, cfg.C, N_per_C, cfg.d_k
            ).mean(axis=2)  # [S, C, d_k]
            old_c = model.retrieval.centroids[...]
            model.retrieval.centroids[...] = 0.9 * old_c + 0.1 * new_c

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
# HuggingFace split auto-detection
# ---------------------------------------------------------------------------

def _detect_hf_splits(hf_path: str, hf_subset: str = "") -> tuple[str, str | None]:
    """
    Probe the dataset for available splits and return (train_split, val_split).

    val_split is None when no validation/test split is found — validation loss
    will be skipped in that case.  Falls back to ("train", None) on any error.
    """
    try:
        from datasets import get_dataset_split_names
        kwargs = {"config_name": hf_subset} if hf_subset else {}
        splits = get_dataset_split_names(hf_path, **kwargs)
    except Exception as e:
        _log(f"[Loader] Could not detect splits for {hf_path!r}: {e} — assuming 'train'")
        return "train", None

    train_candidates = ["train", "training"]
    val_candidates   = ["validation", "valid", "val", "dev", "test"]

    train_split = next((s for s in train_candidates if s in splits), splits[0])
    val_split   = next((s for s in val_candidates   if s in splits), None)

    _log(f"[Loader] {hf_path!r} splits={splits}  → train={train_split!r}  val={val_split!r}")
    return train_split, val_split


# ---------------------------------------------------------------------------
# Grain data loader — multi-host sharding + exact preemption resume
# ---------------------------------------------------------------------------

class HFParquetSource:
    """
    Grain RandomAccessDataSource backed by HuggingFace Hub Parquet shards.

    Uses HTTP byte-range requests via HfFileSystem — fetches one row group
    (~1000 rows, few MB) per cache miss.  No full dataset download required.

    File-level sharding: each host gets every shard_count-th Parquet file
    (round-robin), so hosts see non-overlapping subsets of the dataset.
    Grain's IndexSampler then shuffles and repeats within each host's view.
    """

    def __init__(
        self,
        repo_id: str,
        split: str,
        text_column: str = "text",
        hf_subset: str = "",
        shard_index: int = 0,
        shard_count: int = 1,
    ):
        import bisect as _bisect
        import threading as _threading
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        self._bisect   = _bisect
        self._col      = text_column
        self._lock     = _threading.Lock()
        self._cache: dict[tuple, list] = {}

        fs = HfFileSystem()
        base = f"datasets/{repo_id}"
        # Try patterns in priority order — handles repos with different layouts:
        #   1. data/{subset}/{split}-*.parquet  (subset-in-subdir, split prefix)
        #   2. data/{subset}/*.parquet          (subset-in-subdir, no split prefix)
        #   3. data/{split}-*.parquet           (flat, split prefix)
        #   4. data/*.parquet                   (flat, no split prefix)
        #   5. data/**/*.parquet                (recursive fallback)
        candidate_patterns = []
        if hf_subset:
            candidate_patterns += [
                f"{base}/data/{hf_subset}/{split}-*.parquet",
                f"{base}/data/{hf_subset}/*.parquet",
            ]
        candidate_patterns += [
            f"{base}/data/{split}-*.parquet",
            f"{base}/data/*.parquet",
            f"{base}/data/**/*.parquet",
        ]
        all_paths: list[str] = []
        for pattern in candidate_patterns:
            all_paths = sorted(fs.glob(pattern))
            if all_paths:
                _log(f"[Grain] HFParquetSource: matched pattern {pattern!r}")
                break
        if not all_paths:
            raise FileNotFoundError(
                f"No Parquet shards found for {repo_id!r} split={split!r} "
                f"(subset={hf_subset!r}). Tried: {candidate_patterns}"
            )

        my_paths = all_paths[shard_index::shard_count]
        _log(f"[Grain] HFParquetSource: {len(my_paths)}/{len(all_paths)} shards "
             f"for process {shard_index}/{shard_count}")

        # Footer metadata cache — avoids re-fetching N footer HTTP requests on
        # every startup.  Keyed by a hash of (repo, subset, split, shard assignment).
        import hashlib, json
        _cache_key = hashlib.md5(
            f"{repo_id}|{hf_subset}|{split}|{shard_index}|{shard_count}".encode()
        ).hexdigest()
        _meta_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "hfparquet")
        _meta_cache_path = os.path.join(_meta_cache_dir, f"{_cache_key}.json")

        self._fs       = fs
        self._paths    = my_paths
        self._pq       = pq
        self._pf_cache: dict[int, object] = {}   # lazily opened ParquetFile objects

        if os.path.exists(_meta_cache_path):
            _log(f"[Grain] HFParquetSource: loading footer metadata from cache …")
            with open(_meta_cache_path) as f:
                meta = json.load(f)
            self._rg_starts: list[int] = meta["rg_starts"]
            self._rg_map:    list[list] = meta["rg_map"]
            self._total: int            = meta["total"]
        else:
            _log(f"[Grain] HFParquetSource: reading {len(my_paths)} Parquet footers "
                 f"(one-time, will cache) …")
            self._rg_starts = []
            self._rg_map    = []
            total = 0
            for pf_idx, path in enumerate(my_paths):
                pf = pq.ParquetFile(fs.open(path))
                self._pf_cache[pf_idx] = pf
                for rg in range(pf.num_row_groups):
                    self._rg_starts.append(total)
                    self._rg_map.append([pf_idx, rg])
                    total += pf.metadata.row_group(rg).num_rows
            self._total = total
            os.makedirs(_meta_cache_dir, exist_ok=True)
            with open(_meta_cache_path, "w") as f:
                json.dump({"rg_starts": self._rg_starts,
                           "rg_map":    self._rg_map,
                           "total":     self._total}, f)
            _log(f"[Grain] HFParquetSource: footer metadata cached → {_meta_cache_path}")

        _log(f"[Grain] HFParquetSource: {self._total:,} total rows  "
             f"{len(self._rg_map)} row groups")

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> str:
        rg_pos = self._bisect.bisect_right(self._rg_starts, idx) - 1
        pf_idx, rg_idx = self._rg_map[rg_pos]
        row_in_rg = idx - self._rg_starts[rg_pos]

        key = (pf_idx, rg_idx)
        with self._lock:
            if key not in self._cache:
                # Lazily open ParquetFile on first access (avoids 512 HTTP
                # footer requests at startup — only files actually used are opened)
                if pf_idx not in self._pf_cache:
                    self._pf_cache[pf_idx] = self._pq.ParquetFile(
                        self._fs.open(self._paths[pf_idx])
                    )
                # Single HTTP byte-range request for this row group only
                table = self._pf_cache[pf_idx].read_row_group(rg_idx, columns=[self._col])
                self._cache[key] = table.column(self._col).to_pylist()
        return self._cache[key][row_in_rg]


class GrainLoader:
    """
    Grain-based data loader for TPU training.

    Versus TinyStoriesLoader, this adds:
    - Multi-host sharding: each process receives a non-overlapping shard of the dataset
      via grain.ShardOptions — no duplicated data across hosts.
    - Exact resume: grain iterator state (bytes) + remaining token buffer survive
      preemption; restoring both gives the exact same token sequence as if training
      had never been interrupted.
    - Deterministic shuffling: seed + shard → reproducible order across restarts.

    Requires: pip install grain-nightly  (grain >= 0.2)
    """

    FETCH_SIZE = 8_000  # stories per bulk fetch

    def __init__(
        self,
        tokenizer,
        seq_len: int,
        hf_path: str = "roneneldan/TinyStories",
        hf_subset: str = "",
        text_column: str = "text",
        split: str = "train",
        process_index: int = 0,
        process_count: int = 1,
        seed: int = 42,
        worker_count: int = 4,
    ):
        import grain.python as grain

        self.tokenizer      = tokenizer
        self.seq_len        = seq_len
        self.eos            = tokenizer.eos_token_id or 0
        self._process_index = process_index
        self._process_count = process_count

        # HFParquetSource: byte-range HTTP requests, no full download.
        # File-level sharding is handled inside HFParquetSource (round-robin by
        # shard_index/shard_count), so grain.ShardOptions sees shard_count=1.
        self._source = HFParquetSource(
            repo_id=hf_path,
            split=split,
            text_column=text_column,
            hf_subset=hf_subset,
            shard_index=process_index,
            shard_count=process_count,
        )

        sampler = grain.IndexSampler(
            num_records=len(self._source),
            shard_options=grain.ShardOptions(
                shard_index=0,
                shard_count=1,
                drop_remainder=True,
            ),
            shuffle=True,
            num_epochs=None,   # repeat indefinitely
            seed=seed,
        )
        self._loader = grain.DataLoader(
            data_source=self._source,
            sampler=sampler,
            worker_count=worker_count,
        )
        self._it = iter(self._loader)

        self._buf     = np.empty((0, seq_len), dtype=np.int32)
        self._buf_pos = 0

        self._refill()
        _log(f"[Grain] Loader ready: {len(self._buf)} packed seqs")

    def _fetch_and_pack(self) -> np.ndarray:
        texts = []
        for _ in range(self.FETCH_SIZE):
            try:
                t = next(self._it)
            except StopIteration:
                # grain with num_epochs=None should never stop; reset defensively
                self._it = iter(self._loader)
                t = next(self._it)
            texts.append(t if isinstance(t, str) else str(t))

        ids_list = self.tokenizer(
            texts, add_special_tokens=False,
            return_attention_mask=False, return_token_type_ids=False,
        )["input_ids"]

        total = sum(len(s) + 2 for s in ids_list)
        flat  = np.empty(total, dtype=np.int32)
        pos   = 0
        for ids in ids_list:
            n = len(ids)
            flat[pos]           = self.eos
            flat[pos + 1: pos + 1 + n] = ids
            flat[pos + 1 + n]   = self.eos
            pos += n + 2

        n_seqs = pos // self.seq_len
        return flat[: n_seqs * self.seq_len].reshape(n_seqs, self.seq_len)

    def _refill(self) -> None:
        leftover      = self._buf[self._buf_pos:]
        new_seqs      = self._fetch_and_pack()
        self._buf     = np.concatenate([leftover, new_seqs], axis=0) if len(leftover) else new_seqs
        self._buf_pos = 0
        _log(f"[Grain] Refill: buf={len(self._buf)} seqs")

    def get_window(self, steps: int, batch_size: int) -> np.ndarray:
        needed = steps * batch_size
        while len(self._buf) - self._buf_pos < needed:
            self._refill()
        out           = self._buf[self._buf_pos: self._buf_pos + needed]
        self._buf_pos += needed
        return out.reshape(steps, batch_size, self.seq_len)

    def set_state(self, state: dict) -> None:
        """
        Restore exact loader position from a checkpoint state dict.

        state may come in two formats:
          Orbax format:  {"grain_mngr": CheckpointManager, "step": int, "buf": array}
          Legacy format: {"grain_bytes": bytes, "buf": array}  (older checkpoints)
        """
        grain_mngr = state.get("grain_mngr")
        if grain_mngr is not None:
            # Restore grain iterator via PyGrainCheckpointHandler (orbax-managed)
            import grain.python as grain
            new_it = iter(self._loader)
            try:
                grain_mngr.restore(
                    state["step"],
                    args=ocp.args.Composite(grain=grain.PyGrainCheckpointRestore(new_it)),
                )
                self._it = new_it
            except Exception as e:
                _log(f"[Grain] Warning: orbax restore failed ({e}); "
                     f"starting from beginning of shard")
                self._it = iter(self._loader)
        else:
            # Legacy bytes-based restore
            grain_bytes = state.get("grain_bytes")
            if grain_bytes is not None:
                try:
                    new_it = iter(self._loader)
                    new_it.set_state(bytes(grain_bytes))
                    self._it = new_it
                except Exception as e:
                    _log(f"[Grain] Warning: grain state restore failed ({e}); "
                         f"starting from beginning of shard")
                    self._it = iter(self._loader)
            else:
                self._it = iter(self._loader)

        buf_data = state.get("buf")
        if buf_data is not None and len(buf_data):
            arr = np.asarray(buf_data, dtype=np.int32)
            if arr.ndim == 2:
                self._buf = arr
            else:
                n = len(arr) // self.seq_len
                self._buf = arr[: n * self.seq_len].reshape(n, self.seq_len)
        else:
            self._buf = np.empty((0, self.seq_len), dtype=np.int32)
        self._buf_pos = 0
        _log(f"[Grain] Loader restored: {len(self._buf)} packed seqs buffered")


# ---------------------------------------------------------------------------
# HuggingFace streaming loader (fallback when grain is not installed)
# ---------------------------------------------------------------------------

class TinyStoriesLoader:
    """
    Streaming HuggingFace dataset loader with background prefetching.

    Fetches FETCH_SIZE rows at a time via ds.iter(batch_size=FETCH_SIZE) —
    no full dataset download.  Tokenizes in a background thread while training
    runs so get_window() never blocks after the first fill.
    """

    FETCH_SIZE = 50_000  # rows per bulk fetch

    def __init__(self, tokenizer, seq_len: int,
                 hf_path: str = "roneneldan/TinyStories",
                 hf_subset: str = "",
                 text_column: str = "text",
                 split: str = "train"):
        from datasets import load_dataset
        self.tokenizer   = tokenizer
        self.seq_len     = seq_len
        self.hf_path     = hf_path
        self.hf_subset   = hf_subset
        self.text_column = text_column
        self.eos         = tokenizer.eos_token_id or 0

        load_kwargs = {"name": hf_subset} if hf_subset else {}
        self.ds          = load_dataset(hf_path, split=split, streaming=True, **load_kwargs)
        self._batch_iter = self.ds.iter(batch_size=self.FETCH_SIZE)
        self._buf        = np.empty((0, seq_len), dtype=np.int32)
        self._buf_pos    = 0

        # Prefetch state — written by worker thread, read by main thread
        self._pf_buf:  np.ndarray | None = None
        self._pf_event = threading.Event()

        _log(f"[DWA] HF loader (streaming): {hf_path!r}  "
             f"col={text_column!r}  seq_len={seq_len}  fetch={self.FETCH_SIZE:,}")

        # First blocking fill so training can start immediately
        self._buf = self._fetch_and_pack()
        _log(f"[DWA] Loader ready: {len(self._buf)} seqs")

        # Kick off prefetch for the second chunk right away
        self._start_prefetch()

    def _fetch_and_pack(self) -> np.ndarray:
        """Fetch FETCH_SIZE rows via streaming; return packed [N, seq_len]."""
        try:
            batch = next(self._batch_iter)
        except StopIteration:
            self._batch_iter = self.ds.iter(batch_size=self.FETCH_SIZE)
            batch = next(self._batch_iter)

        texts = batch[self.text_column]  # already a list of FETCH_SIZE strings

        ids_list = self.tokenizer(
            texts, add_special_tokens=False,
            return_attention_mask=False, return_token_type_ids=False,
        )["input_ids"]

        # Wrap each story: <|endoftext|> story tokens <|endoftext|>
        total = sum(len(s) + 2 for s in ids_list)  # +2: BOS + EOS per story
        flat  = np.empty(total, dtype=np.int32)
        pos   = 0
        for ids in ids_list:
            n = len(ids)
            flat[pos]               = self.eos
            flat[pos + 1 : pos + 1 + n] = ids
            flat[pos + 1 + n]       = self.eos
            pos += n + 2

        n_seqs = pos // self.seq_len
        return flat[: n_seqs * self.seq_len].reshape(n_seqs, self.seq_len)

    def _start_prefetch(self) -> None:
        """Launch a daemon thread to prepare the next chunk in the background."""
        self._pf_event.clear()

        def _worker():
            self._pf_buf = self._fetch_and_pack()
            self._pf_event.set()

        threading.Thread(target=_worker, daemon=True).start()

    def _refill(self) -> None:
        """Swap in the prefetched buffer (blocks only if prefetch isn't done yet)."""
        self._pf_event.wait()          # almost always instant — prefetch took ~4s, training ~8s

        leftover      = self._buf[self._buf_pos :]
        new_seqs      = self._pf_buf
        self._buf     = np.concatenate([leftover, new_seqs], axis=0) if len(leftover) else new_seqs
        self._buf_pos = 0
        _log(f"[DWA] Loader: buf={len(self._buf)} seqs")

        # Start the next prefetch immediately while training continues
        self._start_prefetch()

    def get_window(self, steps: int, batch_size: int) -> np.ndarray:
        needed = steps * batch_size
        while len(self._buf) - self._buf_pos < needed:
            self._refill()
        out           = self._buf[self._buf_pos : self._buf_pos + needed]
        self._buf_pos += needed
        return out.reshape(steps, batch_size, self.seq_len)

    def state_dict(self) -> dict:
        return {"buf": self._buf[self._buf_pos :]}

    def load_state_dict(self, state: dict) -> None:
        # Wait for any in-flight prefetch to finish before overwriting state
        self._pf_event.wait()
        buf_data = state.get("buf")
        if buf_data is not None and len(buf_data):
            arr = np.asarray(buf_data, dtype=np.int32)
            if arr.ndim == 2:
                self._buf = arr
            else:
                n = len(arr) // self.seq_len
                self._buf = arr[: n * self.seq_len].reshape(n, self.seq_len)
        else:
            self._buf = np.empty((0, self.seq_len), dtype=np.int32)
        self._buf_pos = 0
        _log(f"[DWA] Loader restored ({len(self._buf)} seqs buffered; streaming position not preserved)")
        self._start_prefetch()


# ---------------------------------------------------------------------------
# StreamingLoader — streaming HF dataset, auto-scaled fetch, background prefetch
# ---------------------------------------------------------------------------

class StreamingLoader:
    """
    Primary data loader for HuggingFace datasets.

    Design goals (all three must hold together):
      1. No full download — HF streaming iter(), never materialises the whole dataset.
      2. One fetch covers ≥1 full training window — FETCH_SIZE auto-scaled to
         steps_per_window × batch_size × seq_len / avg_story_tokens.
      3. Zero stalls — background daemon thread packs the next batch while the
         TPU trains the current window, so get_window() almost never waits.

    Multi-host: ds.shard(num_shards=process_count, index=process_index) gives
    each host a non-overlapping, shuffled slice of the dataset — same guarantee
    as grain.ShardOptions without requiring the full dataset to be downloaded.

    Checkpoint / resume:
      Saves (buf, stories_consumed).  On resume, the stream is re-built and
      .skip(stories_consumed) is called to reach approximately the same position.
      This is approximate (not bit-exact) — the important invariant is that the
      packed-token buffer is restored exactly, so the next get_window() returns
      the same sequences that would have followed without interruption.
    """

    _AVG_STORY_TOKENS = 150   # conservative estimate; used only for FETCH_SIZE tuning

    def __init__(
        self,
        tokenizer,
        seq_len: int,
        steps_per_window: int,
        batch_size: int,                          # per-host batch size
        hf_path: str = "roneneldan/TinyStories",
        hf_subset: str = "",
        text_column: str = "text",
        split: str = "train",
        process_index: int = 0,
        process_count: int = 1,
        seed: int = 42,
    ):
        self.tokenizer      = tokenizer
        self.seq_len        = seq_len
        self.eos            = tokenizer.eos_token_id or 0
        self.hf_path        = hf_path
        self.hf_subset      = hf_subset
        self.text_column    = text_column
        self._train_split   = split
        self.process_index  = process_index
        self.process_count  = process_count
        self.seed           = seed

        # Scale FETCH_SIZE so one fetch covers ≥2 full windows.
        # Formula: seqs_needed = 2 × steps × batch;  stories = seqs × seq_len / avg_tokens
        seqs_2_windows = 2 * steps_per_window * batch_size
        self._fetch_size = max(10_000, seqs_2_windows * seq_len // self._AVG_STORY_TOKENS)

        _log(f"[Loader] Streaming {hf_path!r}  shard {process_index}/{process_count}  "
              f"seq_len={seq_len}  fetch={self._fetch_size:,} stories/batch "
              f"(covers ≈{seqs_2_windows:,} seqs = 2 windows)")

        self._stories_consumed: int = 0
        self._buf     = np.empty((0, seq_len), dtype=np.int32)
        self._buf_pos = 0
        self._pf_buf: np.ndarray | None = None
        self._pf_event = threading.Event()

        self._ds_iter = self._make_iter()

        # Blocking first fill — ensures data is ready before the first window
        self._buf = self._fetch_and_pack()
        _log(f"[Loader] Ready: {len(self._buf):,} seqs buffered")

        # Kick off background prefetch immediately so window-2 data is ready on time
        self._start_prefetch()

    def _make_iter(self, skip: int = 0):
        from datasets import load_dataset
        load_kwargs = {"name": self.hf_subset} if self.hf_subset else {}
        ds = load_dataset(self.hf_path, split=self._train_split, streaming=True, **load_kwargs)
        if self.process_count > 1:
            ds = ds.shard(num_shards=self.process_count, index=self.process_index)
        ds = ds.shuffle(seed=self.seed, buffer_size=10_000)
        if skip > 0:
            ds = ds.skip(skip)
        return iter(ds)

    def _fetch_and_pack(self) -> np.ndarray:
        """Fetch _fetch_size stories from the stream; pack into [N, seq_len] int32."""
        texts: list[str] = []
        while len(texts) < self._fetch_size:
            try:
                row = next(self._ds_iter)
                texts.append(row[self.text_column])
                self._stories_consumed += 1
            except StopIteration:
                # Dataset exhausted — restart for indefinite training
                self._ds_iter = self._make_iter()
                self._stories_consumed = 0

        ids_list = self.tokenizer(
            texts, add_special_tokens=False,
            return_attention_mask=False, return_token_type_ids=False,
        )["input_ids"]

        total = sum(len(s) + 2 for s in ids_list)
        flat  = np.empty(total, dtype=np.int32)
        pos   = 0
        for ids in ids_list:
            n = len(ids)
            flat[pos]                  = self.eos
            flat[pos + 1 : pos + 1 + n] = ids
            flat[pos + 1 + n]          = self.eos
            pos += n + 2

        n_seqs = pos // self.seq_len
        return flat[: n_seqs * self.seq_len].reshape(n_seqs, self.seq_len)

    def _start_prefetch(self) -> None:
        self._pf_event.clear()
        def _worker():
            self._pf_buf = self._fetch_and_pack()
            self._pf_event.set()
        threading.Thread(target=_worker, daemon=True).start()

    def _refill(self) -> None:
        """Swap in the prefetched buffer; log a warning if it wasn't ready yet."""
        if not self._pf_event.is_set():
            _log("[Loader] Waiting for prefetch — FETCH_SIZE may be too small for this config")
        self._pf_event.wait()
        leftover      = self._buf[self._buf_pos:]
        new_seqs      = self._pf_buf
        self._buf     = np.concatenate([leftover, new_seqs], axis=0) if len(leftover) else new_seqs
        self._buf_pos = 0
        self._start_prefetch()

    def get_window(self, steps: int, batch_size: int) -> np.ndarray:
        needed = steps * batch_size
        while len(self._buf) - self._buf_pos < needed:
            self._refill()
        out           = self._buf[self._buf_pos : self._buf_pos + needed]
        self._buf_pos += needed
        return out.reshape(steps, batch_size, self.seq_len)

    def state_dict(self) -> dict:
        # Wait for any in-flight prefetch: ensures _pf_buf is fully written
        # and _stories_consumed is stable before we snapshot state.
        self._pf_event.wait()
        remaining = self._buf[self._buf_pos:]
        # Include the already-prefetched buffer so the restored checkpoint has
        # data from both the current and next fetch batch.  Without this, resume
        # does skip(stories_consumed) which skips past _pf_buf stories that are
        # then absent from the restored buf, causing a data gap.
        if self._pf_buf is not None and len(self._pf_buf):
            full_buf = np.concatenate([remaining, self._pf_buf], axis=0)
        else:
            full_buf = remaining
        return {
            "buf":              full_buf,
            "stories_consumed": np.array(self._stories_consumed, dtype=np.int64),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore loader state from a checkpoint dict.  Stream position is approximate."""
        self._pf_event.wait()   # let any in-flight prefetch finish first
        skip = int(state.get("stories_consumed", 0))
        if skip > 0:
            _log(f"[Loader] Resuming: skipping ≈{skip:,} stories to restore stream position …")
        self._stories_consumed = skip
        self._ds_iter = self._make_iter(skip=skip)
        buf_data = state.get("buf")
        if buf_data is not None and len(buf_data):
            arr = np.asarray(buf_data, dtype=np.int32)
            if arr.ndim == 2:
                self._buf = arr
            else:
                n = len(arr) // self.seq_len
                self._buf = arr[: n * self.seq_len].reshape(n, self.seq_len)
        else:
            self._buf = np.empty((0, self.seq_len), dtype=np.int32)
        self._buf_pos = 0
        self._start_prefetch()
        _log(f"[Loader] Restored: {len(self._buf):,} seqs buffered, prefetch started")


# ---------------------------------------------------------------------------
# Validation cache — fixed validation set preloaded once at startup
# ---------------------------------------------------------------------------

class _ValCache:
    """
    Preloads the HuggingFace 'validation' split once and caches a fixed
    numpy array of shape [val_batches, batch_size, seq_len] for repeated use.
    Because the array is fixed, every val evaluation measures the exact same
    distribution — making train vs val curves directly comparable over time.
    """

    def __init__(self, tokenizer, seq_len: int, hf_path: str, text_column: str,
                 val_batches: int, batch_size: int,
                 hf_subset: str = "", split: str = "validation"):
        from datasets import load_dataset
        load_kwargs = {"name": hf_subset} if hf_subset else {}
        ds    = load_dataset(hf_path, split=split, streaming=True, **load_kwargs)
        eos   = tokenizer.eos_token_id or 0
        batch = next(iter(ds.iter(batch_size=100_000)))
        texts = batch[text_column]

        ids_list = tokenizer(
            texts, add_special_tokens=False,
            return_attention_mask=False, return_token_type_ids=False,
        )["input_ids"]

        total = sum(len(s) + 2 for s in ids_list)  # +2: BOS + EOS per story
        flat  = np.empty(total, dtype=np.int32)
        pos   = 0
        for ids in ids_list:
            n = len(ids)
            flat[pos]               = eos              # BOS  <|endoftext|>
            flat[pos + 1 : pos + 1 + n] = ids
            flat[pos + 1 + n]       = eos              # EOS  <|endoftext|>
            pos += n + 2

        available_seqs = pos // seq_len
        available_batches = available_seqs // batch_size
        if available_batches < val_batches:
            _log(f"[DWA] Val cache: only {available_seqs} seqs available, clamping val_batches {val_batches}→{available_batches}")
            val_batches = available_batches
        if val_batches == 0:
            raise ValueError(f"Val split too small: only {available_seqs} seqs, need at least {batch_size} for one batch")
        needed = val_batches * batch_size
        self.data = flat[: needed * seq_len].reshape(val_batches, batch_size, seq_len)
        self.val_batches = val_batches
        self.batch_size  = batch_size
        self.seq_len     = seq_len
        _log(f"[DWA] Val cache: {val_batches}×{batch_size}×{seq_len}  "
              f"({needed} seqs from {len(texts)} val stories)")


@nnx.jit
def _val_step_jit(model: "DWAModel", batch: jnp.ndarray, lam: float) -> jnp.ndarray:
    """Single forward pass for validation — no gradients, no aux losses."""
    logits, _ = model(batch, lam, is_warmup=False, use_pallas=False)
    return task_loss(logits, batch)


def _compute_val_loss(
    model: "DWAModel",
    val_cache: _ValCache,
    lam: float,
    mesh,
) -> float:
    """Average task loss over all cached validation batches."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    total = 0.0
    for i in range(val_cache.val_batches):
        batch = jnp.array(val_cache.data[i])                         # [B, seq_len]
        batch = jax.device_put(batch, NamedSharding(mesh, P("data", None)))
        total += float(_val_step_jit(model, batch, lam))
    return total / val_cache.val_batches


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

def _gen_scan_body(model, buf, pos_init, key_cache, n_new, lam, temperature, log_rep, vocab_size, rng):
    """
    Run n_new autoregressive decode steps via jax.lax.scan.

    buf is RIGHT-padded: prefix occupies [0, pos_init), zeros fill [pos_init, seq_len).
    Causal attention means the zeros are never attended to from valid positions,
    so they don't corrupt the logits we actually use.

    Two phases tracked by `pos`:
      fill  (pos < seq_len): write next token at pos, increment pos
      slide (pos == seq_len): drop oldest token, append at end (sliding window)
    """
    seq_len = buf.shape[1]

    def step(carry, _):
        tokens_buf, pos, rng = carry

        logits, _ = model(tokens_buf, lam, is_warmup=False,
                          key_cache=key_cache, use_pallas=False)

        # Read logits at the last valid position
        read_pos = jnp.minimum(pos - 1, seq_len - 1)
        logit_vec = logits[0, read_pos].astype(jnp.float32)

        # Repetition penalty on last 64 tokens in the live context
        seen = jnp.zeros(vocab_size).at[tokens_buf[0, -64:]].add(1.0)
        logit_vec -= jnp.clip(seen, 0.0, 1.0) * log_rep
        logit_vec /= temperature

        rng, k = jax.random.split(rng)
        next_tok = jax.random.categorical(k, logit_vec)

        # Fill while there is room; slide once the window is full
        new_buf_fill  = tokens_buf.at[0, pos].set(next_tok)
        new_buf_slide = jnp.concatenate([tokens_buf[:, 1:], next_tok[None, None]], axis=1)
        new_buf = jnp.where(pos < seq_len, new_buf_fill, new_buf_slide)
        new_pos = jnp.minimum(pos + 1, seq_len)

        return (new_buf, new_pos, rng), next_tok

    (_, _, _), new_toks = jax.lax.scan(step, (buf, pos_init, rng), None, length=n_new)
    return new_toks


# Compiled once; reused across all generate() calls (model state updates are
# tracked automatically by nnx.jit).
_jit_gen_scan = nnx.jit(_gen_scan_body, static_argnames=("n_new", "vocab_size"))


def generate(
    model: DWAModel,
    prefix_ids,               # list[int] or 1-D array
    n_new: int,
    tcfg: TrainConfig,
    temperature: float = 0.8,
    repetition_penalty: float = 1.3,
    eos_token_id: int | None = None,
) -> list[int]:
    """
    Temperature-sampled generation.

    The entire n_new-step decode loop runs inside a single nnx.jit + jax.lax.scan
    call — no host↔device transfer per token, no re-tracing, key cache computed once.

    If eos_token_id is given, the returned list is truncated at the first EOS token
    (inclusive) so the caller gets a clean, complete story.
    """
    cfg = model.cfg
    seq_len = cfg.seq_len
    tokens = list(map(int, prefix_ids))
    prefix_len = min(len(tokens), seq_len)

    # Right-pad: prefix at [0, prefix_len), zeros at [prefix_len, seq_len).
    # Causal masking means the trailing zeros never affect logits at valid positions.
    buf = jnp.zeros((1, seq_len), dtype=jnp.int32)
    buf = buf.at[0, :prefix_len].set(jnp.array(tokens[-prefix_len:], dtype=jnp.int32))
    pos_init = jnp.array(prefix_len, dtype=jnp.int32)

    key_cache = model.pool.compute_keys()

    new_toks = _jit_gen_scan(
        model, buf, pos_init, key_cache, n_new,
        float(tcfg.lambda_sharpen_end),
        temperature,
        float(jnp.log(repetition_penalty)),
        cfg.vocab_size,
        jax.random.PRNGKey(0),
    )

    result = tokens + [int(t) for t in new_toks]

    # Truncate at the first EOS in the newly generated portion
    if eos_token_id is not None:
        for i, tok in enumerate(result[len(tokens):], start=len(tokens)):
            if tok == eos_token_id:
                return result[:i + 1]   # include the EOS token itself

    return result


def _verify_learning(
    model: DWAModel, cfg: DWAConfig, tcfg: TrainConfig
) -> None:
    """Print pattern-learning accuracy for 3 random repeat-period test cases."""
    _log(f"\n[DWA] Learning verification (period-{PATTERN_PERIOD} repeat pattern, "
          f"vocab={cfg.vocab_size}):")
    _log(f"  Theoretical min loss ≈ "
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
        _log(f"  [{ok}] pattern={pattern}  gen={gen_new}  exp={exp_new}  acc={acc:.0%}")


# ---------------------------------------------------------------------------
# Text generation helper for TinyStories learning verification
# ---------------------------------------------------------------------------

def _generate_text_sample(model: DWAModel, tokenizer, tcfg: TrainConfig, step: int,
                          prompts: list[str] | None = None) -> None:
    """Generate up to 120 tokens from one or more prompts, stopping at EOS."""
    eos = tokenizer.eos_token_id
    for prompt in (prompts or ["Once upon a time"]):
        ids  = [eos] + tokenizer.encode(prompt)
        out  = generate(model, ids, n_new=120, tcfg=tcfg, eos_token_id=eos)
        text = tokenizer.decode(out[1:], skip_special_tokens=True)
        _log(f"\n[DWA] step={step} prompt={prompt!r}: {text}\n")


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
    tcfg: TrainConfig,
    step: int,
) -> tuple[int, jnp.ndarray]:
    """Replace pool vectors that have near-zero EMA usage.

    Revival strategy: copy donor vectors + small noise (factor=0.2) to keep
    vector norms stable, then boost the revived vectors' EMA to ~1e-3 so they
    survive ~700 steps without being selected. Without the EMA boost, revived
    vectors stay at EMA=0 forever (only hard-selected indices update EMA) and
    get classified as dead again at the next revival check regardless of noise.

    Returns (n_revived, updated_pool_ema).
    """
    ema_np = np.array(pool_ema, dtype=np.float32)
    dead_mask = ema_np < tcfg.dead_vector_threshold
    n_dead = int(dead_mask.sum())
    if n_dead == 0:
        return 0, pool_ema

    # Donors: top-50% by EMA usage (avoids picking just-revived vectors)
    median_ema = float(np.median(ema_np[ema_np > 0])) if (ema_np > 0).any() else 0.0
    donor_idx = np.where(ema_np >= median_ema)[0]
    if len(donor_idx) == 0:
        return 0, pool_ema

    dead_idx = np.where(dead_mask)[0]
    rng      = np.random.default_rng(step)

    # Cap revivals to avoid massive simultaneous parameter perturbations.
    # Revive the most-dead vectors first (lowest EMA → most starved of gradient).
    max_revive = max(1, int(cfg.N * tcfg.max_revival_frac))
    if n_dead > max_revive:
        dead_ema = ema_np[dead_idx]
        dead_idx = dead_idx[np.argsort(dead_ema)[:max_revive]]
        n_dead = max_revive

    chosen_donors = rng.choice(donor_idx, size=n_dead, replace=True)

    if cfg.use_hypernetwork:
        # Revive at the coordinate embeddings level
        emb_np = np.array(model.pool.embeddings[...], dtype=np.float32)
        donor_norm = float(np.linalg.norm(emb_np[chosen_donors], axis=-1).mean()) + 1e-8
        noise = rng.normal(0.0, tcfg.revival_noise_factor * donor_norm / (cfg.d_emb ** 0.5),
                           (n_dead, cfg.d_emb)).astype(emb_np.dtype)
        emb_np[dead_idx] = emb_np[chosen_donors] + noise

        orig_arr = model.pool.embeddings[...]
        new_jax  = jnp.array(emb_np, dtype=orig_arr.dtype)
        orig_sharding = getattr(orig_arr, "sharding", None)
        if orig_sharding is not None:
            new_jax = jax.device_put(new_jax, orig_sharding)
        model.pool.embeddings[...] = new_jax
    else:
        # Standard pool revival
        pool_np  = np.array(model.pool.vectors[...], dtype=np.float32)
        donor_norm = float(np.linalg.norm(pool_np[chosen_donors], axis=-1).mean()) + 1e-8
        noise = rng.normal(0.0, tcfg.revival_noise_factor * donor_norm / (cfg.d_k ** 0.5),
                           (n_dead, cfg.D)).astype(pool_np.dtype)
        pool_np[dead_idx] = pool_np[chosen_donors] + noise

        orig_arr = model.pool.vectors[...]
        new_jax  = jnp.array(pool_np, dtype=orig_arr.dtype)
        orig_sharding = getattr(orig_arr, "sharding", None)
        if orig_sharding is not None:
            new_jax = jax.device_put(new_jax, orig_sharding)
        model.pool.vectors[...] = new_jax

    # Boost EMA for revived vectors so they survive ~700 steps (1e-3 / (1-0.99) = 0.1
    # effective selection rate) before next revival check, giving L_reuse time to
    # route queries to them.  Without this, EMA stays at 0 (only hard-selected indices
    # update EMA) and they die immediately at the next revival check.
    ema_np[dead_idx] = 1e-3
    return n_dead, jnp.array(ema_np)


# ---------------------------------------------------------------------------
# Checkpoint save / restore (orbax-based)
# ---------------------------------------------------------------------------

def _get_ckpt_manager(ckpt_dir: str, keep: int = 3) -> "ocp.CheckpointManager":
    os.makedirs(ckpt_dir, exist_ok=True)
    return ocp.CheckpointManager(
        ckpt_dir,
        options=ocp.CheckpointManagerOptions(max_to_keep=keep),
    )


def _get_grain_ckpt_manager(ckpt_dir: str, process_index: int,
                             keep: int = 3) -> "ocp.CheckpointManager":
    """
    Per-host CheckpointManager for Grain iterator state.

    Each host owns a different dataset shard, so grain state is host-local.
    Stored at ckpt_dir/grain_p{pid}/ — separate from the shared JAX checkpoint.
    """
    import grain.python as grain
    grain_dir = os.path.join(ckpt_dir, f"grain_p{process_index}")
    os.makedirs(grain_dir, exist_ok=True)
    return ocp.CheckpointManager(
        grain_dir,
        item_handlers={"grain": grain.PyGrainCheckpointHandler()},
        options=ocp.CheckpointManagerOptions(max_to_keep=keep),
    )


def _take_ckpt_snapshot(model, optimizer, pool_ema, rng, loader) -> dict:
    """Extract all mutable training state to pure numpy. Must be called synchronously."""
    model_np = jax.tree_util.tree_map(np.array, nnx.state(model, nnx.Param))
    opt_leaves, _ = jax.tree_util.tree_flatten(optimizer.opt_state)
    opt_dict = {f"{i:04d}": np.array(leaf) for i, leaf in enumerate(opt_leaves)}
    snap: dict = {
        "model_np": model_np,
        "opt_dict": opt_dict,
        "opt_step": int(optimizer.step[...]),
        "pool_ema": np.array(pool_ema, dtype=np.float32),
        "rng":      np.array(rng),
    }
    if loader is not None:
        if isinstance(loader, GrainLoader):
            snap["grain_it"]  = loader._it          # grain serializes state at save() time
            snap["grain_buf"] = loader._buf[loader._buf_pos:].copy()
        else:
            snap["loader_state"] = loader.state_dict()
    return snap


def save_checkpoint(
    ckpt_dir: str,
    model: DWAModel,
    optimizer: nnx.Optimizer,
    pool_ema: jnp.ndarray,
    steps_done: int,
    rng: jax.Array,
    loader,
    keep: int = 3,
    process_index: int = 0,
    lr_scale: float = 1.0,
    _snapshot: "dict | None" = None,
) -> None:
    """
    Save a full training checkpoint (multi-host aware).

    JAX arrays (model, optimizer, metadata) go through the shared orbax
    CheckpointManager — all hosts must call this; orbax coordinates internally.

    GrainLoader state is split into two atomic pieces:
      1. Grain iterator position → per-host orbax manager (grain.PyGrainCheckpointHandler)
         at ckpt_dir/grain_p{pid}/  — saved atomically via orbax
      2. Packed token buffer (already-fetched tokens not yet consumed) → npz alongside
         orbax step dir at ckpt_dir/{step}/loader_buf_{pid}.npz

    TinyStoriesLoader: buffer only, host-0, backward compat.
    """
    mngr = _get_ckpt_manager(ckpt_dir, keep)

    if _snapshot is not None:
        model_np = _snapshot["model_np"]
        opt_dict = _snapshot["opt_dict"]
        _opt_step = _snapshot["opt_step"]
        _pool_ema = _snapshot["pool_ema"]
        _rng      = _snapshot["rng"]
    else:
        model_np = jax.tree_util.tree_map(np.array, nnx.state(model, nnx.Param))
        opt_leaves, _ = jax.tree_util.tree_flatten(optimizer.opt_state)
        opt_dict = {f"{i:04d}": np.array(leaf) for i, leaf in enumerate(opt_leaves)}
        _opt_step = int(optimizer.step[...])
        _pool_ema = np.array(pool_ema, dtype=np.float32)
        _rng      = np.array(rng)

    save_item = {
        "model": model_np,
        "opt":   opt_dict,
        "meta": {
            "opt_step":   np.array(_opt_step, dtype=np.int32),
            "pool_ema":   _pool_ema,
            "steps_done": np.array(steps_done, dtype=np.int32),
            "rng":        _rng,
        },
    }

    # ALL hosts call this — orbax coordinates JAX array shard ownership.
    mngr.save(steps_done, args=ocp.args.StandardSave(save_item))
    mngr.wait_until_finished()

    # Loader state — each host saves independently (different shard per host).
    if loader is not None:
        step_dir = os.path.join(ckpt_dir, str(steps_done))
        os.makedirs(step_dir, exist_ok=True)
        if isinstance(loader, RollingParquetLoader):
            # Per-host npz: file_index + grain iterator bytes + packed token buffer
            state = _snapshot["loader_state"] if _snapshot else loader.state_dict()
            grain_bytes = state["grain_bytes"]
            np.savez(
                os.path.join(step_dir, f"loader_state_{process_index}.npz"),
                file_index=np.array(state["file_index"], dtype=np.int64),
                grain_bytes=np.frombuffer(grain_bytes, dtype=np.uint8) if grain_bytes else np.empty(0, dtype=np.uint8),
                buf=state["buf"],
            )
        elif isinstance(loader, StreamingLoader):
            # Per-host npz: packed token buffer + approximate stream position
            state = _snapshot["loader_state"] if _snapshot else loader.state_dict()
            np.savez(
                os.path.join(step_dir, f"loader_state_{process_index}.npz"),
                buf=state["buf"],
                stories_consumed=state["stories_consumed"],
            )
        elif isinstance(loader, GrainLoader):
            # 1. Grain iterator position via PyGrainCheckpointHandler (atomic, orbax-managed)
            import grain.python as grain
            grain_mngr = _get_grain_ckpt_manager(ckpt_dir, process_index, keep)
            _grain_it  = _snapshot["grain_it"] if _snapshot else loader._it
            grain_mngr.save(
                steps_done,
                args=ocp.args.Composite(grain=grain.PyGrainCheckpointSave(_grain_it)),
            )
            grain_mngr.wait_until_finished()
            # 2. Remaining packed token buffer (not tracked by grain — saved alongside)
            buf_path  = os.path.join(step_dir, f"loader_buf_{process_index}.npz")
            _grain_buf = _snapshot["grain_buf"] if _snapshot else loader._buf[loader._buf_pos:].copy()
            np.savez(buf_path, buf=_grain_buf)
        elif process_index == 0:
            # TinyStoriesLoader: only host 0 (backward compat)
            state = _snapshot["loader_state"] if _snapshot else loader.state_dict()
            loader_path = os.path.join(step_dir, "loader_state.npz")
            save_kwargs = {"buf": state["buf"]}
            if "cursor" in state:
                save_kwargs["cursor"] = np.array(state["cursor"], dtype=np.int32)
            np.savez(loader_path, **save_kwargs)

    # Save adaptive LR scale alongside the checkpoint (host-0 only, plain text).
    # Used on resume to prevent the adaptive controller from restarting at 1.0
    # after a preemption, which caused oscillating effective LR in prior runs.
    if process_index == 0:
        lr_scale_path = os.path.join(ckpt_dir, str(steps_done), "lr_scale.txt")
        with open(lr_scale_path, "w") as _f:
            _f.write(str(lr_scale))

    _log(f"[Ckpt] Saved step {steps_done} → {ckpt_dir}/{steps_done}/")


def _abstract_like(item):
    """Recursively build an abstract (shape+dtype) version of a numpy pytree."""
    return jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, item)


def _orbax_to_numpy(arr: np.ndarray) -> np.ndarray:
    """Convert void-dtype arrays that orbax uses for bfloat16 storage to ml_dtypes.bfloat16.

    Orbax serialises bfloat16 arrays as numpy |V2 (2-byte void) because numpy has
    no native bfloat16 dtype.  JAX rejects |V2 outright, so we reinterpret the
    raw bytes via ml_dtypes.bfloat16 (a numpy-compatible extension type that JAX
    accepts).  Any other dtype is returned unchanged.
    """
    if arr.dtype.kind == 'V':
        import ml_dtypes as _mld
        itemsize = arr.dtype.itemsize
        if itemsize == 2:
            return arr.view(_mld.bfloat16)
        # Fallback for other void sizes: view as unsigned int of same width
        return arr.view(np.dtype(f'u{itemsize}'))
    return arr


def load_checkpoint(
    ckpt_dir: str,
    steps_done_target: int,
    model: DWAModel,
    optimizer: nnx.Optimizer,
    cfg: DWAConfig,
    mesh,
    process_index: int = 0,
) -> tuple[int, jax.Array, dict | None, jnp.ndarray, float]:
    """
    Restore model, optimizer, and metadata from a checkpoint (multi-host aware).

    ALL hosts must call this. orbax restores each host's shard of the JAX arrays.
    Loader state is loaded per-host (grain_state_{pid}.bin + loader_buf_{pid}.npz),
    with fallback to the legacy loader_state.npz format.

    Returns (steps_done, rng, loader_state_dict, pool_ema, lr_scale).
    lr_scale is the saved adaptive LR controller multiplier (1.0 if not found in checkpoint).
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
        },
    }

    restored = mngr.restore(steps_done_target, args=ocp.args.StandardRestore(abstract))

    # --- Restore model parameters ---
    n_model = mesh.shape.get("model", 1) if mesh is not None else 1
    pool_sharding = NamedSharding(mesh, P("model", None)) if (mesh is not None and n_model > 1) else None

    if pool_sharding is not None:
        if "vectors" in restored["model"]["pool"]:
            # Re-shard pool vectors without materialising the full array on one device:
            # slice each device's rows and put them directly.
            pool_np  = _orbax_to_numpy(np.array(restored["model"]["pool"]["vectors"]))
            N_local  = pool_np.shape[0] // n_model
            idx_map  = pool_sharding.addressable_devices_indices_map(pool_np.shape)
            per_dev  = []
            for dev in pool_sharding.addressable_devices:
                rows = idx_map[dev][0]
                shard = jax.device_put(
                    jnp.array(pool_np[rows]).astype(model.pool.vectors[...].dtype), dev
                )
                per_dev.append(shard)
            sharded_pool = jax.make_array_from_single_device_arrays(
                pool_np.shape, pool_sharding, per_dev
            )
            # Update model params without pool first, then fix pool
            nnx.update(model, restored["model"])
            model.pool.vectors[...] = sharded_pool
        elif "embeddings" in restored["model"]["pool"]:
            # Re-shard coordinate embeddings without materialising the full array on one device:
            emb_np  = _orbax_to_numpy(np.array(restored["model"]["pool"]["embeddings"]))
            N_local  = emb_np.shape[0] // n_model
            emb_sharding = NamedSharding(mesh, P("model", None))
            idx_map  = emb_sharding.addressable_devices_indices_map(emb_np.shape)
            per_dev  = []
            for dev in emb_sharding.addressable_devices:
                rows = idx_map[dev][0]
                shard = jax.device_put(
                    jnp.array(emb_np[rows]).astype(model.pool.embeddings[...].dtype), dev
                )
                per_dev.append(shard)
            sharded_emb = jax.make_array_from_single_device_arrays(
                emb_np.shape, emb_sharding, per_dev
            )
            # Update model params
            nnx.update(model, restored["model"])
            model.pool.embeddings[...] = sharded_emb
        else:
            raise KeyError("Neither 'vectors' nor 'embeddings' found in pool checkpoint model state")
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

    # --- Loader state ---
    # Priority: StreamingLoader format > GrainLoader format > legacy TinyStoriesLoader
    step_dir       = os.path.join(ckpt_dir, str(steps_done_target))
    streaming_path = os.path.join(step_dir, f"loader_state_{process_index}.npz")
    buf_path       = os.path.join(step_dir, f"loader_buf_{process_index}.npz")
    grain_dir      = os.path.join(ckpt_dir, f"grain_p{process_index}")

    if os.path.exists(streaming_path):
        d = np.load(streaming_path, allow_pickle=False)
        if "file_index" in d.files:
            # RollingParquetLoader format: file_index + grain_bytes + buf
            raw_bytes = d["grain_bytes"].tobytes() if len(d["grain_bytes"]) > 0 else b""
            loader_state = {
                "file_index":  int(d["file_index"]),
                "grain_bytes": raw_bytes,
                "buf":         d["buf"] if "buf" in d.files else np.empty((0,), dtype=np.int32),
            }
        else:
            # StreamingLoader format: buf + stories_consumed
            loader_state = {
                "buf":              d["buf"] if "buf" in d.files else np.empty((0,), dtype=np.int32),
                "stories_consumed": int(d["stories_consumed"]) if "stories_consumed" in d.files else 0,
            }
    elif os.path.isdir(grain_dir) and os.path.exists(buf_path):
        # Grain loader format: restore iterator via PyGrainCheckpointHandler
        try:
            import grain.python as grain  # noqa: F401
            grain_mngr = _get_grain_ckpt_manager(ckpt_dir, process_index)
            d = np.load(buf_path)
            loader_state = {"grain_mngr": grain_mngr, "step": steps_done_target, "buf": d["buf"]}
        except ImportError:
            loader_state = {"buf": np.empty((0,), dtype=np.int32)}
    else:
        # Legacy TinyStoriesLoader / backward compat
        legacy_path = os.path.join(step_dir, "loader_state.npz")
        if os.path.exists(legacy_path):
            d = np.load(legacy_path)
            loader_state = {"buf": d["buf"]}
            if "cursor" in d.files:
                loader_state["cursor"] = int(d["cursor"])
        else:
            buf_npy = os.path.join(step_dir, "buf.npy")
            buf = np.load(buf_npy) if os.path.exists(buf_npy) else np.empty((0,), dtype=np.int32)
            loader_state = {"buf": buf}

    # Restore adaptive LR scale (written by host-0; all hosts read same value).
    # Falls back to 1.0 for old checkpoints that predate this field.
    lr_scale_path = os.path.join(ckpt_dir, str(steps_done_target), "lr_scale.txt")
    saved_lr_scale = 1.0
    if os.path.exists(lr_scale_path):
        try:
            with open(lr_scale_path) as _f:
                saved_lr_scale = float(_f.read().strip())
        except (ValueError, OSError):
            saved_lr_scale = 1.0

    _log(f"[Ckpt] Loaded step {steps_done} from {ckpt_dir}/{steps_done_target}/")
    return steps_done, rng, loader_state, pool_ema, saved_lr_scale


# ---------------------------------------------------------------------------
# Google Drive checkpoint backup via rclone
# ---------------------------------------------------------------------------

def _rclone_available() -> bool:
    import shutil
    return shutil.which("rclone") is not None


def _rclone_push(local_dir: str, remote: str, remote_path: str, extra_args: str = "") -> None:
    """
    Sync local_dir → remote:remote_path via rclone.  Call from host-0 only.

    Uses 'rclone sync' so the remote mirrors local — checkpoints pruned by
    Orbax are also pruned on the remote.  Skips gracefully if rclone is absent.

    Setup once before training:
        apt-get install -y rclone
        rclone config  # follow prompts to add a 'gdrive' remote
    """
    import subprocess
    if not _rclone_available():
        _log("[GDrive] rclone not found — skipping push. "
             "Install: apt-get install -y rclone && rclone config")
        return
    dest = f"{remote}:{remote_path}"
    cmd  = ["rclone", "sync", local_dir, dest, "--stats-one-line", "-v"]
    if extra_args:
        cmd += extra_args.split()
    _log(f"[GDrive] Pushing {local_dir} → {dest} …")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            _log("[GDrive] Push complete.")
        else:
            _log(f"[GDrive] Push failed (rc={result.returncode}): {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        _log("[GDrive] rclone push timed out (10 min).")


def _rclone_pull(local_dir: str, remote: str, remote_path: str,
                  process_index: int = 0, extra_args: str = "") -> None:
    """
    Sync remote:remote_path → local_dir via rclone.

    Called by every host on preemption resume when the local checkpoint dir is
    empty.  A small per-host stagger (process_index × 3 s) avoids hammering
    the GDrive API from all TPU workers simultaneously.
    """
    import subprocess, time as _time
    if not _rclone_available():
        _log("[GDrive] rclone not found — skipping pull. "
             "Install: apt-get install -y rclone && rclone config")
        return
    if process_index > 0:
        _time.sleep(process_index * 3)   # stagger: 3 s per rank
    os.makedirs(local_dir, exist_ok=True)
    src  = f"{remote}:{remote_path}"
    cmd  = ["rclone", "sync", src, local_dir, "--stats-one-line", "-v"]
    if extra_args:
        cmd += extra_args.split()
    _log(f"[GDrive] Pulling {src} → {local_dir} …")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            _log("[GDrive] Pull complete.")
        else:
            _log(f"[GDrive] Pull failed (rc={result.returncode}): {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        _log("[GDrive] rclone pull timed out (10 min).")


# ---------------------------------------------------------------------------
# Pallas probe — check at startup whether the kernel compiles successfully
# ---------------------------------------------------------------------------

def _probe_pallas(cfg: DWAConfig, mesh: Mesh, local_batch: int) -> bool:
    """
    Try to compile the Pallas assembly kernel via shard_map.
    Uses the actual per-device batch size so VMEM and Mosaic constraints
    match what the training window will encounter.
    """
    from src.dwa.assembly_pallas import pallas_vmem_feasible, shard_pallas_assemble
    kr = cfg.k_max * cfg.r
    elem_bytes = 2 if cfg.compute_dtype == jnp.bfloat16 else 4
    if not pallas_vmem_feasible(local_batch, cfg.seq_len, cfg.d_B, kr, cfg.d_A, elem_bytes):
        _log(f"[DWA] Pallas assembly: not feasible for B_local={local_batch}, "
              f"T={cfg.seq_len}, d={cfg.d_A}, kr={kr} (VMEM/Mosaic constraints)")
        return False
    try:
        B_probe = len(mesh.devices)
        gathered = jax.device_put(
            jnp.zeros((B_probe, cfg.k_max, cfg.D)),
            NamedSharding(mesh, P("data", None, None)),
        )
        alphas = jax.device_put(
            jnp.ones((B_probe, cfg.k_max)) / cfg.k_max,
            NamedSharding(mesh, P("data", None)),
        )
        h_A = jax.device_put(
            jnp.zeros((B_probe, cfg.seq_len, cfg.d_A)),
            NamedSharding(mesh, P("data", None, None)),
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
        _log(f"[DWA] Pallas probe failed: {e}")
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

    _log(f"\n[DWA] Parameter breakdown — {total_n / 1e6:.1f}M params, "
          f"{total_b / 1e6:.0f} MB storage (dtype-aware):")
    _log(f"  {'Component':<{W}}  {'Params':>10}  {'Storage':>10}  {'Share':>6}")
    _log(f"  {'─' * W}  {'─' * 10}  {'─' * 10}  {'─' * 6}")
    for label, _ in GROUPS:
        n, b = counts[label], mbytes[label]
        if n == 0:
            continue
        _log(f"  {label:<{W}}  {n / 1e6:>8.3f} M  {b / 1e6:>7.1f} MB  {100 * n / total_n:>5.1f}%")
    if other_n > 0:
        _log(f"  {'(unmatched)':<{W}}  {other_n / 1e6:>8.3f} M")
    _log(f"  {'─' * W}  {'─' * 10}  {'─' * 10}  {'─' * 6}")
    _log(f"  {'TOTAL':<{W}}  {total_n / 1e6:>8.3f} M  {total_b / 1e6:>7.1f} MB  100.0%\n")


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
    global _IS_HOST0

    # Must be called before any JAX computation; safe no-op if already initialized.
    process_index, process_count = _init_distributed()
    _IS_HOST0 = (process_index == 0)

    cfg  = run_cfg.model
    tcfg = run_cfg.train

    # --- Data source setup (before model build so vocab_size is set) ---
    tokenizer   = None
    use_pattern = False
    _train_split: str       = "train"
    _val_split:   str | None = None
    if run_cfg.data.source in ("tiny_stories", "hf"):
        from transformers import AutoTokenizer
        tok_name  = run_cfg.data.hf_tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        # Pad vocab to nearest multiple of 64 so vocab_parallel sharding
        # divides evenly for any n_model in {1,2,4,8}.
        cfg.vocab_size = ((tokenizer.vocab_size + 63) // 64) * 64
        _log(f"[DWA] HF dataset mode: tokenizer={tok_name!r}  "
             f"padded vocab_size={cfg.vocab_size}")
        _train_split, _val_split = _detect_hf_splits(
            run_cfg.data.hf_path, run_cfg.data.hf_subset
        )
    elif run_cfg.data.source == "pattern":
        use_pattern = True

    gen_every  = run_cfg.data.gen_every
    ckpt_dir   = run_cfg.checkpoint.dir
    ckpt_every = run_cfg.checkpoint.every
    resume     = run_cfg.checkpoint.resume

    # jax.devices() returns ALL devices across all hosts after distributed init.
    devices = jax.devices()
    n_devices = len(devices)
    device_kind = devices[0].device_kind

    # --- Mesh: 2D (data × model) ---
    n_model = _select_n_model(cfg, n_devices, run_cfg.sharding.n_model)
    n_data = n_devices // n_model
    mesh = _build_mesh(devices, n_model)

    sharding_src = ("auto" if run_cfg.sharding.n_model == "auto"
                    else f"explicit n_model={run_cfg.sharding.n_model}")
    _log(f"[DWA] Training on {n_devices}× {device_kind}  "
         f"({process_count} host{'s' if process_count > 1 else ''})")
    _log(f"[DWA] Mesh: {n_data}×data  {n_model}×model  "
          f"(pool+Adam/device ≈ {cfg.N * cfg.D * 4 * 3 / n_model / 1e9:.1f} GB)"
          f"  [{sharding_src}]")
    _log(f"[DWA] Model config: N={cfg.N}, D={cfg.D}, d_A={cfg.d_A}, r={cfg.r}, "
          f"layers={cfg.n_layers_A}+{cfg.n_layers_B}, vocab={cfg.vocab_size}")

    # Per data-replica batch size
    assert tcfg.batch_size % n_data == 0, "batch_size must be divisible by n_data"
    local_batch = tcfg.batch_size // n_data

    # Per-host batch: sequences this host's local devices need to supply.
    # In multi-host mode each host provides its own slice of the global batch.
    n_local_devices = jax.local_device_count()
    n_data_local    = n_local_devices // n_model
    per_host_batch  = local_batch * n_data_local   # == batch_size for single-host

    scheduler = PhaseScheduler(tcfg)
    lambda_array   = scheduler.make_lambda_array()                                           # [total_steps]
    gate_mix_array = jnp.array([scheduler.gate_mix(s) for s in range(tcfg.total_steps)])    # [total_steps]

    # Initialise model.  For configs needing model parallelism (n_model > 1),
    # create the pool directly on each device's HBM before building the model
    # — this avoids materialising the full ~4 GB pool on device 0 first.
    # The optimizer is built after so Adam states inherit the pool's sharding.
    rng = jax.random.PRNGKey(tcfg.seed)
    sharded_pool = _make_sharded_pool_vectors(cfg, mesh, rng)  # None if n_model==1
    model = DWAModel(cfg, nnx.Rngs(rng), pool_vectors=sharded_pool)
    optimizer = _build_optimizer(model, tcfg, scheduler)
    pool_ema = jnp.zeros(cfg.N)

    _print_param_table(model)

    step_flops = compute_step_flops(cfg, tcfg)
    data_label = run_cfg.data.hf_path if tokenizer is not None else ("pattern" if use_pattern else "random")
    _log(f"[DWA] FLOPs/step (fwd+bwd, approx): {step_flops / 1e9:.1f}G"
          f"  (data={data_label})")

    # Data loader selection:
    #   "grain"     — GrainLoader: HF Hub Parquet byte-range requests, exact resume
    #                 via PyGrainCheckpointHandler.  Requires: pip install grain-nightly
    #   "streaming" — StreamingLoader: HF streaming API, no full download, approximate resume.
    loader = None
    if tokenizer is not None:
        _want_grain = (run_cfg.data.loader == "grain")
        if _want_grain:
            try:
                import grain.python as _grain  # noqa: F401
                loader = RollingParquetLoader(
                    tokenizer, cfg.seq_len,
                    repo_id=run_cfg.data.hf_path,
                    split=_train_split,
                    hf_subset=run_cfg.data.hf_subset,
                    text_column=run_cfg.data.hf_text_column,
                    shard_index=process_index,
                    shard_count=process_count,
                    seed=tcfg.seed,
                )
                s = loader.stats()
                _log(f"[Loader] RollingParquetLoader ready: "
                     f"file={s['file_idx']}/{s['n_files']}  "
                     f"queue={s['queue_size']}/{s['queue_max']}  "
                     f"buf={s['buf_seqs']:,} seqs  "
                     f"next_file_ready={s['dl_ready']}")
            except ImportError:
                _log("[Loader] grain not installed — falling back to StreamingLoader. "
                     "Run: pip install grain-nightly")
                _want_grain = False

        if not _want_grain:
            loader = StreamingLoader(
                tokenizer, cfg.seq_len,
                steps_per_window=tcfg.steps_per_window,
                batch_size=per_host_batch,
                hf_path=run_cfg.data.hf_path,
                hf_subset=run_cfg.data.hf_subset,
                text_column=run_cfg.data.hf_text_column,
                split=_train_split,
                process_index=process_index,
                process_count=process_count,
                seed=tcfg.seed,
            )

    # Validation cache — preloaded once, fixed for the entire run.
    # If val_hf_path is set, use it as a separate validation dataset (auto-detect its splits).
    # Otherwise fall back to the training dataset's detected val split.
    # Skipped entirely if no validation split is available.
    val_every  = run_cfg.data.val_every
    val_cache: "_ValCache | None" = None
    if tokenizer is not None and val_every > 0:
        _val_hf_path   = run_cfg.data.val_hf_path   or run_cfg.data.hf_path
        _val_hf_subset = run_cfg.data.val_hf_subset or run_cfg.data.hf_subset
        _val_text_col  = run_cfg.data.val_hf_text_column or run_cfg.data.hf_text_column
        # Separate val dataset: always auto-detect its train split (use as the val split here).
        # Same dataset: use the already-detected _val_split.
        if run_cfg.data.val_hf_path:
            _effective_val_split, _ = _detect_hf_splits(_val_hf_path, _val_hf_subset)
            _log(f"[Val] Using separate validation dataset: {_val_hf_path!r}  "
                 f"split={_effective_val_split!r}")
        else:
            _effective_val_split = _val_split
        if _effective_val_split is not None:
            val_cache = _ValCache(
                tokenizer, cfg.seq_len,
                hf_path=_val_hf_path,
                text_column=_val_text_col,
                val_batches=run_cfg.data.val_batches,
                batch_size=tcfg.batch_size,
                hf_subset=_val_hf_subset,
                split=_effective_val_split,
            )
        else:
            _log("[Val] No validation split found — validation loss disabled.")

    # Pallas assembly uses custom_vjp: forward runs in VMEM (fused, no HBM
    # writes for delta_W / U / V intermediates), backward is pure-JAX (no
    # VMEM pressure in scan backward).  Probe at startup to confirm the
    # kernel compiles on this device configuration.
    _use_pallas = _probe_pallas(cfg, mesh, local_batch)
    if _use_pallas:
        _log(f"[DWA] Pallas assembly: ENABLED (custom_vjp fused forward, pure-JAX backward)")
    else:
        _log(f"[DWA] Pallas assembly: disabled (probe failed — falling back to pure JAX)")

    # Pre-JIT train windows for each phase
    # Recompilation happens only at phase boundaries (is_warmup changes once).
    # gate_mix is now a per-step JAX array threaded through the scan, so it
    # never appears in the compile key.
    compiled_fns: dict[tuple, object] = {}

    def get_train_fn(is_warmup: bool, aux_on: bool):
        key = (is_warmup, aux_on)
        if key not in compiled_fns:
            compiled_fns[key] = _make_train_window(
                cfg, tcfg, is_warmup, aux_on,
                use_pallas=_use_pallas,
                mesh=mesh,
            )
        return compiled_fns[key]

    n_windows  = tcfg.total_steps // tcfg.steps_per_window
    steps_done = 0
    start_window = 0
    rng = jax.random.PRNGKey(tcfg.seed + 1)

    # --- GDrive pull: fetch remote checkpoint when local dir has nothing ---
    gdrive_cfg = run_cfg.gdrive
    if (resume and ckpt_dir
            and gdrive_cfg.enabled
            and gdrive_cfg.pull_on_resume
            and gdrive_cfg.remote_path):
        mngr_early = _get_ckpt_manager(ckpt_dir)
        if mngr_early.latest_step() is None:
            _log("[GDrive] No local checkpoint — pulling from GDrive before resume …")
            _rclone_pull(
                ckpt_dir,
                gdrive_cfg.rclone_remote,
                gdrive_cfg.remote_path,
                process_index=process_index,
                extra_args=gdrive_cfg.rclone_args,
            )

    # --- Resume from checkpoint ---
    if resume and ckpt_dir:
        mngr_probe = _get_ckpt_manager(ckpt_dir)
        latest = mngr_probe.latest_step()
        if latest is not None:
            steps_done, rng, loader_state, pool_ema, _ckpt_lr_scale = load_checkpoint(
                ckpt_dir, latest, model, optimizer, cfg, mesh,
                process_index=process_index,
            )
            model.pool_ema[...] = pool_ema
            start_window = steps_done // tcfg.steps_per_window
            if loader is not None and loader_state is not None:
                loader.load_state_dict(loader_state)
                if isinstance(loader, RollingParquetLoader):
                    s = loader.stats()
                    _log(f"[Loader] Resumed: file={s['file_idx']}/{s['n_files']}  "
                         f"buf={s['buf_seqs']:,} seqs")
            # Restore adaptive LR scale and rebuild optimizer.tx to match.
            # For checkpoints written before lr_scale was added (old format), fall
            # back to 1.0 so the cosine schedule alone controls the LR magnitude.
            _current_lr_scale = _ckpt_lr_scale
            if abs(_ckpt_lr_scale - 1.0) > 1e-4:
                _log(f"[Ckpt] Restored lr_scale={_ckpt_lr_scale:.4f} from checkpoint")
            _log(f"[Ckpt] Resuming from step {steps_done} (window {start_window}/{n_windows})"
                 f"  lr_scale={_current_lr_scale:.4f}")
        else:
            _log(f"[Ckpt] No checkpoint in '{ckpt_dir}', starting fresh.")

    _log(f"[DWA] Training for {tcfg.total_steps} steps "
          f"({n_windows} windows × {tcfg.steps_per_window} steps)")
    if ckpt_dir:
        _log(f"[DWA] Checkpointing: dir='{ckpt_dir}'  every={ckpt_every} steps  keep=3")
    _log(f"[DWA] Safety: grad_clip={tcfg.grad_clip_norm}  "
          f"nan_stop={tcfg.nan_emergency_stop}w  "
          f"spike_sigma={tcfg.loss_spike_sigma}σ  "
          f"revival_every={tcfg.revival_interval_steps}s")

    # ── Weights & Biases — host-0 only ───────────────────────────────────────
    _wb = None
    _wb_log_every = run_cfg.wandb.log_every
    if run_cfg.wandb.enabled and _IS_HOST0:
        try:
            import wandb as _wandb_mod
            _wb = _wandb_mod.init(
                project  = run_cfg.wandb.project,
                entity   = run_cfg.wandb.entity   or None,
                name     = run_cfg.wandb.name     or run_cfg.name,
                tags     = run_cfg.wandb.tags     or None,
                notes    = run_cfg.wandb.notes    or None,
                config   = to_dict(run_cfg),
                resume   = "allow",
                id       = run_cfg.wandb.name     or run_cfg.name,
            )
            url_str = _wb.url or f"offline ({_wb.dir})"
            _log(f"[W&B]  Run: {url_str}")
        except ImportError:
            _log("[W&B]  wandb not installed — skipping. Run: pip install wandb")

    def _wlog(metrics: dict, step: int) -> None:
        if _wb is not None:
            _wb.log(metrics, step=step)

    # Steady-state tracking
    # _win_times_rolling: all recent window durations (rolling window of 10).
    # Compile detection: a window is "compile" if it is >3× the minimum of the
    # last 10 durations.  Bootstrapping: the first 2 windows are always excluded
    # so that long compile windows don't set a falsely high baseline.
    _ss_steps, _ss_time = 0, 0.0
    _win_times_rolling: deque = deque(maxlen=10)

    # Safety state
    _consecutive_nan = 0
    _loss_window: deque = deque(maxlen=50)   # rolling buffer for spike detection

    # Last computed validation loss (updated every val_every steps)
    _last_val_loss: float | None = None

    # Advanced pool collapse detector and adaptive LR controller.
    # LR controller: when a plateau/divergence is detected, optimizer.tx is
    # rebuilt with a new lr_scale multiplier (Adam M/V moments are preserved
    # since they are LR-independent; only the schedule magnitude changes).
    collapse_detector = PoolCollapseDetector(cfg.N, cfg.k_max)
    lr_ctrl           = LossAdaptiveLRController(floor=0.5)
    # _current_lr_scale is set by the checkpoint load block above (restoring
    # the saved adaptive scale); default to 1.0 only on a fresh start.
    _current_lr_scale = locals().get("_current_lr_scale", 1.0)
    if abs(_current_lr_scale - 1.0) > 1e-4:
        # Optimizer was built with lr_scale=1.0 at init; rebuild with restored scale.
        optimizer.tx = _build_tx(model, tcfg, scheduler, _current_lr_scale)

    t0 = time.time()
    for window_idx in range(start_window, n_windows):
        start_step = window_idx * tcfg.steps_per_window
        phase = scheduler.get_phase(start_step)
        is_warmup = scheduler.is_warmup(start_step)
        aux_on = scheduler.aux_enabled(start_step)

        # Slice lambda and gate_mix schedules for this window
        lam_window = lambda_array[start_step: start_step + tcfg.steps_per_window]
        gm_window  = gate_mix_array[start_step: start_step + tcfg.steps_per_window]

        # Build this host's local data slice [steps, per_host_batch, seq_len].
        # In multi-host mode each host independently provides its own non-overlapping
        # shard; host_local_array_to_global_array assembles the global sharded array.
        rng, data_rng = jax.random.split(rng)
        # Fold in process_index so synthetic data differs across hosts.
        host_data_rng = jax.random.fold_in(data_rng, process_index)
        if loader is not None:
            data_local = jnp.array(loader.get_window(tcfg.steps_per_window, per_host_batch))
        elif use_pattern:
            data_local = _make_pattern_window(
                host_data_rng, tcfg.steps_per_window, per_host_batch,
                cfg.seq_len, cfg.vocab_size,
            )
        else:
            data_local = _synthetic_window(
                host_data_rng, tcfg.steps_per_window, per_host_batch,
                cfg.seq_len, cfg.vocab_size,
            )

        # Assemble global sharded array from per-host local slices.
        # For single-host this is equivalent to the old jax.device_put approach.
        data_sharded = host_local_array_to_global_array(
            data_local, mesh, P(None, "data", None)
        )

        train_fn = get_train_fn(is_warmup, aux_on)
        t_win = time.time()
        model, optimizer, pool_ema, info = train_fn(
            model, optimizer, data_sharded, lam_window, gm_window, pool_ema, tcfg.ema_decay,
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
        _win_times_rolling.append(win_secs)
        recent_min = min(_win_times_rolling)
        # A window is compile if fewer than 3 windows seen yet, or if this
        # window took >3× the fastest recent window (XLA recompilation spike).
        is_compile_win = (len(_win_times_rolling) < 3
                          or win_secs > 3.0 * recent_min)
        if not is_compile_win:
            _ss_steps += tcfg.steps_per_window
            _ss_time  += win_secs
        val_str = f"  val={_last_val_loss:.4f}" if _last_val_loss is not None else ""
        lr_base = scheduler.get_lr_scale(start_step)
        lr_eff  = lr_base * _current_lr_scale
        lr_str  = (f"lr={lr_base:.3f}×{_current_lr_scale:.3f}={lr_eff:.4f}"
                   if abs(_current_lr_scale - 1.0) > 0.001 else f"lr={lr_base:.3f}")
        _log(
            f"[DWA] step={steps_done:6d}/{tcfg.total_steps} "
            f"phase={phase:8s} λ={scheduler.get_lambda(start_step):.2f} "
            f"{lr_str} "
            f"loss={mean_loss:.4f}{val_str} "
            f"win={win_secs:.1f}s  steps/s={win_steps_per_sec:.1f}  "
            f"tok/s={win_tok_per_sec:,}  TFLOP/s={achieved_tflops:.1f}"
            + ("  [compile]" if is_compile_win else "")
        )
        # Loader health — every 10 windows so it's visible but not spammy
        if isinstance(loader, RollingParquetLoader) and window_idx % 10 == 0:
            s = loader.stats()
            stall = "" if s["queue_size"] > 0 else "  ⚠ queue empty"
            _log(f"[Loader] file={s['file_idx']}/{s['n_files']}  "
                 f"queue={s['queue_size']}/{s['queue_max']}  "
                 f"buf={s['buf_seqs']:,} seqs  "
                 f"next_dl={'ready' if s['dl_ready'] else 'downloading'}"
                 f"{stall}")

        # ── Safety checks ────────────────────────────────────────────────────

        # 1. NaN / Inf in losses / gradients
        nan_count  = int(info["nan_flags"].sum())
        mean_gnorm = float(info["grad_norms"].mean())
        max_gnorm  = float(info["grad_norms"].max())
        if nan_count > 0:
            _consecutive_nan += 1
            _log(
                f"[Safety] NaN/Inf: {nan_count}/{tcfg.steps_per_window} steps "
                f"(grads zeroed) — consecutive bad windows: "
                f"{_consecutive_nan}/{tcfg.nan_emergency_stop}"
            )
            if _consecutive_nan >= tcfg.nan_emergency_stop:
                _log("[Safety] EMERGENCY STOP: too many consecutive NaN windows.")
                break
        else:
            _consecutive_nan = 0

        # 2. Loss spike detector (rolling mean ± σ over last 50 windows)
        if len(_loss_window) >= 10:
            mu    = float(np.mean(_loss_window))
            sigma = float(np.std(_loss_window)) + 1e-8
            if mean_loss > mu + tcfg.loss_spike_sigma * sigma:
                _log(
                    f"[Safety] Loss spike: {mean_loss:.4f} vs "
                    f"rolling {mu:.4f} ± {sigma:.4f} "
                    f"({(mean_loss - mu) / sigma:.1f}σ)"
                )
        _loss_window.append(mean_loss)

        # 3. Periodic parameter NaN check (every 10 windows — host-side scan)
        if window_idx % 10 == 0:
            has_bad, bad_name = _check_nan_params(model)
            if has_bad:
                _log(f"[Safety] CRITICAL: NaN/Inf in parameter '{bad_name}'. Stopping.")
                break

        # 4. Pool-collapse detector (multi-signal, state machine)
        last_idx      = np.array(info["last_indices"])   # [B, k_max] on host
        collapse_info = collapse_detector.update(np.array(pool_ema), last_idx)
        _log(collapse_detector.format_line(collapse_info) +
              f"  gnorm={mean_gnorm:.3f}(max={max_gnorm:.3f})")
        if collapse_info["changed"]:
            _log(f"[Collapse] State → {collapse_info['state']}  "
                  f"(entropy_slope={collapse_info['entropy_slope']:+.4f}/w  "
                  f"active_slope={collapse_info['active_slope']:+.4f}/w)")
        if "revive_now" in collapse_info["actions"]:
            n_revived, pool_ema = _revive_dead_vectors(model, pool_ema, cfg, tcfg, steps_done)
            _log(f"[Collapse] CRITICAL: immediately revived {n_revived}/{cfg.N} vectors.")

        # 4b. Loss-adaptive LR controller
        lr_ctrl_info = lr_ctrl.update(mean_loss, mean_gnorm)
        _log(lr_ctrl.format_line(lr_ctrl_info))
        if lr_ctrl_info["event"]:
            # Rebuild optimizer.tx with new LR scale (preserves Adam M/V moments).
            # Adam's moments are LR-independent (they track gradient statistics),
            # so carrying them over to the new schedule is mathematically correct.
            # The next window will trigger a JIT recompile (~1× cost, acceptable
            # since LR reductions happen at most ~3 times per training run).
            _current_lr_scale = lr_ctrl_info["lr_scale"]
            optimizer.tx = _build_tx(model, tcfg, scheduler, _current_lr_scale)
            compiled_fns.clear()  # force recompile with new schedule
            _log(f"[LR]   Rebuilt optimizer schedule: ×{_current_lr_scale:.4f} "
                  f"(recompile next window)")

        # Text generation check every gen_every steps
        prev_steps = steps_done - tcfg.steps_per_window

        # 5. Dead vector revival (boundary-crossing check so it fires every
        #    ~revival_interval_steps regardless of steps_per_window alignment)
        _n_revived_this_win = 0
        if (steps_done // tcfg.revival_interval_steps) > (prev_steps // tcfg.revival_interval_steps):
            _n_revived_this_win, pool_ema = _revive_dead_vectors(model, pool_ema, cfg, tcfg, steps_done)
            if _n_revived_this_win > 0:
                _log(f"[Safety] Revived {_n_revived_this_win}/{cfg.N} dead pool vectors.")

        # Validation loss (boundary crossing — fires every val_every steps)
        if val_cache is not None and val_every > 0 and \
                (steps_done // val_every) > (prev_steps // val_every):
            lam_val = float(tcfg.lambda_sharpen_end)
            _last_val_loss = _compute_val_loss(model, val_cache, lam_val, mesh)
            _log(f"[Val]  step={steps_done:6d}  train={mean_loss:.4f}  val={_last_val_loss:.4f}  "
                  f"gap={_last_val_loss - mean_loss:+.4f}")
            _wlog({"val/loss": _last_val_loss, "val/gap": _last_val_loss - mean_loss},
                  step=steps_done)

        # Text generation — host-0 only (model state is replicated so any host could do it)
        if _IS_HOST0 and tokenizer is not None and (steps_done // gen_every) > (prev_steps // gen_every):
            _generate_text_sample(model, tokenizer, tcfg, steps_done, run_cfg.data.gen_prompts)

        # ── W&B per-window log ────────────────────────────────────────────────
        if window_idx % _wb_log_every == 0:
            _collapse_state_int = {"OK": 0, "WARNING": 1, "CRITICAL": 2, "COLLAPSED": 3}
            _wlog({
                # Training
                "train/loss":            mean_loss,
                "train/loss_fast":       lr_ctrl_info["loss_fast"],
                "train/loss_slow":       lr_ctrl_info["loss_slow"],
                "train/grad_norm_mean":  mean_gnorm,
                "train/grad_norm_max":   max_gnorm,
                "train/nan_count":       int(info["nan_flags"].sum()),
                "train/phase":           phase,
                "train/lambda":          scheduler.get_lambda(start_step),
                # Learning rate
                "lr/base_scale":         lr_base,
                "lr/adaptive_scale":     _current_lr_scale,
                "lr/effective":          lr_base * _current_lr_scale,
                "lr/improvement_rate":   lr_ctrl_info["improvement_rate"],
                "lr/plateau_count":      lr_ctrl_info["no_improve"],
                "lr/cooldown":           lr_ctrl_info["cooldown"],
                "lr/n_reductions":       lr_ctrl_info["n_reductions"],
                # Pool collapse
                "pool/entropy":          collapse_info["entropy"],
                "pool/entropy_slope":    collapse_info["entropy_slope"],
                "pool/active_frac":      collapse_info["active_frac"],
                "pool/active_slope":     collapse_info["active_slope"],
                "pool/unique_frac":      collapse_info["unique_frac"],
                "pool/gini":             collapse_info["gini"],
                "pool/top10_conc":       collapse_info["top10_conc"],
                "pool/state":            _collapse_state_int[collapse_info["state"]],
                "pool/revived":          _n_revived_this_win,
                # Performance
                "perf/steps_per_sec":   win_steps_per_sec,
                "perf/tokens_per_sec":  win_tok_per_sec,
                "perf/tflops":          achieved_tflops,
                "perf/window_secs":     win_secs,
                "perf/is_compile":      int(is_compile_win),
            }, step=steps_done)

        # Checkpoint save — snapshot mutable state synchronously, then write in background.
        # Non-JAX side effects (config dump, rclone push) are host-0 only.
        if ckpt_dir and ckpt_every > 0 and (steps_done // ckpt_every) > (prev_steps // ckpt_every):
            global _ckpt_bg_thread
            snap = _take_ckpt_snapshot(model, optimizer, pool_ema, rng, loader)
            _ckpt_dir        = ckpt_dir
            _ckpt_steps      = steps_done
            _ckpt_proc       = process_index
            _ckpt_lr_scale   = _current_lr_scale
            _ckpt_keep       = run_cfg.checkpoint.keep if hasattr(run_cfg.checkpoint, "keep") else 3
            _do_gdrive       = (_IS_HOST0 and gdrive_cfg.enabled
                                and gdrive_cfg.push_on_save and gdrive_cfg.remote_path)
            _gdrive_remote   = gdrive_cfg.rclone_remote
            _gdrive_path     = gdrive_cfg.remote_path
            _gdrive_args     = gdrive_cfg.rclone_args

            if _ckpt_bg_thread is not None and _ckpt_bg_thread.is_alive():
                _log("[Ckpt] Waiting for previous background checkpoint to finish …")
                _ckpt_bg_thread.join()

            def _bg_ckpt():
                save_checkpoint(
                    _ckpt_dir, model, optimizer, pool_ema, _ckpt_steps, rng, loader,
                    keep=_ckpt_keep,
                    process_index=_ckpt_proc,
                    lr_scale=_ckpt_lr_scale,
                    _snapshot=snap,
                )
                if _do_gdrive:
                    _rclone_push(_ckpt_dir, _gdrive_remote, _gdrive_path, extra_args=_gdrive_args)

            if _IS_HOST0:
                cfg_out = os.path.join(ckpt_dir, "effective_config.yaml")
                if not os.path.exists(cfg_out):
                    save_config(run_cfg, cfg_out)

            _ckpt_bg_thread = threading.Thread(target=_bg_ckpt, daemon=True, name="ckpt-bg")
            _ckpt_bg_thread.start()
            _log(f"[Ckpt] Background save started for step {steps_done}.")

        # Update model's pool EMA (non-trainable variable)
        model.pool_ema[...] = pool_ema

    if _ckpt_bg_thread is not None and _ckpt_bg_thread.is_alive():
        _log("[Ckpt] Waiting for final background checkpoint to finish …")
        _ckpt_bg_thread.join()

    elapsed_total = time.time() - t0
    _log(f"[DWA] Training complete in {elapsed_total:.1f}s")
    if _ss_time > 0:
        ss_sps    = _ss_steps / _ss_time
        ss_tflops = step_flops * ss_sps / 1e12
        ss_tok    = int(ss_sps * tcfg.batch_size * cfg.seq_len)
        peak = 197.0 * n_devices if "v5" in device_kind.lower() else None
        mfu  = ss_tflops / peak * 100 if peak else None
        mfu_str = (f"  MFU={mfu:.2f}% (vs {peak:.0f} TFLOP/s BF16 peak)" if peak else "")
        _log(f"[DWA] Steady-state throughput: {ss_sps:.0f} steps/s  "
              f"{ss_tok:,} tok/s  {ss_tflops:.1f} TFLOP/s{mfu_str}")
        compile_secs = elapsed_total - _ss_time
        _log(f"[DWA] Time breakdown: {_ss_time:.1f}s training + {compile_secs:.1f}s XLA compilation")
        if _wb is not None:
            _wb.summary.update({
                "summary/ss_steps_per_sec": ss_sps,
                "summary/ss_tok_per_sec":   ss_tok,
                "summary/ss_tflops":        ss_tflops,
                "summary/mfu_pct":          mfu or 0.0,
                "summary/train_secs":       _ss_time,
                "summary/compile_secs":     compile_secs,
                "summary/total_steps":      steps_done,
                "summary/lr_reductions":    lr_ctrl._n_reductions,
            })

    if _wb is not None:
        _wb.finish()

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
        _log(f"[DWA] Loaded config from '{args.config}' (name={run_cfg.name!r})")
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
        _log("[DWA] BF16 compute enabled: linear layers will run in bfloat16")
    if args.remat:
        run_cfg.model.remat = True
        _log("[DWA] Gradient checkpointing enabled: ~4× less activation memory, ~33% more FLOPs")
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
