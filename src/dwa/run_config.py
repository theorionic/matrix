"""
RunConfig — unified YAML-based configuration for a full training run.

Wraps DWAConfig and TrainConfig with extra sections for sharding, data,
and checkpointing.  Supports loading from YAML and round-tripping back.

Usage:
    from src.dwa.run_config import load_config, save_config

    run_cfg = load_config("configs/full.yaml")
    run_cfg.train.total_steps = 200_000   # CLI override example
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import yaml

from .config import DWAConfig, TrainConfig

# ---------------------------------------------------------------------------
# Extra config sections
# ---------------------------------------------------------------------------

@dataclass
class ShardingConfig:
    """
    Controls how the model is distributed across TPU cores.

    n_model: "auto" | int
        "auto"  — pick smallest n_model such that pool+Adam fits ≤4 GB/device.
        integer — explicit model-parallel degree (must evenly divide n_devices).

    Examples (8-core TPU v5e):
        n_model: auto   → n_model=1 for small/medium, 4 for full
        n_model: 1      → 8-way data parallel only (all 8 cores replicate pool)
        n_model: 4      → 4-way model × 2-way data parallel
        n_model: 8      → 8-way model parallel, no data parallel (1 replica)
    """
    n_model: Any = "auto"   # "auto" | int


@dataclass
class DataConfig:
    """Dataset and generation options."""
    source: str = "random"       # "random" | "tiny_stories" | "pattern"
    hf_path: str = "roneneldan/TinyStories"   # HuggingFace dataset path (tiny_stories mode)
    hf_text_column: str = "text"              # column containing raw text
    gen_every: int = 100         # generate text sample every N steps (tiny_stories only)


@dataclass
class CheckpointConfig:
    """Checkpoint saving and resuming."""
    dir: str = ""           # empty → no checkpointing
    every: int = 1000       # save every N steps
    resume: bool = False    # resume from latest checkpoint in dir
    keep: int = 3           # number of checkpoints to retain


@dataclass
class RunConfig:
    """Full configuration for one training run."""
    model:      DWAConfig       = field(default_factory=DWAConfig.small)
    train:      TrainConfig     = field(default_factory=TrainConfig)
    sharding:   ShardingConfig  = field(default_factory=ShardingConfig)
    data:       DataConfig      = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    name: str = "dwa_run"


# ---------------------------------------------------------------------------
# YAML ↔ Python helpers
# ---------------------------------------------------------------------------

_MODEL_FIELDS  = {f.name for f in dataclasses.fields(DWAConfig)}
_TRAIN_FIELDS  = {f.name for f in dataclasses.fields(TrainConfig)}
_DATA_FIELDS   = {f.name for f in dataclasses.fields(DataConfig)}
_CKPT_FIELDS   = {f.name for f in dataclasses.fields(CheckpointConfig)}


def _parse_compute_dtype(value: str | None):
    """
    Convert a YAML string to a JAX dtype or None.

    Accepted values: null / float32 / bfloat16
    """
    if value is None or value in ("null", "float32", "none", "None"):
        return None
    if value == "bfloat16":
        import jax.numpy as jnp
        return jnp.bfloat16
    raise ValueError(
        f"compute_dtype must be 'float32', 'bfloat16', or null — got {value!r}"
    )


def _dtype_to_str(dtype) -> str:
    """Convert a JAX dtype (or None) to a YAML-safe string."""
    if dtype is None:
        return "float32"
    return str(dtype)    # "bfloat16" for jnp.bfloat16


def _parse_n_model(value) -> int | str:
    if value is None or value == "auto":
        return "auto"
    n = int(value)
    if n not in (1, 2, 4, 8, 16):
        raise ValueError(f"n_model must be 'auto' or a power-of-2 ≤ 16 — got {n}")
    return n


def _build_dwa_config(raw: dict) -> DWAConfig:
    """Build DWAConfig from a raw YAML dict, handling compute_dtype."""
    raw = dict(raw)   # copy so we can pop
    compute_dtype_str = raw.pop("compute_dtype", None)
    unknown = set(raw) - _MODEL_FIELDS
    if unknown:
        raise ValueError(f"Unknown model config keys: {unknown}")
    cfg = DWAConfig(**raw)
    cfg.compute_dtype = _parse_compute_dtype(compute_dtype_str)
    return cfg


def _dwa_config_to_dict(cfg: DWAConfig) -> dict:
    d = dataclasses.asdict(cfg)
    d["compute_dtype"] = _dtype_to_str(cfg.compute_dtype)
    return d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str) -> RunConfig:
    """
    Load a RunConfig from a YAML file.

    Unknown keys raise ValueError to surface typos early.
    All sections are optional — missing sections use defaults.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    # Model
    model_raw = raw.get("model", {})
    model_cfg = _build_dwa_config(model_raw) if model_raw else DWAConfig()

    # Train
    train_raw = raw.get("train", {})
    unknown_t = set(train_raw) - _TRAIN_FIELDS
    if unknown_t:
        raise ValueError(f"Unknown train config keys: {unknown_t}")
    train_cfg = TrainConfig(**train_raw)

    # Sharding
    sharding_raw = raw.get("sharding", {})
    sharding_cfg = ShardingConfig(
        n_model=_parse_n_model(sharding_raw.get("n_model", "auto")),
    )

    # Data
    data_raw = raw.get("data", {})
    unknown_d = set(data_raw) - _DATA_FIELDS
    if unknown_d:
        raise ValueError(f"Unknown data config keys: {unknown_d}")
    data_cfg = DataConfig(**data_raw)

    # Checkpoint
    ckpt_raw = raw.get("checkpoint", {})
    unknown_c = set(ckpt_raw) - _CKPT_FIELDS
    if unknown_c:
        raise ValueError(f"Unknown checkpoint config keys: {unknown_c}")
    ckpt_cfg = CheckpointConfig(**ckpt_raw)

    return RunConfig(
        model=model_cfg,
        train=train_cfg,
        sharding=sharding_cfg,
        data=data_cfg,
        checkpoint=ckpt_cfg,
        name=raw.get("name", "dwa_run"),
    )


def to_dict(run_cfg: RunConfig) -> dict:
    """Serialize a RunConfig to a plain dict (YAML-safe)."""
    return {
        "name": run_cfg.name,
        "model": _dwa_config_to_dict(run_cfg.model),
        "train": dataclasses.asdict(run_cfg.train),
        "sharding": {"n_model": run_cfg.sharding.n_model},
        "data": dataclasses.asdict(run_cfg.data),
        "checkpoint": dataclasses.asdict(run_cfg.checkpoint),
    }


def save_config(run_cfg: RunConfig, path: str) -> None:
    """Write a RunConfig to a YAML file (useful for saving effective config alongside checkpoints)."""
    with open(path, "w") as f:
        yaml.dump(to_dict(run_cfg), f, default_flow_style=False, sort_keys=False, allow_unicode=True)
