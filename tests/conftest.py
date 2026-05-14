"""Shared fixtures for DWA tests."""

import jax._src.config as _jax_cfg

# Patch JAX config before any imports (fixes optax/JAX version mismatch)
_orig = _jax_cfg.config.update


def _safe(n, v):
    try:
        _orig(n, v)
    except AttributeError:
        pass


_jax_cfg.config.update = _safe

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from src.dwa.config import DWAConfig, TrainConfig
from src.dwa.model import DWAModel


@pytest.fixture(scope="session")
def cfg():
    return DWAConfig.small()


@pytest.fixture(scope="session")
def tcfg():
    return TrainConfig()


@pytest.fixture
def model(cfg):
    return DWAModel(cfg, nnx.Rngs(0))


@pytest.fixture
def input_ids(cfg):
    return jnp.ones((4, cfg.seq_len), dtype=jnp.int32)
