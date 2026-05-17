"""Tests for DWAConfig, TrainConfig, and RunConfig YAML round-trips."""

import dataclasses
import os
import tempfile

import pytest

from src.dwa.config import DWAConfig, TrainConfig
from src.dwa.run_config import (
    CheckpointConfig,
    DataConfig,
    RunConfig,
    ShardingConfig,
    load_config,
    save_config,
    to_dict,
)


# ---------------------------------------------------------------------------
# DWAConfig
# ---------------------------------------------------------------------------

class TestDWAConfig:
    def test_small_preset_valid(self):
        cfg = DWAConfig.small()
        assert cfg.D >= cfg.d_B * cfg.r + cfg.r * cfg.d_A + cfg.d_B
        assert cfg.d_A == cfg.d_B
        assert cfg.d_A % cfg.n_heads == 0

    def test_medium_preset_valid(self):
        cfg = DWAConfig.medium()
        assert cfg.D >= cfg.d_B * cfg.r + cfg.r * cfg.d_A + cfg.d_B

    def test_full_preset_valid(self):
        cfg = DWAConfig()
        assert cfg.D >= cfg.d_B * cfg.r + cfg.r * cfg.d_A + cfg.d_B

    def test_pattern_test_preset_valid(self):
        cfg = DWAConfig.pattern_test()
        assert cfg.vocab_size == 16

    def test_ivf_divisibility(self):
        cfg = DWAConfig.small()
        if cfg.use_ivf:
            assert cfg.N % cfg.C == 0

    def test_factor_split_covers_D(self):
        cfg = DWAConfig.small()
        s1, s2, s3 = cfg.factor_split
        assert s1 == cfg.d_B * cfg.r
        assert s2 == s1 + cfg.r * cfg.d_A
        assert s3 == s2 + cfg.d_B
        assert s3 <= cfg.D

    def test_mismatched_hidden_dims_raises(self):
        with pytest.raises(AssertionError, match="d_A must equal d_B"):
            DWAConfig(d_A=64, d_B=128, D=4096, N=128, r=4, C=4)

    def test_insufficient_D_raises(self):
        with pytest.raises(AssertionError, match="D="):
            DWAConfig(D=10, N=64, d_A=64, d_B=64, r=4, C=4)


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------

class TestTrainConfig:
    def test_defaults_valid(self):
        tcfg = TrainConfig()
        assert tcfg.total_steps > tcfg.gate_on_steps > tcfg.warmup_steps > 0
        assert tcfg.lr_pool > 0
        assert 0 < tcfg.ema_decay < 1

    def test_lr_warmup_property(self):
        tcfg = TrainConfig()
        assert tcfg.lr_warmup == tcfg.lr_warmup_steps


# ---------------------------------------------------------------------------
# RunConfig YAML round-trip
# ---------------------------------------------------------------------------

class TestRunConfigRoundTrip:
    def _make_run_cfg(self):
        return RunConfig(
            name="test_run",
            model=DWAConfig.small(),
            train=TrainConfig(),
            sharding=ShardingConfig(n_model=1),
            data=DataConfig(source="random", gen_every=50),
            checkpoint=CheckpointConfig(dir="/tmp/ckpt", every=100, resume=False),
        )

    def test_to_dict_has_all_sections(self):
        d = to_dict(self._make_run_cfg())
        assert set(d.keys()) >= {"name", "model", "train", "sharding", "data", "checkpoint"}

    def test_yaml_round_trip(self):
        run_cfg = self._make_run_cfg()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            path = f.name
        try:
            save_config(run_cfg, path)
            loaded = load_config(path)
            assert loaded.name == run_cfg.name
            assert loaded.model.D == run_cfg.model.D
            assert loaded.model.N == run_cfg.model.N
            assert loaded.train.total_steps == run_cfg.train.total_steps
            assert loaded.sharding.n_model == run_cfg.sharding.n_model
            assert loaded.data.source == run_cfg.data.source
            assert loaded.checkpoint.dir == run_cfg.checkpoint.dir
        finally:
            os.unlink(path)

    def test_unknown_model_key_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("model:\n  D: 2048\n  UNKNOWN_KEY: 999\n")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unknown model config keys"):
                load_config(path)
        finally:
            os.unlink(path)

    def test_compute_dtype_bfloat16(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("model:\n  compute_dtype: bfloat16\n")
            path = f.name
        try:
            import jax.numpy as jnp
            cfg = load_config(path)
            assert cfg.model.compute_dtype == jnp.bfloat16
        finally:
            os.unlink(path)

    def test_compute_dtype_float32(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("model:\n  compute_dtype: float32\n")
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg.model.compute_dtype is None
        finally:
            os.unlink(path)

    def test_all_preset_yaml_files_load(self):
        yaml_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
        for fname in os.listdir(yaml_dir):
            if fname.endswith(".yaml"):
                path = os.path.join(yaml_dir, fname)
                cfg = load_config(path)
                assert cfg.model.D > 0, f"Bad config in {fname}"

    def test_n_model_auto_parsed(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("sharding:\n  n_model: auto\n")
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg.sharding.n_model == "auto"
        finally:
            os.unlink(path)
