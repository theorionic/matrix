"""Tests for checkpoint save/restore round-trip."""

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from src.dwa.config import DWAConfig, TrainConfig
from src.dwa.model import DWAModel
from src.dwa.schedule import PhaseScheduler


def _tiny_cfg():
    return DWAConfig(
        D=2048, N=64, d_A=64, d_B=64, r=4,
        k_max=4, S=2, d_k=32, m=2,
        n_heads=2, n_layers_A=1, n_layers_B=1,
        ffn_mult=2, vocab_size=32, seq_len=16,
        use_ivf=True, C=4,
    )


def _tiny_tcfg():
    t = TrainConfig()
    t.warmup_steps = 2
    t.gate_on_steps = 4
    t.total_steps = 8
    t.batch_size = 4
    t.steps_per_window = 2
    return t


def _get_mesh():
    devices = jax.devices()
    n = len(devices)
    return Mesh(np.array(devices).reshape(n, 1), ("data", "model"))


@pytest.fixture(scope="module")
def mesh():
    return _get_mesh()


class TestCheckpointRoundTrip:
    def _setup(self, mesh):
        from train import _build_optimizer, save_checkpoint, load_checkpoint
        cfg = _tiny_cfg()
        tcfg = _tiny_tcfg()
        scheduler = PhaseScheduler(tcfg)
        model = DWAModel(cfg, nnx.Rngs(0))
        optimizer = _build_optimizer(model, tcfg, scheduler)
        pool_ema = jax.random.normal(jax.random.PRNGKey(42), (cfg.N,)) * 0.1
        rng = jax.random.PRNGKey(123)
        return cfg, tcfg, model, optimizer, pool_ema, rng

    def test_save_creates_directory(self, mesh):
        from train import save_checkpoint
        cfg, tcfg, model, optimizer, pool_ema, rng = self._setup(mesh)
        with tempfile.TemporaryDirectory() as ckpt_dir:
            save_checkpoint(ckpt_dir, model, optimizer, pool_ema, 10, rng, None)
            assert os.path.isdir(os.path.join(ckpt_dir, "10"))

    def test_restore_model_params(self, mesh):
        from train import save_checkpoint, load_checkpoint
        cfg, tcfg, model, optimizer, pool_ema, rng = self._setup(mesh)

        pool_before = np.array(model.pool.vectors[...]).copy()

        with tempfile.TemporaryDirectory() as ckpt_dir:
            save_checkpoint(ckpt_dir, model, optimizer, pool_ema, 10, rng, None)

            # Corrupt the model params
            model.pool.vectors[...] = jnp.zeros_like(model.pool.vectors[...])

            steps_done, rng_out, loader_state, ema_out = load_checkpoint(
                ckpt_dir, 10, model, optimizer, cfg, mesh
            )

            pool_after = np.array(model.pool.vectors[...])
            assert np.allclose(pool_before, pool_after, atol=1e-4), \
                "Pool vectors not correctly restored"

    def test_restore_metadata(self, mesh):
        from train import save_checkpoint, load_checkpoint
        cfg, tcfg, model, optimizer, pool_ema, rng = self._setup(mesh)

        with tempfile.TemporaryDirectory() as ckpt_dir:
            save_checkpoint(ckpt_dir, model, optimizer, pool_ema, 42, rng, None)
            steps_done, rng_out, loader_state, ema_out = load_checkpoint(
                ckpt_dir, 42, model, optimizer, cfg, mesh
            )
            assert steps_done == 42

    def test_restore_pool_ema(self, mesh):
        from train import save_checkpoint, load_checkpoint
        cfg, tcfg, model, optimizer, pool_ema, rng = self._setup(mesh)

        with tempfile.TemporaryDirectory() as ckpt_dir:
            save_checkpoint(ckpt_dir, model, optimizer, pool_ema, 10, rng, None)
            _, _, _, ema_out = load_checkpoint(ckpt_dir, 10, model, optimizer, cfg, mesh)

            assert np.allclose(np.array(pool_ema), np.array(ema_out), atol=1e-5)

    def test_restore_rng_key(self, mesh):
        from train import save_checkpoint, load_checkpoint
        cfg, tcfg, model, optimizer, pool_ema, rng = self._setup(mesh)

        with tempfile.TemporaryDirectory() as ckpt_dir:
            save_checkpoint(ckpt_dir, model, optimizer, pool_ema, 7, rng, None)
            _, rng_out, _, _ = load_checkpoint(ckpt_dir, 7, model, optimizer, cfg, mesh)

            assert np.array_equal(np.array(rng), np.array(rng_out))

    def test_save_with_loader_state(self, mesh):
        from train import save_checkpoint, load_checkpoint
        cfg, tcfg, model, optimizer, pool_ema, rng = self._setup(mesh)

        # Simulate a loader state dict
        class FakeLoader:
            _cursor = 1234
            seq_len = cfg.seq_len
            def state_dict(self):
                buf = np.zeros((10, self.seq_len), dtype=np.int32)
                return {"cursor": self._cursor, "buf": buf}

        loader = FakeLoader()
        with tempfile.TemporaryDirectory() as ckpt_dir:
            save_checkpoint(ckpt_dir, model, optimizer, pool_ema, 5, rng, loader)
            loader_path = os.path.join(ckpt_dir, "5", "loader_state.npz")
            assert os.path.exists(loader_path)
            d = np.load(loader_path)
            assert int(d["cursor"]) == 1234
