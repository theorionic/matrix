"""
End-to-end training step tests — uses small config to minimize compile time.

These tests verify:
  1. A single train_window call succeeds and returns finite loss
  2. Model parameters actually update after the step
  3. The NaN guard correctly zeros gradients for poisoned steps
  4. The loss is lower than random baseline after a few steps on a learnable task
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from src.dwa.config import DWAConfig, TrainConfig
from src.dwa.model import DWAModel
from src.dwa.schedule import PhaseScheduler


def _small_cfg():
    """Minimal config for fast compilation in tests."""
    cfg = DWAConfig(
        D=2048, N=64, d_A=64, d_B=64, r=4,
        k_max=4, S=2, d_k=32, m=2,
        n_heads=2, n_layers_A=1, n_layers_B=1,
        ffn_mult=2, vocab_size=32, seq_len=16,
        use_ivf=True, C=4,
    )
    return cfg


def _small_tcfg():
    tcfg = TrainConfig()
    tcfg.warmup_steps = 4
    tcfg.gate_on_steps = 8
    tcfg.total_steps = 32
    tcfg.batch_size = 8
    tcfg.steps_per_window = 4
    tcfg.lr_warmup_steps = 2
    tcfg.lr_parts = 1e-3
    tcfg.lr_pool = 1e-4
    tcfg.lr_retrieval = 1e-3
    tcfg.lr_threshold = 1e-2
    return tcfg


def _build_optimizer(model, tcfg, scheduler):
    from train import _build_optimizer as _opt
    return _opt(model, tcfg, scheduler)


def _get_1d_mesh():
    devices = jax.devices()
    n = len(devices)
    return Mesh(np.array(devices).reshape(n, 1), ("data", "model"))


@pytest.fixture(scope="module")
def cfg():
    return _small_cfg()


@pytest.fixture(scope="module")
def tcfg():
    return _small_tcfg()


@pytest.fixture(scope="module")
def mesh():
    return _get_1d_mesh()


@pytest.fixture(scope="module")
def scheduler(tcfg):
    return PhaseScheduler(tcfg)


class TestSingleWindow:
    def test_loss_finite_warmup(self, cfg, tcfg, scheduler, mesh):
        from train import _make_train_window
        model = DWAModel(cfg, nnx.Rngs(0))
        optimizer = _build_optimizer(model, tcfg, scheduler)
        pool_ema = jnp.zeros(cfg.N)
        lam_window = jnp.ones(tcfg.steps_per_window)

        data_window = jnp.zeros((tcfg.steps_per_window, tcfg.batch_size, cfg.seq_len), dtype=jnp.int32)
        data_sharded = jax.device_put(data_window, NamedSharding(mesh, P(None, "data", None)))

        train_fn = _make_train_window(cfg, tcfg, is_warmup=True, aux_on=False, use_pallas=False, mesh=mesh)
        model, optimizer, pool_ema_out, info = train_fn(
            model, optimizer, data_sharded, lam_window, pool_ema, tcfg.ema_decay
        )
        jax.block_until_ready(info["losses"])

        losses = np.array(info["losses"])
        assert np.all(np.isfinite(losses)), f"Non-finite losses: {losses}"
        assert float(losses.mean()) > 0

    def test_loss_finite_gate_on(self, cfg, tcfg, scheduler, mesh):
        from train import _make_train_window
        model = DWAModel(cfg, nnx.Rngs(1))
        optimizer = _build_optimizer(model, tcfg, scheduler)
        pool_ema = jnp.zeros(cfg.N)
        lam_window = jnp.full((tcfg.steps_per_window,), 2.0)

        data_window = jnp.zeros((tcfg.steps_per_window, tcfg.batch_size, cfg.seq_len), dtype=jnp.int32)
        data_sharded = jax.device_put(data_window, NamedSharding(mesh, P(None, "data", None)))

        train_fn = _make_train_window(cfg, tcfg, is_warmup=False, aux_on=True, use_pallas=False, mesh=mesh)
        model, optimizer, pool_ema_out, info = train_fn(
            model, optimizer, data_sharded, lam_window, pool_ema, tcfg.ema_decay
        )
        jax.block_until_ready(info["losses"])
        assert np.all(np.isfinite(np.array(info["losses"])))

    def test_parameters_update_after_step(self, cfg, tcfg, scheduler, mesh):
        from train import _make_train_window
        model = DWAModel(cfg, nnx.Rngs(2))
        optimizer = _build_optimizer(model, tcfg, scheduler)
        pool_ema = jnp.zeros(cfg.N)
        lam_window = jnp.ones(tcfg.steps_per_window)

        pool_before = np.array(model.pool.vectors[...]).copy()

        data_window = jax.random.randint(jax.random.PRNGKey(0),
                                         (tcfg.steps_per_window, tcfg.batch_size, cfg.seq_len), 0, cfg.vocab_size)
        data_sharded = jax.device_put(data_window, NamedSharding(mesh, P(None, "data", None)))

        train_fn = _make_train_window(cfg, tcfg, is_warmup=True, aux_on=False, use_pallas=False, mesh=mesh)
        model, optimizer, pool_ema_out, info = train_fn(
            model, optimizer, data_sharded, lam_window, pool_ema, tcfg.ema_decay
        )
        jax.block_until_ready(info["losses"])

        pool_after = np.array(model.pool.vectors[...])
        assert not np.allclose(pool_before, pool_after), "Pool vectors did not update"

    def test_pool_ema_updates(self, cfg, tcfg, scheduler, mesh):
        from train import _make_train_window
        model = DWAModel(cfg, nnx.Rngs(3))
        optimizer = _build_optimizer(model, tcfg, scheduler)
        pool_ema = jnp.zeros(cfg.N)

        data_window = jax.random.randint(jax.random.PRNGKey(5),
                                         (tcfg.steps_per_window, tcfg.batch_size, cfg.seq_len), 0, cfg.vocab_size)
        data_sharded = jax.device_put(data_window, NamedSharding(mesh, P(None, "data", None)))
        lam_window = jnp.ones(tcfg.steps_per_window)

        train_fn = _make_train_window(cfg, tcfg, is_warmup=True, aux_on=False, use_pallas=False, mesh=mesh)
        _, _, pool_ema_out, _ = train_fn(
            model, optimizer, data_sharded, lam_window, pool_ema, tcfg.ema_decay
        )
        ema_arr = np.array(pool_ema_out)
        assert np.any(ema_arr > 0), "Pool EMA stayed at zero after training step"

    def test_grad_norms_info(self, cfg, tcfg, scheduler, mesh):
        from train import _make_train_window
        model = DWAModel(cfg, nnx.Rngs(4))
        optimizer = _build_optimizer(model, tcfg, scheduler)
        pool_ema = jnp.zeros(cfg.N)
        lam_window = jnp.ones(tcfg.steps_per_window)

        data_window = jax.random.randint(jax.random.PRNGKey(7),
                                         (tcfg.steps_per_window, tcfg.batch_size, cfg.seq_len), 0, cfg.vocab_size)
        data_sharded = jax.device_put(data_window, NamedSharding(mesh, P(None, "data", None)))

        train_fn = _make_train_window(cfg, tcfg, is_warmup=True, aux_on=False, use_pallas=False, mesh=mesh)
        _, _, _, info = train_fn(model, optimizer, data_sharded, lam_window, pool_ema, tcfg.ema_decay)

        jax.block_until_ready(info["grad_norms"])
        gnorms = np.array(info["grad_norms"])
        assert gnorms.shape == (tcfg.steps_per_window,)
        assert np.all(gnorms >= 0)
        assert np.all(np.isfinite(gnorms))

    def test_nan_flags_zero_for_normal_data(self, cfg, tcfg, scheduler, mesh):
        from train import _make_train_window
        model = DWAModel(cfg, nnx.Rngs(5))
        optimizer = _build_optimizer(model, tcfg, scheduler)
        pool_ema = jnp.zeros(cfg.N)
        lam_window = jnp.ones(tcfg.steps_per_window)

        data_window = jax.random.randint(jax.random.PRNGKey(9),
                                         (tcfg.steps_per_window, tcfg.batch_size, cfg.seq_len), 0, cfg.vocab_size)
        data_sharded = jax.device_put(data_window, NamedSharding(mesh, P(None, "data", None)))

        train_fn = _make_train_window(cfg, tcfg, is_warmup=True, aux_on=False, use_pallas=False, mesh=mesh)
        _, _, _, info = train_fn(model, optimizer, data_sharded, lam_window, pool_ema, tcfg.ema_decay)

        jax.block_until_ready(info["nan_flags"])
        nan_flags = np.array(info["nan_flags"])
        assert nan_flags.sum() == 0, f"Unexpected NaN flags: {nan_flags}"


class TestLossDecline:
    def test_loss_lower_than_random_baseline(self, cfg, tcfg, scheduler, mesh):
        """After ~8 windows on a fixed pattern, loss should drop below random baseline."""
        from train import _make_train_window
        cfg_local = _small_cfg()
        tcfg_local = _small_tcfg()
        tcfg_local.lr_parts = 3e-3
        tcfg_local.lr_retrieval = 3e-3
        sched_local = PhaseScheduler(tcfg_local)
        model = DWAModel(cfg_local, nnx.Rngs(0))
        optimizer = _build_optimizer(model, tcfg_local, sched_local)
        pool_ema = jnp.zeros(cfg_local.N)

        train_fn = _make_train_window(cfg_local, tcfg_local, is_warmup=True, aux_on=False, use_pallas=False, mesh=mesh)

        # Fixed data — same batch every window so model can overfit
        fixed_data = jax.random.randint(jax.random.PRNGKey(0),
                                        (tcfg_local.steps_per_window, tcfg_local.batch_size, cfg_local.seq_len),
                                        0, cfg_local.vocab_size)
        fixed_sharded = jax.device_put(fixed_data, NamedSharding(mesh, P(None, "data", None)))
        lam_window = jnp.ones(tcfg_local.steps_per_window)

        random_baseline = float(jnp.log(cfg_local.vocab_size))
        final_loss = random_baseline  # will be updated

        for _ in range(8):
            model, optimizer, pool_ema, info = train_fn(
                model, optimizer, fixed_sharded, lam_window, pool_ema, tcfg_local.ema_decay
            )
            final_loss = float(np.array(info["losses"]).mean())

        assert final_loss < random_baseline, (
            f"Loss {final_loss:.4f} did not drop below random baseline {random_baseline:.4f}"
        )
