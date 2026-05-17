"""Tests for task_loss and aux_losses."""

import jax
import jax.numpy as jnp
import pytest

from src.dwa.config import DWAConfig, TrainConfig
from src.dwa.losses import aux_losses, task_loss


@pytest.fixture(scope="module")
def cfg():
    return DWAConfig.small()


@pytest.fixture(scope="module")
def tcfg():
    return TrainConfig()


class TestTaskLoss:
    def test_returns_scalar(self, cfg):
        B, T, V = 4, cfg.seq_len, cfg.vocab_size
        logits = jnp.ones((B, T, V))
        targets = jnp.zeros((B, T), dtype=jnp.int32)
        loss = task_loss(logits, targets)
        assert loss.shape == ()

    def test_uniform_logits_approx_log_V(self, cfg):
        B, T, V = 4, cfg.seq_len, cfg.vocab_size
        logits = jnp.ones((B, T, V))
        targets = jnp.zeros((B, T), dtype=jnp.int32)
        loss = task_loss(logits, targets)
        assert float(loss) == pytest.approx(jnp.log(V).item(), rel=0.01)

    def test_perfect_prediction_near_zero(self, cfg):
        B, T, V = 2, cfg.seq_len, cfg.vocab_size
        targets = jnp.zeros((B, T), dtype=jnp.int32)
        # Put very large logit on target token (index 0 for next token = index 1 in targets)
        logits = jnp.full((B, T, V), -1e9).at[:, :, 0].set(1e9)
        loss = task_loss(logits, targets)
        assert float(loss) < 0.01

    def test_loss_jittable(self, cfg):
        B, T, V = 2, cfg.seq_len, cfg.vocab_size
        logits = jnp.ones((B, T, V))
        targets = jnp.zeros((B, T), dtype=jnp.int32)
        loss = jax.jit(task_loss)(logits, targets)
        assert jnp.isfinite(loss)

    def test_loss_differentiable(self, cfg):
        B, T, V = 2, cfg.seq_len, cfg.vocab_size
        targets = jnp.zeros((B, T), dtype=jnp.int32)
        grad_fn = jax.grad(lambda logits: task_loss(logits, targets))
        logits = jax.random.normal(jax.random.PRNGKey(0), (B, T, V))
        grads = grad_fn(logits)
        assert grads.shape == (B, T, V)
        assert jnp.all(jnp.isfinite(grads))


class TestAuxLosses:
    def _make_inputs(self, cfg, tcfg, B=4):
        rng = jax.random.PRNGKey(42)
        k = cfg.k_max
        N = cfg.N
        S = cfg.S
        d_k = cfg.d_k
        alphas = jax.nn.softmax(jax.random.normal(rng, (B, k)), axis=-1)
        indices = jnp.broadcast_to(jnp.arange(k)[None], (B, k))
        pool_keys = jax.random.normal(rng, (S, N, d_k))
        W = jax.random.normal(rng, (B, cfg.d_B, cfg.d_A)) * 0.01
        W_base = jnp.zeros((cfg.d_B, cfg.d_A))
        soft_full = jax.nn.softmax(jax.random.normal(rng, (B, N)), axis=-1)
        return alphas, indices, pool_keys, W, W_base, soft_full

    def test_returns_all_keys(self, cfg, tcfg):
        inputs = self._make_inputs(cfg, tcfg)
        result = aux_losses(*inputs, cfg, tcfg)
        expected = {"l_util", "l_div", "l_norm", "l_sparse", "total_aux"}
        assert expected <= set(result.keys())

    def test_all_scalars(self, cfg, tcfg):
        inputs = self._make_inputs(cfg, tcfg)
        result = aux_losses(*inputs, cfg, tcfg)
        for key, val in result.items():
            assert val.shape == (), f"{key} not scalar: {val.shape}"

    def test_all_finite(self, cfg, tcfg):
        inputs = self._make_inputs(cfg, tcfg)
        result = aux_losses(*inputs, cfg, tcfg)
        for key, val in result.items():
            assert jnp.isfinite(val), f"{key} is not finite: {val}"

    def test_total_aux_is_sum_of_components(self, cfg, tcfg):
        inputs = self._make_inputs(cfg, tcfg)
        r = aux_losses(*inputs, cfg, tcfg)
        expected = (
            tcfg.lambda_util * r["l_util"]
            + tcfg.lambda_div * r["l_div"]
            + tcfg.lambda_norm * r["l_norm"]
            + tcfg.lambda_sparse * r["l_sparse"]
        )
        assert float(r["total_aux"]) == pytest.approx(float(expected), rel=1e-5)

    def test_l_util_uniform_is_one(self, cfg, tcfg):
        """Uniform soft distribution over N vectors → l_util ≈ 1.0."""
        B, k, N = 4, cfg.k_max, cfg.N
        alphas = jnp.ones((B, k)) / k
        indices = jnp.broadcast_to(jnp.arange(k)[None], (B, k))
        pool_keys = jnp.ones((cfg.S, N, cfg.d_k))
        W = jnp.zeros((B, cfg.d_B, cfg.d_A))
        W_base = jnp.zeros((cfg.d_B, cfg.d_A))
        soft_full = jnp.ones((B, N)) / N   # perfectly uniform
        r = aux_losses(alphas, indices, pool_keys, W, W_base, soft_full, cfg, tcfg)
        # l_util = N * dot(f, P); f and P both ≈ 1/N → N * N * (1/N)^2 = 1.0
        assert float(r["l_util"]) == pytest.approx(1.0, rel=0.05)

    def test_aux_losses_jittable(self, cfg, tcfg):
        inputs = self._make_inputs(cfg, tcfg)
        fn = jax.jit(lambda a, b, c, d, e, f: aux_losses(a, b, c, d, e, f, cfg, tcfg))
        result = fn(*inputs)
        assert jnp.isfinite(result["total_aux"])
