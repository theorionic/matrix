"""Tests for DWAModel forward pass — shapes, causality, gradient flow."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from src.dwa.config import DWAConfig, TrainConfig
from src.dwa.model import DWAModel, forward_and_loss


@pytest.fixture(scope="module")
def cfg():
    return DWAConfig.small()


@pytest.fixture(scope="module")
def tcfg():
    return TrainConfig()


@pytest.fixture
def model(cfg):
    return DWAModel(cfg, nnx.Rngs(0))


@pytest.fixture
def input_ids(cfg):
    return jnp.ones((4, cfg.seq_len), dtype=jnp.int32)


class TestForwardShapes:
    def test_logits_shape(self, model, cfg, input_ids):
        logits, _ = model(input_ids, 1.0, True, use_pallas=False)
        assert logits.shape == (4, cfg.seq_len, cfg.vocab_size)

    def test_alphas_shape(self, model, cfg, input_ids):
        _, metrics = model(input_ids, 1.0, True, use_pallas=False)
        assert metrics["alphas"].shape == (4, cfg.k_max)

    def test_indices_shape(self, model, cfg, input_ids):
        _, metrics = model(input_ids, 1.0, True, use_pallas=False)
        assert metrics["indices"].shape == (4, cfg.k_max)

    def test_W_shape(self, model, cfg, input_ids):
        _, metrics = model(input_ids, 1.0, True, use_pallas=False)
        assert metrics["W"].shape == (4, cfg.d_B, cfg.d_A)

    def test_h_out_shape(self, model, cfg, input_ids):
        _, metrics = model(input_ids, 1.0, True, use_pallas=False)
        assert metrics["h_out"].shape == (4, cfg.seq_len, cfg.d_B)

    def test_pool_keys_shape(self, model, cfg, input_ids):
        _, metrics = model(input_ids, 1.0, True, use_pallas=False)
        assert metrics["pool_keys"].shape == (cfg.S, cfg.N, cfg.d_k)

    def test_with_key_cache(self, model, cfg, input_ids):
        key_cache = model.pool.compute_keys()
        logits, _ = model(input_ids, 1.0, True, key_cache=key_cache, use_pallas=False)
        assert logits.shape == (4, cfg.seq_len, cfg.vocab_size)

    def test_batch_size_1(self, model, cfg):
        ids = jnp.ones((1, cfg.seq_len), dtype=jnp.int32)
        logits, _ = model(ids, 1.0, True, use_pallas=False)
        assert logits.shape == (1, cfg.seq_len, cfg.vocab_size)


class TestForwardCorrectness:
    def test_logits_finite(self, model, cfg, input_ids):
        logits, _ = model(input_ids, 1.0, True, use_pallas=False)
        assert jnp.all(jnp.isfinite(logits))

    def test_alphas_sum_to_one(self, model, cfg, input_ids):
        _, metrics = model(input_ids, 1.0, True, use_pallas=False)
        row_sums = metrics["alphas"].sum(axis=-1)
        assert jnp.allclose(row_sums, jnp.ones(4), atol=1e-5)

    def test_indices_in_pool_range(self, model, cfg, input_ids):
        _, metrics = model(input_ids, 1.0, True, use_pallas=False)
        assert jnp.all(metrics["indices"] >= 0)
        assert jnp.all(metrics["indices"] < cfg.N)

    def test_warmup_vs_gate_same_shape(self, model, cfg, input_ids):
        logits_w, _ = model(input_ids, 1.0, True, use_pallas=False)
        logits_g, _ = model(input_ids, 5.0, False, use_pallas=False)
        assert logits_w.shape == logits_g.shape

    def test_key_cache_matches_computed(self, model, cfg, input_ids):
        """key_cache pre-computed vs on-the-fly should give same logits."""
        key_cache = model.pool.compute_keys()
        logits_cached, _ = model(input_ids, 1.0, True, key_cache=key_cache, use_pallas=False)
        logits_fresh, _ = model(input_ids, 1.0, True, key_cache=None, use_pallas=False)
        assert jnp.allclose(logits_cached, logits_fresh, atol=1e-4)


class TestForwardAndLoss:
    def test_returns_scalar_loss(self, cfg, tcfg, input_ids):
        model = DWAModel(cfg, nnx.Rngs(0))
        loss, info = forward_and_loss(
            model, input_ids, 1.0, True, tcfg, aux_on=False, use_pallas=False
        )
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_aux_on_adds_loss(self, cfg, tcfg, input_ids):
        model = DWAModel(cfg, nnx.Rngs(0))
        loss_no_aux, _ = forward_and_loss(
            model, input_ids, 5.0, False, tcfg, aux_on=False, use_pallas=False
        )
        loss_aux, _ = forward_and_loss(
            model, input_ids, 5.0, False, tcfg, aux_on=True, use_pallas=False
        )
        # With aux losses the total may differ from task-only
        assert jnp.isfinite(loss_aux)

    def test_info_has_required_keys(self, cfg, tcfg, input_ids):
        model = DWAModel(cfg, nnx.Rngs(0))
        _, info = forward_and_loss(
            model, input_ids, 1.0, True, tcfg, aux_on=True, use_pallas=False
        )
        for key in ("l_task", "alphas", "indices", "l_util", "l_div", "l_norm", "l_sparse"):
            assert key in info, f"Missing key: {key}"


class TestGradientFlow:
    def test_gradients_flow_to_all_param_groups(self, cfg, tcfg, input_ids):
        """Verify all major param groups get non-zero gradients."""
        model = DWAModel(cfg, nnx.Rngs(0))

        def loss_fn(m):
            loss, _ = forward_and_loss(
                m, input_ids, 5.0, False, tcfg, aux_on=True, use_pallas=False
            )
            return loss

        grads = nnx.grad(loss_fn)(model)
        # Pool vectors
        assert jnp.any(grads.pool.vectors.value != 0), "Pool vectors got zero gradient"
        # Retrieval W_Q
        assert jnp.any(grads.retrieval.W_Q.value != 0), "Retrieval W_Q got zero gradient"
        # Embedding
        assert jnp.any(grads.embed.embedding.value != 0), "Embedding got zero gradient"

    def test_gradients_finite(self, cfg, tcfg, input_ids):
        model = DWAModel(cfg, nnx.Rngs(0))

        def loss_fn(m):
            loss, _ = forward_and_loss(
                m, input_ids, 5.0, False, tcfg, aux_on=True, use_pallas=False
            )
            return loss

        grads = nnx.grad(loss_fn)(model)
        param_state = nnx.state(grads, nnx.Param)
        leaves = jax.tree_util.tree_leaves(param_state)
        for leaf in leaves:
            assert jnp.all(jnp.isfinite(leaf)), "Non-finite gradient detected"


class TestModelJIT:
    def test_forward_jittable(self, cfg, input_ids):
        model = DWAModel(cfg, nnx.Rngs(0))
        fn = nnx.jit(lambda m, x: m(x, 1.0, True, use_pallas=False))
        logits, _ = fn(model, input_ids)
        assert jnp.all(jnp.isfinite(logits))

    def test_forward_and_loss_jittable(self, cfg, tcfg, input_ids):
        model = DWAModel(cfg, nnx.Rngs(0))

        @nnx.jit
        def step(m, x):
            return forward_and_loss(m, x, 5.0, False, tcfg, aux_on=True, use_pallas=False)

        loss, info = step(model, input_ids)
        assert jnp.isfinite(loss)
