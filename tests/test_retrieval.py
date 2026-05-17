"""Tests for MultiAspectRetrieval — shapes, alphas invariants, index validity."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from src.dwa.config import DWAConfig
from src.dwa.retrieval import MultiAspectRetrieval


@pytest.fixture(scope="module")
def cfg():
    return DWAConfig.small()


@pytest.fixture(scope="module")
def retrieval(cfg):
    return MultiAspectRetrieval(cfg, nnx.Rngs(0))


@pytest.fixture(scope="module")
def pool_keys(cfg):
    return jax.random.normal(jax.random.PRNGKey(1), (cfg.S, cfg.N, cfg.d_k))


def _make_query(cfg, B=4):
    return jax.random.normal(jax.random.PRNGKey(99), (B, cfg.d_A))


class TestRetrievalShapes:
    def test_warmup_output_shapes(self, retrieval, cfg, pool_keys):
        B = 4
        z = _make_query(cfg, B)
        alphas, indices, soft_full = retrieval(z, pool_keys, 1.0, True)
        assert alphas.shape == (B, cfg.k_max)
        assert indices.shape == (B, cfg.k_max)
        assert soft_full.shape[0] == B

    def test_gate_output_shapes(self, retrieval, cfg, pool_keys):
        B = 4
        z = _make_query(cfg, B)
        alphas, indices, soft_full = retrieval(z, pool_keys, 5.0, False)
        assert alphas.shape == (B, cfg.k_max)
        assert indices.shape == (B, cfg.k_max)


class TestRetrievalInvariants:
    def test_alphas_sum_to_one(self, retrieval, cfg, pool_keys):
        z = _make_query(cfg)
        alphas, _, _ = retrieval(z, pool_keys, 1.0, True)
        row_sums = alphas.sum(axis=-1)
        assert jnp.allclose(row_sums, jnp.ones(4), atol=1e-5)

    def test_alphas_non_negative(self, retrieval, cfg, pool_keys):
        z = _make_query(cfg)
        alphas, _, _ = retrieval(z, pool_keys, 5.0, False)
        assert jnp.all(alphas >= 0)

    def test_indices_in_valid_range(self, retrieval, cfg, pool_keys):
        z = _make_query(cfg)
        _, indices, _ = retrieval(z, pool_keys, 1.0, True)
        assert jnp.all(indices >= 0)
        assert jnp.all(indices < cfg.N)

    def test_soft_full_sums_to_one(self, retrieval, cfg, pool_keys):
        z = _make_query(cfg)
        _, _, soft_full = retrieval(z, pool_keys, 1.0, True)
        row_sums = soft_full.sum(axis=-1)
        assert jnp.allclose(row_sums, jnp.ones(4), atol=1e-4)


class TestRetrievalGradients:
    def test_gradient_flows_to_W_Q(self, cfg, pool_keys):
        retrieval = MultiAspectRetrieval(cfg, nnx.Rngs(42))

        def loss_fn(m):
            z = _make_query(cfg)
            alphas, _, _ = m(z, pool_keys, 5.0, False)
            return alphas.sum()

        grads = nnx.grad(loss_fn)(retrieval)
        W_Q_grad = grads.W_Q.value
        assert jnp.any(W_Q_grad != 0), "W_Q gradient is all zeros"
        assert jnp.all(jnp.isfinite(W_Q_grad))

    def test_gradient_flows_to_tau(self, cfg, pool_keys):
        retrieval = MultiAspectRetrieval(cfg, nnx.Rngs(42))

        def loss_fn(m):
            z = _make_query(cfg)
            alphas, _, soft_full = m(z, pool_keys, 5.0, False)
            return alphas.sum() + soft_full.sum()

        grads = nnx.grad(loss_fn)(retrieval)
        tau_grad = grads.tau.value
        assert jnp.isfinite(tau_grad)


class TestRetrievalJIT:
    def test_warmup_jittable(self, retrieval, cfg, pool_keys):
        z = _make_query(cfg)
        fn = nnx.jit(lambda m, z, pk: m(z, pk, 1.0, True))
        alphas, indices, soft_full = fn(retrieval, z, pool_keys)
        assert jnp.all(jnp.isfinite(alphas))

    def test_gate_jittable(self, retrieval, cfg, pool_keys):
        z = _make_query(cfg)
        fn = nnx.jit(lambda m, z, pk: m(z, pk, 5.0, False))
        alphas, indices, soft_full = fn(retrieval, z, pool_keys)
        assert jnp.all(jnp.isfinite(alphas))
