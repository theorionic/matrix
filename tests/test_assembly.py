"""Tests for the weight assembler — assemble_jax correctness and shapes."""

import jax
import jax.numpy as jnp
import pytest

from src.dwa.assembly_pallas import assemble_jax
from src.dwa.config import DWAConfig


@pytest.fixture(scope="module")
def cfg():
    return DWAConfig.small()


def _make_assembly_inputs(cfg, B=2, T=None):
    T = T or cfg.seq_len
    rng = jax.random.PRNGKey(7)
    k = cfg.k_max
    D = cfg.D
    gathered = jax.random.normal(rng, (B, k, D)) * 0.02
    alphas = jax.nn.softmax(jax.random.normal(rng, (B, k)), axis=-1)
    h_A = jax.random.normal(rng, (B, T, cfg.d_A)) * 0.1
    W_base = jnp.zeros((cfg.d_B, cfg.d_A))
    b_base = jnp.zeros(cfg.d_B)
    gamma = jnp.array(cfg.gamma_init)
    return gathered, alphas, h_A, W_base, b_base, gamma


class TestAssembleJax:
    def test_output_shapes(self, cfg):
        B, T = 2, cfg.seq_len
        inputs = _make_assembly_inputs(cfg, B, T)
        h_mid, W = assemble_jax(*inputs, cfg.d_B, cfg.r, cfg.d_A)
        assert h_mid.shape == (B, T, cfg.d_B)
        assert W.shape == (B, cfg.d_B, cfg.d_A)

    def test_zero_alphas_gives_base(self, cfg):
        """With α=0, W = W_base and bias = b_base, so h_mid = h_A + γ·h_A@W_base^T + b_base."""
        B, T = 2, cfg.seq_len
        gathered, alphas, h_A, W_base, b_base, gamma = _make_assembly_inputs(cfg, B, T)
        alphas_zero = jnp.zeros_like(alphas)
        h_mid, W = assemble_jax(gathered, alphas_zero, h_A, W_base, b_base, gamma, cfg.d_B, cfg.r, cfg.d_A)
        # W should equal W_base (broadcast)
        assert jnp.allclose(W, W_base[None], atol=1e-6)

    def test_gamma_zero_gives_pass_through(self, cfg):
        """With γ=0 and W_base=0 and b_base=0 and α=0: h_mid = h_A."""
        B, T = 2, cfg.seq_len
        gathered, alphas, h_A, _, _, _ = _make_assembly_inputs(cfg, B, T)
        W_base = jnp.zeros((cfg.d_B, cfg.d_A))
        b_base = jnp.zeros(cfg.d_B)
        gamma_zero = jnp.array(0.0)
        alphas_zero = jnp.zeros_like(alphas)
        h_mid, W = assemble_jax(gathered, alphas_zero, h_A, W_base, b_base, gamma_zero, cfg.d_B, cfg.r, cfg.d_A)
        assert jnp.allclose(h_mid, h_A, atol=1e-6)

    def test_output_finite(self, cfg):
        inputs = _make_assembly_inputs(cfg)
        h_mid, W = assemble_jax(*inputs, cfg.d_B, cfg.r, cfg.d_A)
        assert jnp.all(jnp.isfinite(h_mid))
        assert jnp.all(jnp.isfinite(W))

    def test_jittable(self, cfg):
        inputs = _make_assembly_inputs(cfg)

        @jax.jit
        def run(gathered, alphas, h_A, W_base, b_base, gamma):
            return assemble_jax(gathered, alphas, h_A, W_base, b_base, gamma, cfg.d_B, cfg.r, cfg.d_A)

        h_mid, W = run(*inputs)
        assert jnp.all(jnp.isfinite(h_mid))

    def test_gradients_flow(self, cfg):
        gathered, alphas, h_A, W_base, b_base, gamma = _make_assembly_inputs(cfg)

        def loss_fn(gathered, alphas, h_A, W_base, b_base, gamma):
            h_mid, W = assemble_jax(gathered, alphas, h_A, W_base, b_base, gamma, cfg.d_B, cfg.r, cfg.d_A)
            return h_mid.sum()

        grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))(
            gathered, alphas, h_A, W_base, b_base, gamma
        )
        for i, g in enumerate(grads):
            assert jnp.all(jnp.isfinite(g)), f"grad[{i}] has non-finite values"
            assert jnp.any(g != 0), f"grad[{i}] is all zeros"

    def test_batch_size_independence(self, cfg):
        """Output for batch[0] should not depend on batch[1] values."""
        gathered, alphas, h_A, W_base, b_base, gamma = _make_assembly_inputs(cfg, B=2)
        h_mid_full, _ = assemble_jax(gathered, alphas, h_A, W_base, b_base, gamma, cfg.d_B, cfg.r, cfg.d_A)
        h_mid_single, _ = assemble_jax(
            gathered[:1], alphas[:1], h_A[:1], W_base, b_base, gamma, cfg.d_B, cfg.r, cfg.d_A
        )
        assert jnp.allclose(h_mid_full[:1], h_mid_single, atol=1e-5)
