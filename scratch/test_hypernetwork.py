"""Test script for hypernetwork pool generator in DWA model."""

import os
import sys

# Patch JAX config BEFORE any other import (fixes optax/JAX version mismatch)
import jax._src.config as _jax_cfg
_orig_update = _jax_cfg.config.update
def _safe_update(name: str, val) -> None:
    try:
        _orig_update(name, val)
    except AttributeError:
        pass
_jax_cfg.config.update = _safe_update

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# Ensure matrix repo is in path
sys.path.append("/kaggle/working/matrix")

from src.dwa.config import DWAConfig, TrainConfig
from src.dwa.model import DWAModel
from src.dwa.pool import VectorPool


def test_hypernetwork_basic():
    print("=== Running test_hypernetwork_basic ===")
    cfg = DWAConfig.small()
    cfg.use_hypernetwork = True
    cfg.d_emb = 32
    cfg.mlp_hidden_dims = (64, 128)
    
    rngs = nnx.Rngs(42)
    pool = VectorPool(cfg, rngs)
    
    print("Embeddings shape:", pool.embeddings.shape)
    assert pool.embeddings.shape == (cfg.N, cfg.d_emb)
    
    # Compute keys
    keys = pool.compute_keys()
    print("Keys shape:", keys.shape)
    assert keys.shape == (cfg.S, cfg.N, cfg.d_k)
    
    # Get factors
    indices = jnp.array([0, 1, 2, 3])
    U, V, b = pool.get_factors(indices)
    print("U shape:", U.shape)
    print("V shape:", V.shape)
    print("b shape:", b.shape)
    
    assert U.shape == (4, cfg.d_B, cfg.r)
    assert V.shape == (4, cfg.r, cfg.d_A)
    assert b.shape == (4, cfg.d_B)
    
    # Test DynamicVectors representation
    full_vecs = pool.vectors[...]
    print("Full vectors shape:", full_vecs.shape)
    assert full_vecs.shape == (cfg.N, cfg.D)
    
    print("test_hypernetwork_basic passed!\n")


def test_dwamodel_hypernetwork():
    print("=== Running test_dwamodel_hypernetwork ===")
    cfg = DWAConfig.small()
    cfg.use_hypernetwork = True
    cfg.d_emb = 32
    cfg.mlp_hidden_dims = (64, 128)
    cfg.min_explore_noise = 0.0
    cfg.warmup_explore_noise = 0.0
    
    rngs = nnx.Rngs(42)
    model = DWAModel(cfg, rngs)
    
    # Fake batch inputs
    input_ids = jnp.zeros((2, cfg.seq_len), dtype=jnp.int32)
    
    # Test model forward
    logits, metrics = model(input_ids, lambda_val=jnp.array(1.0), is_warmup=False, use_pallas=False)
    print("Logits shape:", logits.shape)
    assert logits.shape == (2, cfg.seq_len, cfg.vocab_size)
    assert jnp.isfinite(logits).all()
    
    # Test gradient flow to MLP and embeddings
    def loss_fn(m):
        out, _ = m(input_ids, lambda_val=jnp.array(1.0), is_warmup=False, use_pallas=False)
        return jnp.mean(out ** 2)
        
    grads = nnx.grad(loss_fn)(model)
    
    # Gradients should exist for embeddings and MLP parameters
    assert hasattr(grads.pool, "embeddings")
    assert grads.pool.embeddings.value is not None
    assert jnp.any(grads.pool.embeddings.value != 0), "Embeddings got zero gradient"
    
    # Check MLP parameters
    mlp_grads = nnx.state(grads.pool.mlp)
    flat_grads, _ = jax.tree_util.tree_flatten(mlp_grads)
    assert len(flat_grads) > 0
    for val in flat_grads:
        assert val is not None
        assert jnp.any(val != 0), "MLP parameter got zero gradient"
        
    print("Gradient flow verified!")
    
    # Test model forward JIT-compilability
    @nnx.jit
    def jit_forward(m, x):
        return m(x, lambda_val=jnp.array(1.0), is_warmup=False, use_pallas=False)[0]
        
    out_jit = jit_forward(model, input_ids)
    assert jnp.allclose(logits, out_jit, atol=1e-5)
    print("JIT compilation verified!")
    print("test_dwamodel_hypernetwork passed!\n")


def test_export_dense_pool():
    print("=== Running test_export_dense_pool ===")
    cfg = DWAConfig.small()
    cfg.use_hypernetwork = True
    cfg.d_emb = 32
    cfg.mlp_hidden_dims = (64, 128)
    
    rngs = nnx.Rngs(42)
    hyper_pool = VectorPool(cfg, rngs)
    
    # Export to dense pool
    dense_pool = hyper_pool.export_dense_pool(nnx.Rngs(123))
    
    assert not dense_pool.cfg.use_hypernetwork
    assert hasattr(dense_pool, "vectors")
    assert dense_pool.vectors.__class__.__name__ != "DynamicVectors"
    assert dense_pool.vectors.shape == (cfg.N, cfg.D)
    
    # Assert numerical equivalence between generated and exported vectors
    assert jnp.allclose(hyper_pool.vectors[...], dense_pool.vectors[...], atol=1e-5)
    assert jnp.allclose(hyper_pool.key_proj[...], dense_pool.key_proj[...], atol=1e-5)
    
    print("Exported pool keys and vectors match exactly!")
    print("test_export_dense_pool passed!\n")


if __name__ == "__main__":
    test_hypernetwork_basic()
    test_dwamodel_hypernetwork()
    test_export_dense_pool()
    print("All hypernetwork pool tests passed successfully!")
