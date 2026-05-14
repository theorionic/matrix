"""Shared math utilities — pure functions, no state."""

import jax
import jax.numpy as jnp


def cosine_similarity(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Cosine similarity along the last axis. a and b must share their last dim."""
    a_norm = jnp.linalg.norm(a, axis=-1, keepdims=True).clip(eps)
    b_norm = jnp.linalg.norm(b, axis=-1, keepdims=True).clip(eps)
    return (a / a_norm) * (b / b_norm)


def cosine_sim_batched(queries: jnp.ndarray, keys: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """
    Batch cosine similarity.
    queries: [..., d_k]
    keys:    [N, d_k]
    returns: [..., N]
    """
    q_norm = jnp.linalg.norm(queries, axis=-1, keepdims=True).clip(eps)
    k_norm = jnp.linalg.norm(keys, axis=-1, keepdims=True).clip(eps)
    q = queries / q_norm   # [..., d_k]
    k = keys / k_norm      # [N, d_k]
    return jnp.einsum("...d,nd->...n", q, k)


def ema_update(ema: jnp.ndarray, new_val: jnp.ndarray, decay: float) -> jnp.ndarray:
    return decay * ema + (1.0 - decay) * new_val
