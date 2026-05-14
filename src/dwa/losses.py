"""Auxiliary losses for DWA training."""

import jax
import jax.numpy as jnp

from .config import DWAConfig, TrainConfig


def aux_losses(
    alphas: jnp.ndarray,      # [B, k_max] — normalized assembly weights
    indices: jnp.ndarray,     # [B, k_max] — pool indices
    pool_keys: jnp.ndarray,   # [S, N, d_k]
    W: jnp.ndarray,           # [B, d_B, d_A] — assembled weight matrix
    W_base: jnp.ndarray,      # [d_B, d_A]
    pool_ema: jnp.ndarray,    # [N] — EMA of per-vector utilization
    cfg: DWAConfig,
    tcfg: TrainConfig,
) -> dict[str, jnp.ndarray]:
    """
    Compute all four auxiliary losses.  All inputs are plain jnp arrays
    (no NNX wrappers) so this is a pure function compatible with jax.grad.
    """
    B, k = alphas.shape

    # L_util: prevent dead pool vectors
    # Penalises vectors with low EMA usage: -Σ log(1 - exp(-β·ema_i))
    beta = 10.0
    safe_ema = jnp.clip(pool_ema, 1e-6, 1.0)
    l_util = -jnp.log(1.0 - jnp.exp(-beta * safe_ema) + 1e-8).mean()

    # L_div: prevent key collapse among retrieved keys
    # Gather the S-aspect keys for retrieved vectors: [B, k, S, d_k]
    # pool_keys: [S, N, d_k] → gather → [B, k, S, d_k]
    retrieved_keys = pool_keys[:, indices, :]           # [S, B, k, d_k]
    retrieved_keys = retrieved_keys.transpose(1, 2, 0, 3)  # [B, k, S, d_k]
    # Mean over aspects → [B, k, d_k]; cosine among k vectors
    rk_mean = retrieved_keys.mean(axis=2)               # [B, k, d_k]
    rk_norm = rk_mean / (jnp.linalg.norm(rk_mean, axis=-1, keepdims=True) + 1e-8)
    # All-pairs cosine (upper triangle)
    cos_mat = jnp.einsum("bkd,bld->bkl", rk_norm, rk_norm)  # [B, k, k]
    mask = jnp.triu(jnp.ones((k, k), jnp.bool_), k=1)
    l_div = (cos_mat * mask[None]).sum() / (mask.sum() * B + 1e-8)

    # L_norm: prevent assembly explosion ‖W - W_base‖²_F
    W_delta = W - W_base[None]                          # [B, d_B, d_A]
    l_norm = (W_delta ** 2).sum(axis=(-1, -2)).mean()

    # L_sparse: weight entropy -Σ α_i log(α_i)
    safe_alpha = jnp.clip(alphas, 1e-8, 1.0)
    l_sparse = -(safe_alpha * jnp.log(safe_alpha)).sum(axis=-1).mean()

    return {
        "l_util": l_util,
        "l_div": l_div,
        "l_norm": l_norm,
        "l_sparse": l_sparse,
        "total_aux": (
            tcfg.lambda_util * l_util
            + tcfg.lambda_div * l_div
            + tcfg.lambda_norm * l_norm
            + tcfg.lambda_sparse * l_sparse
        ),
    }


def task_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Cross-entropy language-model loss (next-token prediction)."""
    # logits: [B, T, V], targets: [B, T]
    # Shift: predict token t+1 from token t
    logits_shifted = logits[:, :-1, :]   # [B, T-1, V]
    targets_shifted = targets[:, 1:]     # [B, T-1]
    B, T, V = logits_shifted.shape
    log_probs = jax.nn.log_softmax(logits_shifted.reshape(B * T, V), axis=-1)
    one_hot = jax.nn.one_hot(targets_shifted.reshape(B * T), V)
    loss = -(log_probs * one_hot).sum(axis=-1)
    return loss.mean()
