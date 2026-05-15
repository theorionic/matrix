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
    soft_full: jnp.ndarray,   # [B, N_cands] — full pre-top-k soft distribution
    cfg: DWAConfig,
    tcfg: TrainConfig,
) -> dict[str, jnp.ndarray]:
    """
    Compute all four auxiliary losses.  All inputs are plain jnp arrays
    (no NNX wrappers) so this is a pure function compatible with jax.grad.
    """
    B, k = alphas.shape

    # L_util: maximise entropy of the full pre-top-k retrieval distribution.
    # soft_full [B, N_cands] is the differentiable soft distribution returned
    # by the retrieval module before the discrete top-k selection.  Its entropy
    # has real gradient through W_Q / key_proj → drives diverse pool coverage.
    # l_util = max_entropy - H(soft_full), so 0 = uniform, max = collapsed.
    safe_soft = jnp.clip(soft_full, 1e-8, 1.0)
    H = -(safe_soft * jnp.log(safe_soft)).sum(axis=-1).mean()
    max_H = jnp.log(jnp.array(soft_full.shape[-1], dtype=jnp.float32))
    l_util = max_H - H   # 0 when perfectly uniform, positive when collapsed

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
    flat_logits  = logits_shifted.reshape(B * T, V)
    flat_targets = targets_shifted.reshape(B * T)
    log_probs = jax.nn.log_softmax(flat_logits, axis=-1)
    loss = -log_probs[jnp.arange(B * T), flat_targets]  # gather, no [B*T, V] one_hot
    return loss.mean()
