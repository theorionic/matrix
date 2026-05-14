"""WeightAssembler — factorized low-rank assembly + residual application."""

import jax
import jax.numpy as jnp
from flax import nnx

from .config import DWAConfig


class WeightAssembler(nnx.Module):
    """
    Owns W_base, b_base, gamma, and LayerNorm.

    Given gathered pool vectors and their normalized weights, assembles
    W = W_base + Σ_k α_k · (U_k @ V_k)  and applies the residual:
        h_mid = LN(h_A + γ · h_A @ W^T + b_assembled)

    W_base is initialized near zero so the model starts as identity-like
    and the assembly learns incremental perturbations (LoRA-style).
    """

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs) -> None:
        self.cfg = cfg
        self.W_base = nnx.Param(
            jax.random.normal(rngs.params(), (cfg.d_B, cfg.d_A)) * 0.01
        )
        self.b_base = nnx.Param(jnp.zeros(cfg.d_B))
        self.gamma = nnx.Param(jnp.array(cfg.gamma_init))
        self.layer_norm = nnx.RMSNorm(cfg.d_B, rngs=rngs)

    def __call__(
        self,
        h_A: jnp.ndarray,      # [B, seq, d_A]
        gathered: jnp.ndarray, # [B, k_max, D] — raw pool vectors
        alphas: jnp.ndarray,   # [B, k_max]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns:
            h_mid: [B, seq, d_B]
            W:     [B, d_B, d_A]  — assembled weight matrix (for aux losses)
        """
        cfg = self.cfg
        s1, s2, s3 = cfg.factor_split

        # Factorize gathered vectors into (U, V, b) per sample per vector
        U = gathered[..., :s1].reshape(
            gathered.shape[0], cfg.k_max, cfg.d_B, cfg.r
        )  # [B, k, d_B, r]
        V = gathered[..., s1:s2].reshape(
            gathered.shape[0], cfg.k_max, cfg.r, cfg.d_A
        )  # [B, k, r, d_A]
        b_vec = gathered[..., s2:s3]  # [B, k, d_B]

        # Assemble W = W_base + Σ_k α_k · (U_k @ V_k)
        # einsum: [B,k] * [B,k,d_B,r] * [B,k,r,d_A] → [B,d_B,d_A]
        delta_W = jnp.einsum("bk,bkur,bkra->bua", alphas, U, V)
        W = self.W_base[...][None, ...] + delta_W   # [B, d_B, d_A]

        delta_b = jnp.einsum("bk,bkd->bd", alphas, b_vec)
        bias = self.b_base[...][None, :] + delta_b  # [B, d_B]

        # Residual: h_mid = LN(h_A + γ · h_A @ W^T + b)
        # h_A: [B, seq, d_A], W: [B, d_B, d_A], W^T: [B, d_A, d_B]
        # h_A @ W^T → [B, seq, d_B]
        h_residual = jnp.einsum("bsd,bud->bsu", h_A, W)   # [B, seq, d_B]
        h_mid = h_A + self.gamma[...] * h_residual + bias[:, None, :]
        h_mid = self.layer_norm(h_mid)
        return h_mid, W
