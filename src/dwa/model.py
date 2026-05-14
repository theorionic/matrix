"""DWAModel — top-level NNX module orchestrating the full forward pass."""

import jax
import jax.numpy as jnp
from flax import nnx

from .assembly import WeightAssembler
from .config import DWAConfig, TrainConfig
from .losses import aux_losses, task_loss
from .parts import PartA, PartB
from .pool import VectorPool
from .retrieval import MultiAspectRetrieval


class DWAModel(nnx.Module):
    """
    Full DWA forward pass:
        embed → PartA → retrieval → assembly → PartB → LM head

    The pool EMA (non-trainable) tracks per-vector utilization for L_util.
    """

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs) -> None:
        self.cfg = cfg
        # Token embedding + positional (sinusoidal, fixed)
        self.embed = nnx.Embed(cfg.vocab_size, cfg.d_A, rngs=rngs)
        self.part_a = PartA(cfg, rngs)
        self.part_b = PartB(cfg, rngs)
        self.pool = VectorPool(cfg, rngs)
        self.retrieval = MultiAspectRetrieval(cfg, rngs)
        self.assembler = WeightAssembler(cfg, rngs)
        self.lm_head = nnx.Linear(cfg.d_B, cfg.vocab_size, use_bias=False, rngs=rngs)
        # Non-trainable EMA of per-vector utilization [N]
        self.pool_ema = nnx.Variable(jnp.zeros(cfg.N))

    def __call__(
        self,
        input_ids: jnp.ndarray,  # [B, seq_len]
        lambda_val: float,        # current retrieval sharpness
        is_warmup: bool,          # static — controls warmup vs gate selection
    ) -> tuple[jnp.ndarray, dict]:
        """
        Returns:
            logits:  [B, seq_len, vocab_size]
            metrics: dict with alphas, indices, W, aux-loss components
        """
        cfg = self.cfg

        # Sinusoidal position encoding (added to embeddings)
        x = self.embed(input_ids)           # [B, T, d_A]
        T = x.shape[1]
        pos = _sinusoidal_pos_enc(T, cfg.d_A)
        x = x + pos[None]                   # broadcast over batch

        # Part A
        h_A = self.part_a(x)                # [B, T, d_A]

        # Hidden state for retrieval: pool over sequence (mean of last tokens)
        z = h_A.mean(axis=1)               # [B, d_A]

        # Pool keys (recomputed each forward pass; XLA fuses the matmul)
        pool_keys = self.pool.compute_keys()   # [S, N, d_k]

        # Retrieval
        alphas, indices = self.retrieval(z, pool_keys, lambda_val, is_warmup)
        # alphas: [B, k_max], indices: [B, k_max]

        # Gather pool vectors for retrieved indices: [B, k_max, D]
        gathered = self.pool.vectors[...][indices]

        # Assemble W and apply residual
        h_mid, W = self.assembler(h_A, gathered, alphas)   # [B, T, d_B]

        # Part B
        h_out = self.part_b(h_mid)          # [B, T, d_B]

        # LM head
        logits = self.lm_head(h_out)        # [B, T, vocab_size]

        metrics = {
            "alphas": alphas,
            "indices": indices,
            "W": W,
            "pool_keys": pool_keys,
            "W_base": self.assembler.W_base[...],
        }
        return logits, metrics


def _sinusoidal_pos_enc(seq_len: int, d_model: int) -> jnp.ndarray:
    """Standard sinusoidal position encoding [seq_len, d_model]."""
    pos = jnp.arange(seq_len)[:, None]           # [T, 1]
    i = jnp.arange(d_model // 2)[None, :]        # [1, d/2]
    angle = pos / (10000 ** (2 * i / d_model))
    enc = jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)
    # If d_model is odd, drop last column
    return enc[:, :d_model]


def forward_and_loss(
    model: DWAModel,
    input_ids: jnp.ndarray,
    lambda_val: float,
    is_warmup: bool,
    tcfg: TrainConfig,
    aux_on: bool,
) -> tuple[jnp.ndarray, dict]:
    """Combined forward + loss for use with nnx.value_and_grad."""
    logits, metrics = model(input_ids, lambda_val, is_warmup)
    l_task = task_loss(logits, input_ids)

    if aux_on:
        aux = aux_losses(
            metrics["alphas"],
            metrics["indices"],
            metrics["pool_keys"],
            metrics["W"],
            metrics["W_base"],
            model.pool_ema[...],
            model.cfg,
            tcfg,
        )
        total_loss = l_task + aux["total_aux"]
    else:
        aux = {k: jnp.zeros(()) for k in ("l_util", "l_div", "l_norm", "l_sparse", "total_aux")}
        total_loss = l_task

    info = {**aux, "l_task": l_task, "loss": total_loss,
            "alphas": metrics["alphas"], "indices": metrics["indices"]}
    return total_loss, info
