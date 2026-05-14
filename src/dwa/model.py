"""DWAModel — top-level NNX module orchestrating the full forward pass."""

import jax
import jax.numpy as jnp
from flax import nnx

from .assembly import WeightAssembler
from .assembly_pallas import assemble_jax, pallas_assemble, shard_pallas_assemble
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
        input_ids: jnp.ndarray,          # [B, seq_len]
        lambda_val: float,                # current retrieval sharpness
        is_warmup: bool,                  # static — controls warmup vs gate
        key_cache: jnp.ndarray | None = None,  # [S, N, d_k] pre-computed keys
        use_pallas: bool = True,          # use Pallas assembly kernel
        mesh=None,                        # jax.sharding.Mesh for shard_map Pallas
    ) -> tuple[jnp.ndarray, dict]:
        """
        Returns:
            logits:  [B, seq_len, vocab_size]
            metrics: dict with alphas, indices, W, aux-loss components

        key_cache: if provided (recommended), skip the [N,D]×[D,d_k] key
            projection matmul — use the pre-computed cache instead.
            Compute it once per window with compute_key_cache() and pass in.
        """
        cfg = self.cfg

        # Sinusoidal position encoding
        x = self.embed(input_ids)           # [B, T, d_A]
        T = x.shape[1]
        pos = _sinusoidal_pos_enc(T, cfg.d_A)
        x = x + pos[None]

        # Part A
        h_A = self.part_a(x)                # [B, T, d_A]
        z = h_A.mean(axis=1)               # [B, d_A] — retrieval query

        # Key cache: use provided cache or compute on-the-fly
        if key_cache is None:
            pool_keys = self.pool.compute_keys()   # [S, N, d_k] — expensive
        else:
            pool_keys = key_cache                  # [S, N, d_k] — free

        # Retrieval
        alphas, indices = self.retrieval(z, pool_keys, lambda_val, is_warmup)

        # Gather pool vectors: [B, k_max, D]; cast to float32 if pool is bfloat16
        gathered = self.pool.vectors[...][indices]
        if gathered.dtype != jnp.float32:
            gathered = gathered.astype(jnp.float32)

        # Assembly — Pallas kernel keeps W in VMEM; falls back to pure JAX
        W_base = self.assembler.W_base[...]
        b_base = self.assembler.b_base[...]
        gamma  = self.assembler.gamma[...]

        if use_pallas and mesh is not None:
            h_mid_no_ln, W = shard_pallas_assemble(
                gathered, alphas, h_A, W_base, b_base, gamma,
                cfg.d_B, cfg.r, cfg.d_A, mesh,
            )
            h_mid = self.assembler.layer_norm(h_mid_no_ln)
        elif use_pallas:
            h_mid_no_ln, W = pallas_assemble(
                gathered, alphas, h_A, W_base, b_base, gamma,
                cfg.d_B, cfg.r, cfg.d_A,
            )
            h_mid = self.assembler.layer_norm(h_mid_no_ln)
        else:
            h_mid_no_ln, W = assemble_jax(
                gathered, alphas, h_A, W_base, b_base, gamma,
                cfg.d_B, cfg.r, cfg.d_A,
            )
            h_mid = self.assembler.layer_norm(h_mid_no_ln)

        # Part B
        h_out = self.part_b(h_mid)          # [B, T, d_B]
        logits = self.lm_head(h_out)        # [B, T, vocab_size]

        metrics = {
            "alphas": alphas,
            "indices": indices,
            "W": W,
            "pool_keys": pool_keys,
            "W_base": W_base,
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
    key_cache: jnp.ndarray | None = None,
    use_pallas: bool = True,
    mesh=None,
) -> tuple[jnp.ndarray, dict]:
    """Combined forward + loss for use with nnx.value_and_grad."""
    logits, metrics = model(input_ids, lambda_val, is_warmup, key_cache, use_pallas, mesh)
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
