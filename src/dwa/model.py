"""DWAModel — top-level NNX module orchestrating the full forward pass."""

import jax
import jax.numpy as jnp
from flax import nnx

from .assembly import WeightAssembler
from .assembly_pallas import assemble_jax, pallas_assemble, shard_pallas_assemble
from .config import DWAConfig, TrainConfig
from .losses import aux_losses, task_loss
from .parts import PartA, PartB, precompute_rope_freqs
from .pool import VectorPool
from .retrieval import MultiAspectRetrieval


def _distributed_gather(pool_vecs, indices, mesh):
    """
    Gather pool vectors from a model-axis-sharded pool.

    Each device owns pool_vecs[N_local, D] (sharded on the 'model' axis).
    Each device knows only its own shard; indices are global [B_local, k_max].

    Implementation: each device gathers from its local shard, masks out
    entries that don't belong to it, then psum across the model axis
    to combine — only one shard contributes non-zero for each index.
    """
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    def _fn(local_pool, local_idx):
        N_local = local_pool.shape[0]
        my_shard = jax.lax.axis_index("model")
        local_gathered = local_pool[local_idx % N_local]          # [B, k, D]
        belongs = (local_idx // N_local == my_shard)[:, :, None]  # [B, k, 1]
        masked = jnp.where(belongs, local_gathered, jnp.zeros_like(local_gathered))
        return jax.lax.psum(masked, axis_name="model")            # [B, k, D]

    return shard_map(
        _fn,
        mesh=mesh,
        in_specs=(P("model", None), P("data", None)),
        out_specs=P("data", None, None),
        check_rep=False,
    )(pool_vecs, indices)


def _vocab_parallel_cross_entropy(
    h_out: jnp.ndarray,   # [B, T, d_B]  — sharded P("data", None, None)
    kernel: jnp.ndarray,  # [d_B, V]     — sharded P(None, "model")
    targets: jnp.ndarray, # [B, T]       — sharded P("data", None)
    mesh,
    n_model: int,
    V: int,
) -> jnp.ndarray:
    """
    Cross-entropy loss without materialising the full [B, T, V] logits tensor.

    Each device holds kernel_local [d_B, V/n_model] and computes its shard of
    the logits locally.  A two-collective distributed softmax then gives the
    exact same loss as the standard formula:
        loss = -log(exp(z_target) / sum_v exp(z_v))
             = log_sum_exp(all z) - z_target
    """
    from jax.experimental.shard_map import shard_map as _smap
    from jax.sharding import PartitionSpec as _P

    V_local = V // n_model

    def body(h_local, k_local, tgt_local):
        h_shifted   = h_local[:, :-1, :]    # [B_local, T-1, d_B]
        tgt_shifted = tgt_local[:, 1:]      # [B_local, T-1]
        B_local, T1, d = h_shifted.shape
        BT       = B_local * T1
        h_flat   = h_shifted.reshape(BT, d)
        tgt_flat = tgt_shifted.reshape(BT)

        local_logits = h_flat @ k_local     # [BT, V_local]

        # Partition function via psum (fully differentiable — no pmax needed).
        # Clip at ±60 prevents float32 overflow: max psum ≈ n_model*V_local*exp(60)
        # ≈ 4*8000*1.1e26 ≈ 3.6e30, well inside float32 range (~3.4e38).
        # In practice logits stay in [-10, 10] so the clip never triggers.
        exp_s    = jnp.exp(jnp.clip(local_logits, -60.0, 60.0))           # [BT, V_local]
        global_Z = jax.lax.psum(exp_s.sum(axis=-1), axis_name="model")    # [BT]

        # Gather the target token's logit from the shard that owns it
        shard_id = jax.lax.axis_index("model")
        offset   = shard_id * V_local
        in_shard = (tgt_flat >= offset) & (tgt_flat < offset + V_local)
        safe_idx = jnp.clip(tgt_flat - offset, 0, V_local - 1)
        mine     = jnp.where(in_shard, local_logits[jnp.arange(BT), safe_idx], 0.0)
        true_logit = jax.lax.psum(mine, axis_name="model")                # [BT]

        per_token = jnp.log(global_Z + 1e-8) - true_logit
        return jax.lax.pmean(per_token.mean(), axis_name="data")

    return _smap(
        body, mesh=mesh,
        in_specs=(_P("data", None, None), _P(None, "model"), _P("data", None)),
        out_specs=_P(),
        check_rep=False,
    )(h_out, kernel, targets)


class DWAModel(nnx.Module):
    """
    Full DWA forward pass:
        embed → PartA → retrieval → assembly → PartB → LM head

    The pool EMA (non-trainable) tracks per-vector utilization for L_util.
    """

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs, pool_vectors=None) -> None:
        self.cfg = cfg
        # Token embedding + positional (sinusoidal, fixed)
        self.embed = nnx.Embed(cfg.vocab_size, cfg.d_A, rngs=rngs)
        self.part_a = PartA(cfg, rngs)
        self.part_b = PartB(cfg, rngs)
        # pool_vectors: pre-sharded [N, D] array; if None, VectorPool generates randomly
        self.pool = VectorPool(cfg, rngs, pool_vectors=pool_vectors)
        self.retrieval = MultiAspectRetrieval(cfg, rngs)
        self.assembler = WeightAssembler(cfg, rngs)
        self.lm_head = nnx.Linear(cfg.d_B, cfg.vocab_size, use_bias=False,
                                   dtype=cfg.compute_dtype, rngs=rngs)
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
        compute_logits: bool = True,      # False when vocab_parallel handles the lm head
    ) -> tuple[jnp.ndarray | None, dict]:
        """
        Returns:
            logits:  [B, seq_len, vocab_size]
            metrics: dict with alphas, indices, W, aux-loss components

        key_cache: if provided (recommended), skip the [N,D]×[D,d_k] key
            projection matmul — use the pre-computed cache instead.
            Compute it once per window with compute_key_cache() and pass in.
        """
        cfg = self.cfg

        T = input_ids.shape[1]

        # Rotary Positional Embeddings or Sinusoidal
        if cfg.use_rope:
            cos, sin = precompute_rope_freqs(T, cfg.d_A // cfg.n_heads)
            pos = None
        else:
            cos = sin = None
            pos = _sinusoidal_pos_enc(T, cfg.d_A)

        x = self.embed(input_ids)           # [B, T, d_A]
        if pos is not None:
            x = x + pos[None]

        # Part A
        h_A = self.part_a(x, cos, sin, mesh)                # [B, T, d_A]
        z = h_A.mean(axis=1)               # [B, d_A] — retrieval query

        # Key cache: use provided cache or compute on-the-fly
        if key_cache is None:
            pool_keys = self.pool.compute_keys()   # [S, N, d_k] — expensive
        else:
            pool_keys = key_cache                  # [S, N, d_k] — free

        # Retrieval — mesh enables model-axis all-gather inside retrieval
        alphas, indices, soft_full = self.retrieval(z, pool_keys, lambda_val, is_warmup, mesh=mesh)

        # Gather pool vectors [B, k_max, D]; uses distributed gather when pool is
        # model-sharded so we never all-gather the full 4 GB pool across devices.
        pool_vecs = self.pool.vectors[...]
        use_dist = (
            mesh is not None
            and "model" in mesh.axis_names
            and mesh.shape["model"] > 1
        )
        if use_dist:
            gathered = _distributed_gather(pool_vecs, indices, mesh)
        else:
            gathered = pool_vecs[indices]
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
        h_out = self.part_b(h_mid, cos, sin, mesh)                    # [B, T, d_B]
        logits = self.lm_head(h_out) if compute_logits else None      # [B, T, vocab_size]

        # aux_losses indexes pool_keys with global indices; ensure it is fully
        # replicated so the gather works regardless of model sharding.
        if use_dist:
            from jax.sharding import NamedSharding, PartitionSpec as P
            pool_keys_full = jax.lax.with_sharding_constraint(
                pool_keys, NamedSharding(mesh, P(None, None, None))
            )  # all-gathers N_local → N on model axis; [S, N, d_k] = ~65 MB
        else:
            pool_keys_full = pool_keys

        metrics = {
            "alphas": alphas,
            "indices": indices,
            "soft_full": soft_full,
            "W": W,
            "pool_keys": pool_keys_full,
            "W_base": W_base,
            "h_out": h_out,
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
    from jax.sharding import NamedSharding, PartitionSpec as P

    n_model = mesh.shape.get("model", 1) if mesh is not None else 1
    use_vp  = model.cfg.vocab_parallel and n_model > 1

    logits, metrics = model(
        input_ids, lambda_val, is_warmup, key_cache, use_pallas, mesh,
        compute_logits=not use_vp,
    )

    if use_vp:
        # Shard lm_head kernel across model axis so each device holds [d_B, V/n_model].
        # Avoids materialising the full [B, T, V] logits tensor (saves ~4-8 GB/device).
        kernel = jax.lax.with_sharding_constraint(
            model.lm_head.kernel[...],
            NamedSharding(mesh, P(None, "model")),
        )
        l_task = _vocab_parallel_cross_entropy(
            metrics["h_out"], kernel, input_ids, mesh, n_model, model.cfg.vocab_size
        )
    else:
        l_task = task_loss(logits, input_ids)

    if aux_on:
        aux = aux_losses(
            metrics["alphas"],
            metrics["indices"],
            metrics["pool_keys"],
            metrics["W"],
            metrics["W_base"],
            metrics["soft_full"],
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
