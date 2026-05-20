"""VectorPool — stores N×D parameter matrix and S key projection heads or maps indices dynamically."""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

from .config import DWAConfig


class PoolGeneratorMLP(nnx.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dims: tuple[int, ...], rngs: nnx.Rngs, dtype=None):
        dims = [in_features] + list(hidden_dims) + [out_features]
        self.num_layers = len(dims) - 1
        for i in range(self.num_layers):
            setattr(self, f"layer_{i}", nnx.Linear(dims[i], dims[i+1], dtype=dtype, rngs=rngs))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i in range(self.num_layers - 1):
            layer = getattr(self, f"layer_{i}")
            x = jax.nn.silu(layer(x))
        last_layer = getattr(self, f"layer_{self.num_layers - 1}")
        return last_layer(x)


class DynamicVectors:
    def __init__(self, pool_module):
        self.pool_module = pool_module

    def __getitem__(self, idx):
        if idx is Ellipsis:
            return self.pool_module.generate_all_vectors()
        return self.pool_module.generate_all_vectors()[idx]

    def __setitem__(self, idx, value):
        if idx is Ellipsis:
            if self.pool_module.cfg.use_hypernetwork:
                # Set all pool parameters to zero/corrupted for checkpoint verification tests
                state = nnx.state(self.pool_module, nnx.Param)
                flat, treedef = jax.tree_util.tree_flatten(state)
                corrupted_flat = [jnp.zeros_like(x) for x in flat]
                corrupted_state = jax.tree_util.tree_unflatten(treedef, corrupted_flat)
                nnx.update(self.pool_module, corrupted_state)
            else:
                self.pool_module._vectors.value = value
        else:
            if self.pool_module.cfg.use_hypernetwork:
                raise NotImplementedError("Direct row assignment not supported in hypernetwork mode")
            else:
                self.pool_module._vectors.value = self.pool_module._vectors.value.at[idx].set(value)

    @property
    def value(self):
        return self.pool_module.generate_all_vectors()

    @property
    def dtype(self):
        if self.pool_module.cfg.use_hypernetwork:
            return jnp.bfloat16 if self.pool_module.cfg.bf16_pool else jnp.float32
        else:
            return self.pool_module._vectors.value.dtype


class VectorPool(nnx.Module):
    """
    Stores the N pool vectors (each of dimension D) and the S key-projection
    matrices used for multi-aspect retrieval.

    Supports either a standard parameter matrix or a hypernetwork-based pool generator.
    """

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs, pool_vectors=None) -> None:
        self.cfg = cfg
        pool_dtype = jnp.bfloat16 if cfg.bf16_pool else jnp.float32

        if cfg.use_hypernetwork:
            # 1. Initialize coordinate embeddings [N, d_emb]
            # Handle sharding if pool_vectors is sharded
            if pool_vectors is not None and hasattr(pool_vectors, "sharding"):
                sharding = pool_vectors.sharding
                mesh = sharding.mesh
                n_model = mesh.shape.get("model", 1)
                N_local = cfg.N // n_model
                emb_sharding = NamedSharding(mesh, P("model", None))
                
                idx_map = emb_sharding.addressable_devices_indices_map((cfg.N, cfg.d_emb))
                per_device_arrays = []
                rng = rngs.params()
                for device in emb_sharding.addressable_devices:
                    idx_tuple = idx_map[device]
                    row_start = idx_tuple[0].start or 0
                    m_idx = row_start // N_local
                    with jax.default_device(device):
                        rng_dev = jax.device_put(rng, device)
                        key = jax.random.fold_in(rng_dev, m_idx)
                        key = jax.random.fold_in(key, 999) 
                        shard = (jax.random.normal(key, (N_local, cfg.d_emb)) * 0.02).astype(pool_dtype)
                    per_device_arrays.append(shard)
                embeddings_val = jax.make_array_from_single_device_arrays(
                    (cfg.N, cfg.d_emb), emb_sharding, per_device_arrays
                )
                self.embeddings = nnx.Param(embeddings_val)
            else:
                self.embeddings = nnx.Param(
                    (jax.random.normal(rngs.params(), (cfg.N, cfg.d_emb)) * 0.02).astype(pool_dtype)
                )

            # 2. Initialize the MLP Generator
            self.mlp = PoolGeneratorMLP(cfg.d_emb, cfg.D, cfg.mlp_hidden_dims, rngs, dtype=pool_dtype)
            
            # 3. Dynamic vectors interface wrapper
            self.vectors = DynamicVectors(self)
        else:
            if pool_vectors is not None:
                self.vectors = nnx.Param(pool_vectors)
            else:
                self.vectors = nnx.Param(
                    (jax.random.normal(rngs.params(), (cfg.N, cfg.D)) * 0.02).astype(pool_dtype)
                )

        # Key projections per aspect: [S, D, d_k]
        self.key_proj = nnx.Param(
            (jax.random.normal(rngs.params(), (cfg.S, cfg.D, cfg.d_k))
             * (cfg.D ** -0.5)).astype(pool_dtype)
        )

    def generate_all_vectors(self) -> jnp.ndarray:
        """Runs the MLP on all embeddings to produce the full [N, D] pool."""
        return self.mlp(self.embeddings[...])

    def compute_keys(self) -> jnp.ndarray:
        """
        Project all pool vectors through each aspect key projection.
        Returns: [S, N, d_k] in float32 regardless of storage dtype.
        """
        if self.cfg.use_hypernetwork:
            vecs = self.generate_all_vectors().astype(jnp.float32)
        else:
            vecs = self.vectors[...].astype(jnp.float32)
        kp = self.key_proj[...].astype(jnp.float32)
        return jnp.einsum("nd,sda->sna", vecs, kp)

    def get_factors(self, indices: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Gather vectors for given indices and split into low-rank factors.

        indices: [k] integer indices into pool
        Returns: U [k, d_B, r], V [k, r, d_A], b [k, d_B]
        """
        cfg = self.cfg
        s1, s2, s3 = cfg.factor_split
        if cfg.use_hypernetwork:
            emb = self.embeddings[...][indices]
            vecs = self.mlp(emb)
        else:
            vecs = self.vectors[...][indices]          # [k, D]
        U = vecs[:, :s1].reshape(-1, cfg.d_B, cfg.r)        # [k, d_B, r]
        V = vecs[:, s1:s2].reshape(-1, cfg.r, cfg.d_A)      # [k, r, d_A]
        b = vecs[:, s2:s3]                                   # [k, d_B]
        return U, V, b

    def export_dense_pool(self, rngs: nnx.Rngs) -> "VectorPool":
        """
        Creates a new VectorPool instance in standard (dense) mode
        initialized with the generated vectors from this hypernetwork pool.
        """
        if not self.cfg.use_hypernetwork:
            return self

        vecs = self.generate_all_vectors()
        kp = self.key_proj[...]

        import copy
        new_cfg = copy.deepcopy(self.cfg)
        new_cfg.use_hypernetwork = False

        dense_pool = VectorPool(new_cfg, rngs, pool_vectors=vecs)
        dense_pool.key_proj[...] = kp
        return dense_pool
