"""DWAConfig and TrainConfig — single source of truth for all hyperparameters."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DWAConfig:
    # Vector pool
    D: int = 16384     # pool vector dim
    N: int = 65536     # pool size

    # IVF retrieval
    use_ivf: bool = True # Use two-stage IVF retrieval
    C: int = 128       # number of centroids
    m: int = 8         # number of clusters to search

    # Transformer hidden dims
    d_A: int = 256     # Part A hidden dim
    d_B: int = 256     # Part B hidden dim (must equal d_A for residual)

    # Assembly rank
    r: int = 24        # low-rank factors per vector

    # Retrieval
    k_max: int = 16    # top-k vectors retrieved per input
    S: int = 4         # number of retrieval aspects
    d_k: int = 64      # key/query dim per aspect
    T: float = 1.0     # temperature for normalized weights

    # Transformer architecture
    n_heads: int = 4           # attention heads in Part A and B
    n_kv_heads: int = 1        # KV heads (GQA); 1 = MQA, n_heads = MHA
    use_rope: bool = True      # Use Rotary Positional Embeddings
    n_layers_A: int = 6        # Transformer layers in Part A
    n_layers_B: int = 6        # Transformer layers in Part B
    ffn_mult: int = 4          # FFN hidden = ffn_mult * d

    # Vocabulary / sequence
    vocab_size: int = 32000
    seq_len: int = 2048

    # Assembly init
    gamma_init: float = 0.01   # LoRA-style residual scalar init

    # Memory / precision
    bf16_pool: bool = True   # store pool vectors in bfloat16 (halves gather bandwidth)
    compute_dtype: Any = None  # None=float32; set to jnp.bfloat16 for ~4× MXU throughput
    remat: bool = False        # gradient checkpointing: recompute activations, cut ~4× activation memory
    use_flash_attn: bool = False  # Pallas flash attention (hits VMEM limit inside scan+vjp; inference-only)

    def __post_init__(self):
        needed = self.d_B * self.r + self.r * self.d_A + self.d_B
        assert self.D >= needed, (
            f"D={self.D} < needed={needed} "
            f"(d_B*r + r*d_A + d_B = {self.d_B}*{self.r} + {self.r}*{self.d_A} + {self.d_B})"
        )
        assert self.d_A == self.d_B, "d_A must equal d_B for the residual connection"
        assert self.d_A % self.n_heads == 0, "d_A must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads={self.n_heads} must be divisible by n_kv_heads={self.n_kv_heads}"
        )
        if self.use_ivf:
            assert self.N % self.C == 0, (
                f"N={self.N} must be divisible by C={self.C} for IVF equal-size buckets"
            )

    @classmethod
    def small(cls) -> "DWAConfig":
        """Fast iteration / testing config."""
        return cls(
            D=2048, N=512, d_A=64, d_B=64, r=4,
            k_max=8, S=2, d_k=32,
            n_heads=2, n_layers_A=1, n_layers_B=1,
            ffn_mult=4, vocab_size=1000, seq_len=64,
        )

    @classmethod
    def medium(cls) -> "DWAConfig":
        """
        Fits on 8×TPU v5e-16GB.

        Pool+Adam memory per device: 16384×4096×12B ≈ 3.2GB
        Leaves ~12GB for activations, other params, and VMEM buffers.
        D=4096 ≥ d_B*r + r*d_A + d_B = 128*12 + 12*128 + 128 = 3200 ✓
        """
        return cls(
            D=4096, N=16384, d_A=128, d_B=128, r=12,
            k_max=16, S=4, d_k=64,
            n_heads=4, n_layers_A=4, n_layers_B=4,
            ffn_mult=4, vocab_size=32000, seq_len=256,
        )

    @classmethod
    def medium_mxu(cls, bf16: bool = False) -> "DWAConfig":
        """
        MXU-aligned variant of medium: r=128 so assembly matmuls are 128×128×128
        (perfect fit for the TPU v5e systolic array vs r=12 which fills ~9% of a tile).

        D=33024 (=258×128) ≥ d_B*r + r*d_A + d_B = 128*128 + 128*128 + 128 = 32896 ✓
        N reduced to 2048 to keep pool+Adam ≈ 800MB — same as medium.
        """
        return cls(
            D=33024, N=2048, d_A=128, d_B=128, r=128,
            k_max=16, S=4, d_k=64,
            n_heads=4, n_layers_A=4, n_layers_B=4,
            ffn_mult=4, vocab_size=32000, seq_len=256,
            bf16_pool=bf16,
        )

    @classmethod
    def full_wide(cls) -> "DWAConfig":
        """
        d=512 for ~3-4× better MXU fill vs d=256, same pool memory as --full.

        N=32768 × D=32768 = same 4 GB pool footprint as --full (N=65536 × D=16384).
        Pool+Adam/device with 4-way model parallel: 4 GB×3 / 4 = 3 GB — same budget.
        n_heads=8 (head_dim=64).  Needs D ≥ d_B*r + r*d_A + d_B = 25088 → D=32768 ✓
        """
        return cls(
            D=32768, N=32768, d_A=512, d_B=512, r=24,
            k_max=16, S=4, d_k=64,
            n_heads=8, n_kv_heads=1, n_layers_A=6, n_layers_B=6,
            ffn_mult=4, vocab_size=32000, seq_len=2048,
            bf16_pool=True,
        )

    @classmethod
    def large(cls) -> "DWAConfig":
        """
        2× medium: double pool size and transformer depth.
        Pool+Adam per device: 32768×4096×12B ≈ 1.6GB — fits 16GB TPU with room.
        """
        return cls(
            D=4096, N=32768, d_A=128, d_B=128, r=12,
            k_max=16, S=4, d_k=64,
            n_heads=4, n_layers_A=8, n_layers_B=8,
            ffn_mult=4, vocab_size=32000, seq_len=256,
        )

    @classmethod
    def pattern_test(cls) -> "DWAConfig":
        """
        Small model with 16-token vocab for fast pattern-learning verification.
        With period=4, seq=64: theoretical min loss ≈ 4/64 * log(16) ≈ 0.17 nats.
        Random baseline: log(16) ≈ 2.77 nats.  Converges in ~3000 steps.
        """
        return cls(
            D=2048, N=512, d_A=64, d_B=64, r=4,
            k_max=8, S=2, d_k=32,
            n_heads=2, n_layers_A=4, n_layers_B=4,
            ffn_mult=4, vocab_size=16, seq_len=64,
        )

    @property
    def factor_split(self) -> tuple[int, int, int]:
        """(split1, split2, split3) indices for slicing pool vectors into (U, V, b)."""
        s1 = self.d_B * self.r
        s2 = s1 + self.r * self.d_A
        s3 = s2 + self.d_B
        return s1, s2, s3


@dataclass
class TrainConfig:
    # Phase boundaries (in optimizer steps)
    warmup_steps: int = 1_000
    gate_on_steps: int = 10_000
    total_steps: int = 100_000

    # Sharpness annealing
    lambda_gate_start: float = 1.0
    lambda_gate_end: float = 5.0
    lambda_sharpen_end: float = 10.0

    # Auxiliary loss weights
    lambda_util: float = 0.01
    lambda_div: float = 0.01
    lambda_norm: float = 0.001
    lambda_sparse: float = 0.001

    # Per-component learning rates
    lr_pool: float = 3e-5
    lr_parts: float = 1e-4
    lr_retrieval: float = 1e-4
    lr_threshold: float = 1e-3

    # Training loop
    batch_size: int = 128        # total across all devices
    steps_per_window: int = 32   # steps inside one jax.lax.scan call
    seed: int = 42

    # LR warmup (cosine decay starts at gate_on_steps)
    lr_warmup_steps: int = 500

    # EMA decay for pool utilization tracking
    ema_decay: float = 0.99

    @property
    def lr_warmup(self) -> int:
        return self.lr_warmup_steps
