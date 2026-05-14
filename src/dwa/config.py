"""DWAConfig and TrainConfig — single source of truth for all hyperparameters."""

from dataclasses import dataclass, field


@dataclass
class DWAConfig:
    # Vector pool
    D: int = 16384     # pool vector dim
    N: int = 65536     # pool size

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
    n_layers_A: int = 6        # Transformer layers in Part A
    n_layers_B: int = 6        # Transformer layers in Part B
    ffn_mult: int = 4          # FFN hidden = ffn_mult * d

    # Vocabulary / sequence
    vocab_size: int = 32000
    seq_len: int = 512

    # Assembly init
    gamma_init: float = 0.01   # LoRA-style residual scalar init

    def __post_init__(self):
        needed = self.d_B * self.r + self.r * self.d_A + self.d_B
        assert self.D >= needed, (
            f"D={self.D} < needed={needed} "
            f"(d_B*r + r*d_A + d_B = {self.d_B}*{self.r} + {self.r}*{self.d_A} + {self.d_B})"
        )
        assert self.d_A == self.d_B, "d_A must equal d_B for the residual connection"
        assert self.d_A % self.n_heads == 0, "d_A must be divisible by n_heads"

    @classmethod
    def small(cls) -> "DWAConfig":
        """Fast iteration / testing config."""
        return cls(
            D=2048, N=512, d_A=64, d_B=64, r=4,
            k_max=8, S=2, d_k=32,
            n_heads=2, n_layers_A=1, n_layers_B=1,
            ffn_mult=4, vocab_size=1000, seq_len=64,
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
