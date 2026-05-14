# Dynamic Weight Assembly (DWA) — Project Architecture

## 1. High-Level System Architecture

One model split into two halves. The middle layer weight matrix is dynamically assembled from a pool of vectors per input. Part A produces a query → retrieves relevant vectors → vectors are reshaped into low-rank matrix factors → assembled middle layer → Part B generates output.

```mermaid
graph TB
    subgraph "Input Pipeline"
        X["Input x"]
    end

    subgraph "Part A"
        PA["Part A Encoder<br/>(Transformer blocks)"]
        LN_A["LayerNorm"]
    end

    subgraph "Retrieval Engine"
        MAQ["Multi-Aspect Queries<br/>q_s = W_Q_s · z"]
        CS["Cosine Similarity<br/>s_i_s = cos(q_s, k_i_s)"]
        SIG["Sigmoid Gate<br/>g_i = σ(λ·(s_i − τ))"]
        NORM["Normalized Weights<br/>α_i = g_i·exp(s_i/T) / Σ"]
    end

    subgraph "Vector Pool"
        POOL["VectorPool<br/>N=65536 vectors, D=16384"]
        KP["Key Projections<br/>k_i_s = W_K_s · v_i"]
        FACTOR["Factorization<br/>v_i → (U_i, V_i, b_i)"]
    end

    subgraph "Assembly"
        WBASE["W_base + b_base<br/>(learnable base weights)"]
        ASSEM["W = W_base + Σ α_i·(U_i@V_i)<br/>b = b_base + Σ α_i·b_i"]
        RESIDUAL["h_mid = LayerNorm(h_A + γ · h_A @ W^T + b)"]
    end

    subgraph "Part B"
        PB["Part B Decoder<br/>(Transformer blocks)"]
    end

    subgraph "Output"
        Y["Output y"]
    end

    X --> PA --> LN_A
    LN_A -->|"z (hidden state)"| MAQ
    POOL --> KP
    KP --> CS
    MAQ --> CS --> SIG --> NORM
    NORM -->|"α_i weights"| ASSEM
    POOL --> FACTOR
    FACTOR -->|"(U_i, V_i, b_i)"| ASSEM
    WBASE --> ASSEM
    LN_A -->|"h_A"| RESIDUAL
    ASSEM -->|"W, b"| RESIDUAL
    RESIDUAL -->|"h_mid"| PB --> Y
```

## 2. Module Dependency Graph

```mermaid
graph TD
    CONFIG["config.py<br/><i>DWAConfig, TrainConfig</i>"]
    POOL["pool.py<br/><i>VectorPool</i>"]
    RETR["retrieval.py<br/><i>MultiAspectRetrieval</i>"]
    ASSEM["assembly.py<br/><i>WeightAssembler</i>"]
    PARTS["parts.py<br/><i>PartA, PartB</i>"]
    MODEL["model.py<br/><i>DWAModel</i>"]
    LOSSES["losses.py<br/><i>AuxLosses</i>"]
    SCHED["schedule.py<br/><i>PhaseScheduler</i>"]
    UTILS["utils.py<br/><i>cosine_sim, etc.</i>"]

    MODEL --> PARTS
    MODEL --> POOL
    MODEL --> RETR
    MODEL --> ASSEM
    MODEL --> LOSSES
    MODEL --> SCHED

    RETR --> POOL
    RETR --> UTILS
    ASSEM --> POOL
    ASSEM --> UTILS

    POOL --> CONFIG
    RETR --> CONFIG
    ASSEM --> CONFIG
    PARTS --> CONFIG
    MODEL --> CONFIG
    LOSSES --> CONFIG
    SCHED --> CONFIG

    style CONFIG fill:#f9f,stroke:#333,stroke-width:2px
    style MODEL fill:#bbf,stroke:#333,stroke-width:2px
    style POOL fill:#bfb,stroke:#333,stroke-width:2px
    style RETR fill:#fbb,stroke:#333,stroke-width:2px
    style ASSEM fill:#fbb,stroke:#333,stroke-width:2px
```

## 3. Forward Pass Data Flow

```mermaid
flowchart LR
    subgraph "Forward Pass"
        direction LR
        A["x"] --> B["PartA(x)"]
        B --> C["z = h_A"]
        C --> D["W_Q^s · z<br/>→ queries"]
        C --> E["h_A (residual)"]
        D --> F["cosine sim<br/>with pool keys"]
        F --> G["sigmoid gate<br/>g_i = σ(λ(s_i − τ))"]
        G --> H["α_i weights<br/>(top k_max)"]
        H --> I["gather v_i<br/>from pool"]
        I --> J["reshape to<br/>(U_i, V_i, b_i)"]
        J --> K["W = W_base + Σα_i·U_i@V_i"]
        K --> L["h_mid = LN(h_A + γ·h_A·W^T + b)"]
        L --> M["PartB(h_mid)"]
        M --> N["ŷ"]
        E --> L
    end
```

## 4. Training Pipeline — Three-Phase Schedule

```mermaid
stateDiagram-v2
    [*] --> Warmup
    Warmup --> GateOn: step ≥ 1K
    GateOn --> Sharpen: step ≥ 10K

    state Warmup {
        [*] --> FixedTopK
        FixedTopK: λ = N/A (fixed softmax)
        FixedTopK: k = 16 (hardcoded)
        FixedTopK: γ = 0.01 (frozen)
        FixedTopK: LR warmup
    }

    state GateOn {
        [*] --> SigmoidActive
        SigmoidActive: λ = 1.0 → 5.0 (annealed)
        SigmoidActive: k = dynamic (sigmoid-gated)
        SigmoidActive: γ = growing
        SigmoidActive: Aux losses enabled
    }

    state Sharpen {
        [*] --> Sharpening
        Sharpening: λ = 5.0 → 10.0 (annealed)
        Sharpening: k = dynamic
        Sharpening: γ = free
        Sharpening: Cosine LR decay
    }
```

| Phase | Steps | λ | k | γ | Notes |
|-------|-------|---|---|---|-------|
| 1 — Warmup | 0–1K | N/A (fixed top-16) | 16 fixed | 0.01 | Softmax over top-16, warmup LR |
| 2 — Gate On | 1K–10K | 1.0 → 5.0 | dynamic | growing | Enable sigmoid gate, aux losses |
| 3 — Sharpen | 10K+ | 5.0 → 10.0 | dynamic | free | Sharper selection, cosine decay |

## 5. Auxiliary Losses

```mermaid
graph TB
    subgraph "Loss Components"
        L_TASK["L_task<br/>Cross-entropy / MSE"]
        L_UTIL["L_util<br/>−Σ log(1 − exp(−β·EMA(α_i)))<br/><i>Prevent dead pool vectors</i>"]
        L_DIV["L_div<br/>cosine penalty between retrieved keys<br/><i>Prevent key collapse</i>"]
        L_NORM["L_norm<br/>‖W − W_base‖²_F<br/><i>Prevent assembly explosion</i>"]
        L_SPARSE["L_sparse<br/>−Σ α_i log(α_i)<br/><i>Weight entropy regularization</i>"]
    end

    L_TOTAL["L_total = L_task<br/>+ λ_util·L_util<br/>+ λ_div·L_div<br/>+ λ_norm·L_norm<br/>+ λ_sparse·L_sparse"]

    L_TASK --> L_TOTAL
    L_UTIL --> L_TOTAL
    L_DIV --> L_TOTAL
    L_NORM --> L_TOTAL
    L_SPARSE --> L_TOTAL

    style L_TOTAL fill:#ff9,stroke:#333,stroke-width:2px
```

## 6. Dual Gradient Path — Key Innovation

Both retrieval and assembly gradients flow through the **same** pool vectors, creating a self-reinforcing loop: retrieval shapes what gets stored, and storage shapes what gets retrieved.

```mermaid
graph LR
    subgraph "Vector v_i"
        V["v_i ∈ ℝ^D"]
    end

    subgraph "Retrieval Path"
        KP["W_K^s · v_i"]
        CS2["→ k_i^s"]
        GR["→ ∂L/∂k_i^s"]
        GKR["→ (W_K^s)^T · ∂L/∂k_i^s"]
    end

    subgraph "Assembly Path"
        RS["reshape: v_i → (U_i, V_i, b_i)"]
        ASM["W = W_base + Σ α_i·(U_i@V_i)"]
        GA["→ ∂L/∂U_i, ∂L/∂V_i, ∂L/∂b_i"]
        GF["→ flatten to ∂L/∂v_i^assembly"]
    end

    V --> KP --> CS2 --> GR --> GKR
    V --> RS --> ASM --> GA --> GF

    GKR -->|"who should retrieve you?"| TOTAL["∂L/∂v_i<br/>(both paths merged)"]
    GF -->|"what transformation to store?"| TOTAL

    style TOTAL fill:#bbf,stroke:#333,stroke-width:3px
```

## 7. Class Hierarchy

```mermaid
classDiagram
    class DWAConfig {
        +int D
        +int d_A
        +int d_B
        +int r
        +int N
        +int k_max
        +int S
        +float gamma_init
        +small() DWAConfig
    }

    class TrainConfig {
        +dict phases
        +dict per_component_lr
        +dict aux_weights
    }

    class VectorPool {
        +Parameter vectors
        +ModuleList key_projections
        +forward(indices) factors, keys
        +get_all_keys() keys
    }

    class MultiAspectRetrieval {
        +ModuleList query_projections
        +Parameter aspect_weights
        +Parameter threshold
        +float sharpness
        +forward(z, pool_keys) alphas, indices
    }

    class WeightAssembler {
        +Parameter W_base
        +Parameter b_base
        +Parameter gamma
        +forward(h_A, factors, alphas) h_mid
    }

    class PartA {
        +forward(x) z, h_A
    }

    class PartB {
        +forward(h_mid) logits
    }

    class DWAModel {
        +PartA part_a
        +PartB part_b
        +VectorPool pool
        +MultiAspectRetrieval retrieval
        +WeightAssembler assembler
        +forward(x) logits, metrics
    }

    class AuxLosses {
        +forward(alphas, keys, W, W_base, pool_ema) losses_dict
    }

    class PhaseScheduler {
        +get_phase(step) str
        +get_lambda(step) float
        +get_lr_scale(step) float
        +should_enable_aux(step) bool
    }

    DWAModel --> PartA
    DWAModel --> PartB
    DWAModel --> VectorPool
    DWAModel --> MultiAspectRetrieval
    DWAModel --> WeightAssembler
    MultiAspectRetrieval --> DWAConfig
    VectorPool --> DWAConfig
    WeightAssembler --> DWAConfig
```

## 8. Module Responsibilities

| Module | File | Responsibility |
|--------|------|----------------|
| **Config** | `config.py` | `DWAConfig` (model hyperparams) and `TrainConfig` (phase schedule, per-component LR, aux loss weights). Single source of truth — all modules derive dimensions/defaults from here. |
| **VectorPool** | `pool.py` | Stores the N×D parameter matrix and S key projection heads. Handles vector gathering by indices and key computation for all pool vectors. The dual-gradient convergence point — both retrieval and assembly gradients update `vectors`. |
| **MultiAspectRetrieval** | `retrieval.py` | Computes S aspect queries from Part A hidden state, cosine similarity against pool keys, sigmoid-gated selection with learnable threshold τ and annealed sharpness λ. Returns normalized α weights and selected indices. |
| **WeightAssembler** | `assembly.py` | Reshapes retrieved vectors into (U, V, b) factors, assembles W and b via weighted sum with base weights, and computes the residual connection `h_mid = LN(h_A + γ·h_A@W^T + b)`. Owns W_base, b_base, γ. |
| **PartA / PartB** | `parts.py` | Standard Transformer encoder stacks. Part A produces the hidden state z and h_A; Part B consumes h_mid and produces output logits. Architecture-agnostic — can be any seq2seq backbone. |
| **DWAModel** | `model.py` | Top-level `nn.Module` orchestrating the full forward pass: PartA → retrieval → assembly → PartB. Returns logits + auxiliary metrics dict. Wires the dual-gradient path. |
| **AuxLosses** | `losses.py` | Computes all four auxiliary losses (util, div, norm, sparse) plus the task loss weighting. Takes α weights, retrieved keys, assembled W, W_base, and pool EMA as inputs. |
| **PhaseScheduler** | `schedule.py` | Controls the three-phase training schedule: warmup → gate on → sharpen. Manages λ annealing, k switching, aux loss gating, and per-component LR scaling. Stateless except for config. |
| **Utils** | `utils.py` | Shared math utilities: cosine similarity, normalization, EMA tracking. No state, pure functions. |

## 9. File Structure

```
matrix/
├── src/
│   └── dwa/
│       ├── __init__.py              # Package exports
│       ├── config.py                # DWAConfig, TrainConfig dataclasses
│       ├── pool.py                  # VectorPool (N×D parameter storage)
│       ├── retrieval.py             # MultiAspectRetrieval (sigmoid-gated)
│       ├── assembly.py              # WeightAssembler (factorization + assembly)
│       ├── parts.py                 # PartA, PartB (Transformer halves)
│       ├── model.py                 # DWAModel (top-level nn.Module)
│       ├── losses.py                # AuxLosses (util, div, norm, sparse)
│       ├── schedule.py              # PhaseScheduler (3-phase training logic)
│       └── utils.py                 # cosine_similarity, normalize, etc.
├── tests/
│   ├── conftest.py                  # Shared fixtures (small config, model instances)
│   ├── test_config.py               # Config validation / defaults
│   ├── test_pool.py                 # VectorPool indexing, reshape, gradients
│   ├── test_retrieval.py            # Cosine sim, sigmoid gate, top-k
│   ├── test_assembly.py             # Factorization, W assembly, residual
│   ├── test_model.py                # End-to-end forward/backward, shapes
│   ├── test_losses.py               # Each aux loss independently
│   └── test_schedule.py             # Phase transitions, λ annealing
├── train.py                         # Training entry point (argparse → Trainer)
├── main.py                          # CLI entry point
├── docs/
│   ├── ARCHITECTURE.md              # Algorithm & math spec
│   ├── PROJECT_ARCHITECTURE.md      # This file — project architecture & diagrams
│   ├── TPU_TRAINING_STRATEGY.md     # TPU training optimization strategy
│   ├── CLAUDE.md                    # Behavioral guidelines
├── pyproject.toml                   # Project metadata & dependencies
└── README.md                        # Project overview
```

## 10. Design Principles

| Principle | Decision |
|-----------|----------|
| **Config-driven** | Single `DWAConfig` dataclass — everything derives from it; `DWAConfig.small()` for fast iteration |
| **Separation of concerns** | `pool.py` = storage, `retrieval.py` = selection, `assembly.py` = construction — each independently testable |
| **Dual-gradient transparency** | Both gradient paths flow through `VectorPool.vectors` — no manual gradient wiring needed, PyTorch autograd handles it |
| **Phase-aware training** | `PhaseScheduler` controls λ annealing, k switching, aux loss gating — no if/else scattered in model code |
| **Testability** | Every module has isolated shape/gradient tests; `small()` config runs full forward+backward in <1s |
| **Per-component LR** | `TrainConfig.per_component_lr` → optimizer param groups, not hardcoded |

## 11. Dimensionality Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| D (vector dim) | 16384 ≈ 2^14 | Close to ~16000 |
| d_A, d_B (hidden) | 256 | Symmetric, power of 2 |
| r (assembly rank) | 24 | Polysemantic meaning slots per vector |
| S (retrieval aspects) | 4 | Multi-facet matching |
| N (pool size) | 65536 | ~1.07B params in pool alone |
| k_max (retrieved) | 16 | Effective rank = 16×24 = 384 > 256 ✓ |

**Small validation config**: D=2048, d_A=d_B=64, r=4, N=512, k_max=8, S=2

## 12. Novelty vs Prior Work

| Work | What it does | DWA difference |
|------|-------------|---------------|
| PKM | Sum retrieved embeddings | Assemble into **weight matrices** |
| LoRA | Fixed low-rank adaptation | Dynamically **retrieved** per input |
| HyperNetworks | Generate weights from scratch | From **retrievable pool** — interpretable, modular |
| MoE | Route to full expert networks | Low-rank vector **fragments** — 1000× smaller |
| RAG | Retrieve text, prepend to context | Knowledge **IS** the computation (weight deltas) |