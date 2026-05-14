# Dynamic Weight Assembly (DWA) — Architecture Design

## Overview

One model split into two halves. The middle layer weight matrix is **dynamically assembled from a pool of vectors** per each input. Part A produces a query → retrieves relevant vectors → vectors are reshaped into low-rank matrix factors → assembled middle layer → Part B generates output.

## 1. Vector → Matrix Assembly: Factorized Rank-R

Each pool vector `v_i ∈ ℝ^D` is reshaped into three components:
- `U_i ∈ ℝ^(d_B × r)` — left factor (first `d_B × r` elements)
- `V_i ∈ ℝ^(r × d_A)` — right factor (next `r × d_A` elements)  
- `b_i ∈ ℝ^d_B` — bias contribution (next `d_B` elements)

### Assembly Formula

```
W_assembled = W_base + Σ_i α_i · (U_i @ V_i)
b_assembled = b_base + Σ_i α_i · b_i
```

### Forward Pass Through Middle

```
h_mid = LayerNorm(h_A + γ · h_A @ W_assembled^T + b)
```

- `γ` initialized to 0.01 (LoRA-style residual — starts as tiny perturbation)
- `W_base` initialized small (~0.01·𝒰) — ensures model works with zero retrieval
- Effective rank = k_max × r > d — full rank achievable

### Why Factorized?

| Approach | Gradient Quality | Param Efficiency | Polysemantic? |
|----------|-----------------|-----------------|---------------|
| Direct reshape | Flat, no structure | D must = d_B×d_A | No |
| **Factorized** | **Structured, spectral reg** | **4× expansion** | **Yes: r meaning slots** |
| Learned projection | Bottlenecked by proj | D×(d_B·d_A) params! | Only if structured |

Critical gradient formulas:
```
∂L/∂U_i = α_i · (∂L/∂W) @ V_i^T
∂L/∂V_i = α_i · U_i^T @ (∂L/∂W)
∂L/∂α_i = ⟨U_i V_i, ∂L/∂W⟩_F
```

## 2. Retrieval: Multi-Aspect Sigmoid-Gated

### Why Not Simple Cosine Similarity?

A single similarity score cannot capture polysemantic matching. A query about "family" may need vectors matching on "kinship" AND "emotion" simultaneously.

### Step-by-Step

**Step 1 — Aspect Decomposition** (S aspects, like multi-head attention):

```
q^(s) = W_Q^(s) · z ∈ ℝ^{d_k}     (aspect queries from Part A)
k_i^(s) = W_K^(s) · v_i ∈ ℝ^{d_k}  (aspect keys from FULL vector)
```

**KEY**: The key projection uses the FULL vector (including U_i, V_i). This couples retrieval and storage — gradient from "this vector was useful" flows through W_K back to the same parameters that store matrix factors.

**Step 2 — Multi-Aspect Similarity**:

```
s_i^(s) = cosine(q^(s), k_i^(s))
s_i = Σ_s w_s · s_i^(s)    where w = softmax(learned_aspect_weights)
```

**Step 3 — Sigmoid-Gated Selection** (core novelty):

```
g_i = σ(λ · (s_i - τ))
```
- σ = sigmoid, λ = sharpness (annealed 1→10), τ = learnable threshold
- **Every vector gets gradient ≠ 0** (not just top-k)
- Vectors near threshold (s_i ≈ τ) get **strongest** gradient — they're learning to become useful

**Step 4 — Normalized Weights**:

```
α_raw_i = g_i · exp(s_i / T)
α_i = α_raw_i / Σ_j α_raw_j
```

Take top k_max for assembly (memory efficiency).

### Dual Gradient Path (THE Key Innovation)

```
∂L/∂v_i = Σ_s (W_K^(s))^T · (∂L/∂k_i^(s))   ← retrieval: "who should retrieve you?"
         + [vec(∂L/∂U_i) ; vec(∂L/∂V_i) ; ∂L/∂b_i]  ← assembly: "what transformation to store"
```

Both paths update the SAME parameters. Self-reinforcing: the retrieval shapes what gets stored.

## 3. Training Strategy

### Three-Phase Schedule

| Phase | Steps | λ | k | γ | Notes |
|-------|-------|---|---|---|-------|
| 1 — Warmup | 0–1K | N/A (fixed top-16) | 16 fixed | 0.01 | Softmax over top-16, warmup LR |
| 2 — Gate On | 1K–10K | 1.0 → 5.0 | dynamic | growing | Enable sigmoid gate, aux losses |
| 3 — Sharpen | 10K+ | 5.0 → 10.0 | dynamic | free | Sharper selection, cosine decay |

### Auxiliary Losses

```
L_total = L_task
        + λ_util · L_util      (prevent dead vectors: -Σ log(1 - exp(-β·EMA(α_i))))
        + λ_div · L_div         (prevent key collapse: cosine between retrieved keys)
        + λ_norm · L_norm       (prevent assembly explosion: ‖W - W_base‖²_F)
        + λ_sparse · L_sparse   (weight entropy: -Σ α_i log(α_i))
```

### Per-Component Learning Rates

pool=3e-5, parts=1e-4, retrieval_proj=1e-4, threshold/gamma=1e-3

## 4. Dimensionality

| Parameter | Value | Notes |
|-----------|-------|-------|
| D (vector dim) | 16384 ≈ 2^14 | Close to requested ~16000 |
| d_A, d_B | 256 | Symmetric, power of 2 |
| r (assembly rank) | 24 | Polysemantic meaning slots per vector |
| S (retrieval aspects) | 4 | Multi-facet matching |
| N (pool size) | 65536 | ~1.07B params |
| k_max | 16 | Effective rank = 16×24 = 384 > 256 ✓ |

**Small validation config**: D=2048, d_A=d_B=64, r=4, N=512, k_max=8, S=2

## 5. Novelty vs Prior Work

| Work | What it does | DWA difference |
|------|-------------|---------------|
| PKM | Sum retrieved embeddings | Assemble into WEIGHT MATRICES |
| LoRA | Fixed low-rank adaptation | Dynamically RETRIEVED per input |
| HyperNetworks | Generate weights from scratch | From RETRIEVABLE POOL — interpretable, modular |
| MoE | Route to full expert networks | Low-rank vector FRAGMENTS — 1000× smaller |
| RAG | Retrieve text, prepend to context | Knowledge IS the computation (weight deltas) |