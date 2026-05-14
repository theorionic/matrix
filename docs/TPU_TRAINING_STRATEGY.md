# TPU Training Strategy for DWA — Hybrid Pallas/XLA Architecture

## Design Goal

Minimize HBM round-trips during training by keeping the **assembly stage** (low-rank matrix accumulation from retrieved vectors) inside the TPU MXU/VMEM. The pool stays in HBM; the heavy lifting is fusing the gather + assembly + forward pass without intermediate CPU/HBM round-trips.

**Inference-time plan (unchanged):** Pool sits on CPU/HBM. Retrieve vectors per token and feed into generation.

**Training-time plan (this document):** Push large tokenized buffers to TPU once, run multi-step windows inside `jax.lax.scan`, and keep assembly in VMEM via a Pallas kernel.

---

## Core Constraint: Pool Size vs. VMEM

| Config | Pool Size | VMEM (TPU v4) | Fits in VMEM? |
|--------|-----------|---------------|---------------|
| Full   | 65536 × 16384 × 2 bytes ≈ **2 GB** | ~32 MB | **NO** |
| Small  | 512 × 2048 × 2 bytes ≈ **2 MB** | ~32 MB | **YES** |

The full pool **cannot** reside in VMEM. Any design claiming to keep the entire pool in on-chip memory is impossible on current TPU hardware. The pool **must** stay in HBM.

However, the **retrieved vectors** are tiny: `k_max = 16` vectors × 16384 dims × 2 bytes = **512 KB**. This fits comfortably in VMEM.

---

## Verdict: Hybrid Architecture (Not Pure Pallas)

| Stage | Tool | Data Movement | Rationale |
|-------|------|---------------|-----------|
| **Retrieval** (cosine sim over 65K) | XLA / `jax.lax` matmul | Pool read from HBM, streamed through MXU tiles | XLA already tiles large matmuls optimally across the MXU systolic array. Pallas cannot improve this — it would only reimplement XLA's scheduler. |
| **Sigmoid gate + top-k** | `jax.lax` ops | ~1 MB of scores in VMEM | Small enough to stay in VMEM; no custom kernel needed. |
| **Assembly** (16 vectors → W) | **Pallas kernel** | 512 KB gathered vectors in VMEM; W accumulated in VMEM | Low-rank outer products (`U_i @ V_i`) accumulated without writing intermediates back to HBM. This is where Pallas shines. |

**Why not one monolithic Pallas kernel?**
A single kernel doing 65K-retrieval + assembly would need to stream 2 GB of pool data through VMEM tiles. XLA already generates optimal HBM→VMEM→MXU pipelining for matmul. A custom Pallas kernel would duplicate XLA's work with more bugs and no performance gain.

---

## Recommended Architecture

### 1. Precompute Key Cache (HBM-Resident)

Instead of projecting `W_K · v_i` on every step, maintain a cached tensor in HBM:

```
key_cache[s, i, d_k] = W_K^s · v_i     # shape: [S, N, d_k]
```

- **Memory:** 65536 × 4 × 64 × 2 bytes ≈ **32 MB** (bf16)
- **Update:** Recomputed every training step when `v_i` changes (via `jax.lax.index_add` scatter updates from gradients)
- **Benefit:** Retrieval becomes a pure matmul `queries @ key_cache.T` instead of full pool projection

### 2. Retrieval Stage (XLA, HBM-Native)

```python
# Forward
q_s = W_Q_s @ z                           # [S, d_k]
s_i = cosine_similarity(q_s, key_cache)   # [N]
g_i = sigmoid(lambda * (s_i - tau))       # sigmoid gate
alpha = g_i * exp(s_i / T)
alpha = alpha / sum(alpha)                # normalize
top_k_indices, top_k_alpha = top_k(alpha, k=k_max)
```

This is a sequence of standard JAX ops. XLA compiles the matmul into tiled MXU operations that stream from HBM efficiently.

### 3. Assembly Stage (Pallas Kernel)

After retrieval, we have `k_max=16` indices and `alpha` weights. The Pallas kernel:

**Forward (Pallas):**
1. Gather 16 vectors from HBM pool into VMEM
2. Reshape each into `(U_i, V_i, b_i)` factors
3. Accumulate: `W = W_base + Σ α_i · (U_i @ V_i)` in VMEM
4. (Optional) Fuse with residual + LayerNorm: `h_mid = LayerNorm(h_A + γ · h_A @ W^T + b)`
5. Return `h_mid` to HBM

**Why Pallas here?**
- 16 vectors = 512 KB fits in VMEM
- 16 low-rank matmuls + accumulation stays on-chip
- No intermediate HBM writes for `W` or per-vector products
- Optional: fuse with LayerNorm to avoid another HBM round-trip

### 4. Gradient Flow (`jax.custom_vjp`)

Pallas autodiff on TPU is experimental and known to be inefficient (transposed access patterns are suboptimal). Write an explicit backward pass:

```python
# Forward: Pallas kernel returns h_mid
# Backward:
# 1. ∂L/∂W = ∂L/∂h_mid · h_A^T          (from residual path)
# 2. ∂L/∂α_i = <U_i @ V_i, ∂L/∂W>_F    (Frobenius inner product)
# 3. ∂L/∂U_i = α_i · ∂L/∂W @ V_i^T
# 4. ∂L/∂V_i = α_i · U_i^T @ ∂L/∂W
# 5. ∂L/∂v_i = [vec(∂L/∂U_i) ; vec(∂L/∂V_i) ; ∂L/∂b_i]  (reshape back)
# 6. Scatter-add ∂L/∂v_i to pool gradient buffer via jax.lax.index_add
```

All gradient accumulation for the 16 vectors happens in VMEM. Only the final scatter-add touches HBM.

### 5. Multi-Step Training Loop (On-Device)

Per the TPU Multi-Step Training guidelines in AGENTS.md:

```python
@nnx.jit
def train_multi_step(state, big_buffer):
    def step_fn(carry, batch_slice):
        state, step = carry
        loss, grads = forward_and_grad(state, batch_slice)  # includes Pallas kernel
        state = update(state, grads)
        jax.debug.print("Step {step} | loss={loss}", step=step, loss=loss)
        return (state, step + 1), loss

    (state, _), losses = jax.lax.scan(step_fn, (state, 0), big_buffer)
    return state, losses
```

- Pool + key_cache + model weights stay in HBM on TPU across the full scan window
- No CPU round-trip per step
- Gradient sync via `jax.lax.pmean` inside the scanned step
- `jax.debug.print` for lightweight scalar logging

---

## Memory Budget (TPU v4 per core, bf16)

| Component | Size |
|-----------|------|
| Pool (HBM) | 2 GB |
| Key cache (HBM) | 32 MB |
| Gathered k_max vectors (VMEM) | 512 KB |
| W_base + W_assembled (VMEM) | 128 KB |
| h_A batch (depends on B) | B × 256 × 2 bytes |
| LayerNorm stats (VMEM) | ~few KB |
| **Total VMEM for Pallas kernel** | **< 1 MB + batch overhead** |

TPU v4 has ~32 MB VMEM per core (megacore splits to ~16 MB). The assembly kernel is well within budget even with batch size 32–64.

---

## Dynamic Sparsity vs. Static Shapes

The sigmoid gate produces a dynamic number of active vectors (not always exactly `k_max`). Pallas requires static block shapes.

**Solution:** Always allocate for `k_max` vectors in the Pallas kernel. Zero-pad `alpha` weights for slots not selected (the sigmoid gate naturally drives unused `α → 0`). This avoids dynamic shape issues entirely.

---

## Implementation Order

1. **Validate with small config** (N=512, D=2048, k_max=8)
   - Full pool fits in VMEM (2 MB)
   - End-to-end pipeline testable without HBM streaming complexity
   - Verify Pallas kernel compiles and gradients flow correctly

2. **Implement hybrid architecture for full config**
   - XLA retrieval with key cache
   - Pallas assembly kernel with `jax.custom_vjp`
   - Multi-step `jax.lax.scan` training loop

3. **Profile and optimize**
   - Confirm HBM bandwidth is the bottleneck (expected)
   - If needed: shard pool across TPU devices, use product quantization for keys, or explore mixed-precision (fp8/bf16) key cache

---

## Edge Cases & Future Work

| Scenario | Mitigation |
|----------|------------|
| Pool grows beyond 65K | Shard similarity computation across devices with `jax.shard_map` or FSDP; consider product quantization for keys |
| Need sub-millisecond inference latency | Use approximate nearest neighbor (ScaNN, FAISS) for retrieval, feed exact top-k to Pallas assembly kernel |
| VMEM pressure with large batch | Reduce `k_max` or fuse fewer ops per kernel; profile with `jax.profiler` |
| Pallas `custom_vjp` bugs on TPU | Fallback to pure JAX assembly if Pallas autodiff is unstable; the performance loss is acceptable for small `k_max` |

---

## Summary

- **Do NOT try to fit 2 GB pool into VMEM.** Impossible.
- **Do NOT write a monolithic Pallas kernel for full retrieval + assembly.** XLA already handles the large matmul better.
- **DO write a focused Pallas kernel for assembly of k_max=16 vectors.** 512 KB fits in VMEM; accumulate low-rank products without HBM round-trips.
- **DO precompute key cache** to reduce retrieval from 2 GB pool read to 32 MB cache read.
- **DO use `jax.custom_vjp`** for explicit gradient flow through the Pallas boundary.
- **DO wrap in `jax.lax.scan`** for multi-step on-device training per AGENTS.md guidelines.

The result: Training avoids per-step HBM round-trips for the assembly stage (the critical path), while letting XLA optimally handle the large retrieval matmul.
