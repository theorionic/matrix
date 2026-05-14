# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.
- **Verify first, then implement.** Run a minimal `python -c "<code>"` snippet (or small throwaway script) to confirm the API, shape, or behavior works exactly as expected before writing production code. This prevents writing wrong code against mistaken assumptions.
- **Prefer better APIs.** Before committing to an implementation path, check whether a more efficient API exists for the task (e.g., `jax.lax.scan` vs Python `for`, `nnx.jit` vs `jax.jit`, vectorized ops vs loops). If a better option exists, use it — but verify it first with a quick `python -c` test.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

### Commands

```bash
# Run
python main.py

# Test
pytest

# Single test
pytest path/to/test.py::test_name

# Lint
ruff check

# Lint + fix
ruff check --fix

# Dependencies (uv)
uv add <package>
uv sync
```

## Architecture

This project implements **Dynamic Weight Assembly (DWA)** — a novel ML architecture where a model's middle layer weight matrix is dynamically assembled per input from a pool of learned vectors. See `docs/ARCHITECTURE.md` for full design spec.

### Core Concept

Model splits into Part A and Part B. Between them:
1. Part A hidden state `z` → multi-aspect retrieval queries
2. Queries retrieve top-k vectors from pool (`N=65536` vectors, each `D=16384`)
3. Each vector `v_i` is reshaped into low-rank factors `(U_i, V_i, b_i)`
4. Assembled: `W = W_base + Σ α_i · (U_i @ V_i)`, applied via residual with scalar `γ`
5. Part B continues from assembled hidden state

### Key Parameters (full scale)
| Param | Value |
|-------|-------|
| D (vector dim) | 16384 |
| d_A = d_B (hidden) | 256 |
| r (rank per vector) | 24 |
| N (pool size) | 65536 |
| k_max (retrieved) | 16 |
| S (retrieval aspects) | 4 |

**Small validation config**: D=2048, d_A=d_B=64, r=4, N=512, k_max=8, S=2 — use this for fast iteration.

### Retrieval Mechanism
- Multi-aspect cosine similarity (S=4 heads) over pool vectors
- Sigmoid gate `g_i = σ(λ·(s_i − τ))` replaces hard top-k — every vector gets non-zero gradient
- Threshold `τ` is learnable; sharpness `λ` anneals 1→10 across training

### Training Phases
1. **Warmup** (0–1K steps): fixed top-16 softmax, LR warmup
2. **Gate On** (1K–10K): sigmoid gate active, aux losses enabled
3. **Sharpen** (10K+): λ→10, cosine LR decay

### Auxiliary Losses
- `L_util`: prevents dead pool vectors
- `L_div`: prevents key collapse (cosine penalty between retrieved keys)
- `L_norm`: penalizes `‖W − W_base‖²_F` explosion
- `L_sparse`: weight entropy regularization

### Per-Component LRs
`pool=3e-5`, `parts=1e-4`, `retrieval_proj=1e-4`, `threshold/gamma=1e-3`

---

## JAX / Flax NNX Best Practices

This project implements DWA using **JAX** and **Flax NNX** for TPU training. All implementation phases must follow these best practices to avoid performance pitfalls and compilation issues.

### 1. JIT Compilation

- Wrap complete training steps (forward + loss + backward + update) inside `nnx.jit` or `jax.jit`.
- Never call `jax.jit` repeatedly inside a loop; apply it once at the top level to a pure function.

### 2. No Python Control Flow in JITed Functions

- **Do NOT use** Python `for` or `while` loops inside a JIT-compiled function.
- Python loops cause long JIT compilation times and graph unrolling.
- **Use** `jax.lax.scan`, `jax.lax.fori_loop`, or `jax.lax.map` instead.

  ```python
  # BAD — causes unrolled graph and long compile
  def bad_fn(x):
      for i in range(N):
          x = x + i
      return x

  # GOOD — constant compile time regardless of N
  def good_fn(x):
      def body(carry, i):
          return carry + i, None
      x, _ = jax.lax.scan(body, x, jnp.arange(N))
      return x
  ```

### 3. NNX Modules and State

- Prefer `nnx.Module` for all trainable components.
- Use `nnx.jit` (not bare `jax.jit`) when working with `nnx.Module` instances, so state handling is managed correctly.
- Split `graphdef` and `state` only when necessary for custom transforms; otherwise rely on NNX's built-in `nnx.jit` / `nnx.grad`.

### 4. Training Loops

- The **entire training step** (forward, loss computation, backward, optimizer update) should be a single JITed function.
- The outer Python loop over steps/batches should be the only Python-level iteration.

### 5. Performance Tips

- Favor `jax.vmap` over manual batching inside JIT.
- Avoid dynamic shapes; keep array sizes static across JIT boundaries.
- Use `jax.lax.scan` for sequential passes (e.g., multi-layer loops, unrolled training phases).
- Keep constants and static metadata outside JIT when possible, or mark them as `static_argnums`/`static_argnames`.

---

## TPU Multi-Step Training (Device-Host Boundary)

To maximize TPU utilization and minimize device↔host transfer overhead, **never trigger a CPU round-trip per step**. Instead, push a large tokenized buffer (e.g., ~1 GB per core) and the model weights to the TPU once, then run many steps entirely on-device with gradient sync inside a single JIT boundary.

### Preferred Pattern

```python
# BAD — host loop calls device every step
for batch in host_loader:
    grads = jax.grad(loss_fn)(state, batch)  # Device↔Host every iteration
    state = update(state, grads)

# GOOD — one host call, many device steps
@nnx.jit
def train_multi_step(state, big_buffer):
    def step_fn(carry, batch_slice):
        state, step = carry
        grads = jax.grad(loss_fn)(state, batch_slice)
        state = update(state, grads)
        return (state, step + 1), None

    (state, _), _ = jax.lax.scan(step_fn, (state, 0), big_buffer)
    return state

state = train_multi_step(state, one_gb_tokenized_buffer)
```

### Rules of Thumb

1. **One large transfer per window** — concatenate / shard tokenized data on host so each TPU core receives a contiguous multi-step buffer. Call the JITed function once per window.
2. **All step logic inside `jax.lax.scan`** — forward, backward, optimizer update, gradient sync (e.g., `jax.lax.pmean` for `pmap` or implicit sync with `nnx.jit`), learning-rate schedules, and metrics accumulation must live inside the scanned body. No Python `for`/`while` at the host level inside the loop.
3. **Static shapes only** — the scan length (number of steps per window) should be a static constant or `static_argname`. Do not pass Python `range()` dynamically inside JIT.
4. **State stays on device** — model parameters, optimizer states, and any step counters must remain in device memory across the scan. Only the final updated state returns to host (and only if you actually need it there).
5. **Gradient sync inside JIT** — if using multiple TPU cores, call `jax.lax.pmean(grads, axis_name="batch")` inside the scanned step so all-reduce happens on-device without host orchestration.
6. **Metrics accumulation** — if you need per-step metrics (loss, learning rate, etc.), accumulate them inside the scan and return a summarized dictionary at the end of the window.
7. **Memory budget** — size your buffer so that the full window (batch_size × seq_len × steps_per_window) plus model weights and optimizer states fit in TPU HBM. Start small and profile with `jax.profiler`.

### Debugging & Logging Inside JIT

- **Use `jax.debug.print`** (not Python `print`) to emit training stats at runtime from within a JIT-compiled function.
- `jax.debug.print` has a `ordered=True` flag; set it when you need strictly ordered output, but be aware it may add synchronization overhead.
- Keep debug printing lightweight — log only scalar summaries (loss, LR, step count) to avoid flooding the output and degrading TPU throughput.

  ```python
  # INSIDE a scanned step or JITed training function
  def step_fn(carry, batch_slice):
      state, step = carry
      loss, grads = forward_and_grad(state, batch_slice)
      state = update(state, grads)
      jax.debug.print("Step {step} | loss={loss}", step=step, loss=loss)
      return (state, step + 1), loss
  ```

- For host-side logging or tensorboard summaries, accumulate metrics in the scan and return them at window-end; do **not** send per-step arrays to host from inside JIT.

### When to Break the Rule

If you need **online data augmentation**, **dynamic curriculum sampling**, or **host-side logging every step**, you must return to host. In that case, amortize the overhead by running the largest multi-step window that still fits your data-dependent logic, or move the logic into the device loop with JAX-compatible random ops (`jax.random`).
