"""PhaseScheduler — stateless functions for three-phase training schedule."""

import math

import jax.numpy as jnp

from .config import TrainConfig


class PhaseScheduler:
    """
    Controls the three-phase training schedule.  Entirely stateless —
    all methods are pure functions of the step counter.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

    def get_phase(self, step: int) -> str:
        if step < self.cfg.warmup_steps:
            return "warmup"
        elif step < self.cfg.gate_on_steps:
            return "gate_on"
        else:
            return "sharpen"

    def is_warmup(self, step: int) -> bool:
        return step < self.cfg.warmup_steps

    def get_lambda(self, step: int) -> float:
        """Sharpness λ — annealed linearly within each phase."""
        cfg = self.cfg
        if step < cfg.warmup_steps:
            return 1.0  # unused during warmup
        elif step < cfg.gate_on_steps:
            t = (step - cfg.warmup_steps) / max(cfg.gate_on_steps - cfg.warmup_steps, 1)
            return cfg.lambda_gate_start + t * (cfg.lambda_gate_end - cfg.lambda_gate_start)
        else:
            t = min((step - cfg.gate_on_steps) / max(cfg.total_steps - cfg.gate_on_steps, 1), 1.0)
            return cfg.lambda_gate_end + t * (cfg.lambda_sharpen_end - cfg.lambda_gate_end)

    def get_lr_scale(self, step: int) -> float:
        """Global LR scale: linear warmup then cosine decay."""
        cfg = self.cfg
        if step < cfg.lr_warmup_steps:
            return step / max(cfg.lr_warmup_steps, 1)
        if step < cfg.gate_on_steps:
            return 1.0
        # cosine decay from gate_on_steps to total_steps, floored at lr_min_scale
        t = (step - cfg.gate_on_steps) / max(cfg.total_steps - cfg.gate_on_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
        return cfg.lr_min_scale + (1.0 - cfg.lr_min_scale) * cosine

    def aux_enabled(self, step: int) -> bool:
        return step >= self.cfg.warmup_steps

    def make_lambda_array(self, total_steps: int | None = None) -> jnp.ndarray:
        """Pre-compute λ schedule as a JAX array for use inside lax.scan."""
        n = total_steps or self.cfg.total_steps
        return jnp.array([self.get_lambda(s) for s in range(n)])

    def make_lr_scale_array(self, total_steps: int | None = None) -> jnp.ndarray:
        n = total_steps or self.cfg.total_steps
        return jnp.array([self.get_lr_scale(s) for s in range(n)])

    def make_optax_schedule(self, base_lr: float):
        """
        Returns a JAX-compatible schedule function (optax step count → lr).

        The pre-computed lr_scale array is captured in a closure so the
        schedule is a pure array lookup — no Python branching inside JIT.
        Clamps the index at total_steps-1 so the function is defined beyond
        the training horizon (e.g. during checkpoint-resumed runs).
        """
        lr_arr = self.make_lr_scale_array()   # [total_steps], float32
        def schedule(count):
            idx = jnp.minimum(count, lr_arr.shape[0] - 1)
            return base_lr * lr_arr[idx]
        return schedule
