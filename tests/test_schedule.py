"""Tests for PhaseScheduler — phase boundaries, lambda values, LR schedule."""

import math

import jax
import jax.numpy as jnp
import pytest

from src.dwa.config import TrainConfig
from src.dwa.schedule import PhaseScheduler


@pytest.fixture(scope="module")
def tcfg():
    t = TrainConfig()
    t.warmup_steps = 100
    t.gate_on_steps = 500
    t.total_steps = 1000
    t.lr_warmup_steps = 50
    t.lr_min_scale = 0.1
    return t


@pytest.fixture(scope="module")
def sched(tcfg):
    return PhaseScheduler(tcfg)


class TestPhases:
    def test_warmup_phase(self, sched, tcfg):
        assert sched.get_phase(0) == "warmup"
        assert sched.get_phase(tcfg.warmup_steps - 1) == "warmup"
        assert sched.is_warmup(0)

    def test_gate_on_phase(self, sched, tcfg):
        assert sched.get_phase(tcfg.warmup_steps) == "gate_on"
        assert sched.get_phase(tcfg.gate_on_steps - 1) == "gate_on"
        assert not sched.is_warmup(tcfg.warmup_steps)

    def test_sharpen_phase(self, sched, tcfg):
        assert sched.get_phase(tcfg.gate_on_steps) == "sharpen"
        assert sched.get_phase(tcfg.total_steps) == "sharpen"

    def test_aux_enabled_after_warmup(self, sched, tcfg):
        assert not sched.aux_enabled(tcfg.warmup_steps - 1)
        assert sched.aux_enabled(tcfg.warmup_steps)


class TestLambda:
    def test_warmup_lambda_fixed(self, sched, tcfg):
        assert sched.get_lambda(0) == 1.0
        assert sched.get_lambda(tcfg.warmup_steps - 1) == 1.0

    def test_gate_on_lambda_range(self, sched, tcfg):
        lam_start = sched.get_lambda(tcfg.warmup_steps)
        lam_end = sched.get_lambda(tcfg.gate_on_steps - 1)
        assert lam_start == tcfg.lambda_gate_start
        assert lam_end < tcfg.lambda_gate_end

    def test_sharpen_lambda_increases(self, sched, tcfg):
        lam_begin = sched.get_lambda(tcfg.gate_on_steps)
        lam_final = sched.get_lambda(tcfg.total_steps)
        assert lam_begin <= lam_final
        assert lam_final == pytest.approx(tcfg.lambda_sharpen_end, abs=0.1)

    def test_lambda_array_length(self, sched, tcfg):
        arr = sched.make_lambda_array()
        assert arr.shape == (tcfg.total_steps,)

    def test_lambda_array_values_match_scalar(self, sched, tcfg):
        arr = sched.make_lambda_array()
        for step in [0, tcfg.warmup_steps, tcfg.gate_on_steps, tcfg.total_steps - 1]:
            assert float(arr[step]) == pytest.approx(sched.get_lambda(step), rel=1e-4)


class TestLRSchedule:
    def test_warmup_monotone_increasing(self, sched, tcfg):
        scales = [sched.get_lr_scale(s) for s in range(tcfg.lr_warmup_steps + 1)]
        for a, b in zip(scales, scales[1:]):
            assert b >= a, f"LR not monotone: {a} → {b}"

    def test_lr_one_during_gate_on(self, sched, tcfg):
        assert sched.get_lr_scale(tcfg.lr_warmup_steps) == pytest.approx(1.0, abs=1e-6)
        assert sched.get_lr_scale(tcfg.gate_on_steps - 1) == pytest.approx(1.0, abs=1e-6)

    def test_cosine_decay_in_sharpen(self, sched, tcfg):
        s0 = sched.get_lr_scale(tcfg.gate_on_steps)
        s_end = sched.get_lr_scale(tcfg.total_steps)
        assert s0 == pytest.approx(1.0, abs=1e-3)
        assert s_end == pytest.approx(tcfg.lr_min_scale, abs=1e-3)

    def test_lr_scale_array_length(self, sched, tcfg):
        arr = sched.make_lr_scale_array()
        assert arr.shape == (tcfg.total_steps,)

    def test_optax_schedule_in_jit(self, sched):
        schedule_fn = sched.make_optax_schedule(1e-4)
        # Should be callable inside jit
        result = jax.jit(lambda c: schedule_fn(c))(jnp.array(0))
        assert float(result) >= 0.0

    def test_optax_schedule_clamps_beyond_total(self, sched, tcfg):
        schedule_fn = sched.make_optax_schedule(1e-4)
        # Beyond total_steps should not raise — returns last value
        v1 = float(jax.jit(lambda c: schedule_fn(c))(jnp.array(tcfg.total_steps - 1)))
        v2 = float(jax.jit(lambda c: schedule_fn(c))(jnp.array(tcfg.total_steps + 1000)))
        assert v1 == pytest.approx(v2, rel=1e-4)
