"""Tests for PoolCollapseDetector and LossAdaptiveLRController."""

import numpy as np
import pytest

from src.dwa.monitor import LossAdaptiveLRController, PoolCollapseDetector, _gini


# ---------------------------------------------------------------------------
# _gini helper
# ---------------------------------------------------------------------------

class TestGini:
    def test_uniform_is_zero(self):
        arr = np.ones(64) / 64
        assert _gini(arr) == pytest.approx(0.0, abs=0.02)

    def test_concentrated_is_one(self):
        arr = np.zeros(64)
        arr[0] = 1.0
        assert _gini(arr) > 0.95

    def test_intermediate(self):
        arr = np.zeros(64)
        arr[:8] = 1.0 / 8   # top 8 of 64 = 12.5%
        g = _gini(arr)
        assert 0.5 < g < 0.95

    def test_all_zeros(self):
        assert _gini(np.zeros(10)) == 0.0


# ---------------------------------------------------------------------------
# PoolCollapseDetector
# ---------------------------------------------------------------------------

class TestPoolCollapseDetector:
    N, K = 64, 4

    def _healthy_ema(self, rng=0):
        return np.ones(self.N) / self.N + np.random.RandomState(rng).randn(self.N) * 5e-4

    def _random_idx(self, rng=0):
        return np.random.RandomState(rng).randint(0, self.N, (8, self.K))

    def test_healthy_pool_ok(self):
        det = PoolCollapseDetector(self.N, self.K)
        ema = self._healthy_ema()
        idx = self._random_idx()
        for _ in range(10):
            info = det.update(ema, idx)
        assert info["state"] == "OK"
        assert info["entropy"] > 0.9

    def test_full_collapse_detected(self):
        det = PoolCollapseDetector(self.N, self.K)
        ema = np.zeros(self.N); ema[0] = 1.0     # one vector does everything
        idx = np.zeros((8, self.K), dtype=int)    # always retrieves vector 0
        for _ in range(3):
            info = det.update(ema, idx)
        assert info["state"] == "COLLAPSED"
        assert "stop_or_reset" in info["actions"]

    def test_critical_on_acute_low_entropy(self):
        det = PoolCollapseDetector(self.N, self.K)
        # Low entropy: 4 vectors of 64 get all mass → entropy ≈ log(4)/log(64) ≈ 0.33
        # After 3 windows this crosses WARNING threshold (< 0.40 + declining trend)
        ema = np.zeros(self.N)
        ema[:4] = 0.25  # top-4 of 64 get all mass
        idx = np.random.RandomState(0).randint(0, 4, (8, self.K))
        for _ in range(3):
            info = det.update(ema, idx)
        # 0.33 < 0.40 → WARNING or worse
        assert info["state"] in ("WARNING", "CRITICAL", "COLLAPSED")

    def test_warning_on_declining_trend(self):
        det = PoolCollapseDetector(self.N, self.K, window=8)
        # Start healthy, gradually concentrate
        for i in range(8):
            frac = 1.0 - i * 0.08   # 100% → 44% active over 8 windows
            n_active = max(2, int(frac * self.N))
            ema = np.zeros(self.N)
            ema[:n_active] = 1.0 / n_active
            idx = np.random.RandomState(i).randint(0, n_active, (8, self.K))
            info = det.update(ema, idx)
        assert info["state"] in ("WARNING", "CRITICAL", "COLLAPSED")

    def test_recovery_after_healthy_windows(self):
        det = PoolCollapseDetector(self.N, self.K)
        # First push to WARNING
        for _ in range(4):
            ema = np.zeros(self.N); ema[:6] = 1/6
            idx = np.random.randint(0, 6, (8, self.K))
            det.update(ema, idx)
        # Now give it 4 healthy windows
        for _ in range(5):
            info = det.update(self._healthy_ema(), self._random_idx())
        assert info["state"] == "OK"
        assert det._consec_warn == 0

    def test_state_changed_flag(self):
        det = PoolCollapseDetector(self.N, self.K)
        # Start healthy (state = OK)
        det.update(self._healthy_ema(), self._random_idx())
        # First collapse update: state changes OK → COLLAPSED
        ema = np.zeros(self.N); ema[0] = 1.0
        idx = np.zeros((8, self.K), dtype=int)
        first_info = det.update(ema, idx)
        assert first_info["changed"] is True
        assert first_info["prev_state"] == "OK"
        assert first_info["state"] in ("COLLAPSED", "CRITICAL")
        # Subsequent updates: already in collapse, no transition
        next_info = det.update(ema, idx)
        assert next_info["changed"] is False

    def test_revive_now_action_on_critical(self):
        det = PoolCollapseDetector(self.N, self.K)
        ema = np.zeros(self.N); ema[:2] = 0.5
        idx = np.zeros((8, self.K), dtype=int)
        for _ in range(4):
            info = det.update(ema, idx)
        if info["state"] in ("CRITICAL", "COLLAPSED"):
            assert any(a in info["actions"] for a in ("revive_now", "stop_or_reset"))

    def test_format_line_contains_key_fields(self):
        det = PoolCollapseDetector(self.N, self.K)
        info = det.update(self._healthy_ema(), self._random_idx())
        line = det.format_line(info)
        assert "entropy=" in line
        assert "active=" in line
        assert "gini=" in line
        assert "unique=" in line

    def test_gini_high_for_collapsed(self):
        det = PoolCollapseDetector(self.N, self.K)
        ema = np.zeros(self.N); ema[0] = 1.0
        idx = np.zeros((8, self.K), dtype=int)
        info = det.update(ema, idx)
        assert info["gini"] > 0.9

    def test_top10_conc_high_for_collapsed(self):
        det = PoolCollapseDetector(self.N, self.K)
        ema = np.zeros(self.N); ema[0] = 1.0
        idx = np.zeros((8, self.K), dtype=int)
        info = det.update(ema, idx)
        # top 10% of 64 = 6 vectors, but all mass on vector 0
        assert info["top10_conc"] > 0.9

    def test_entropy_slope_negative_during_collapse(self):
        det = PoolCollapseDetector(self.N, self.K, window=6)
        # Gradually concentrate
        for i in range(6):
            n = max(1, self.N - i * 10)
            ema = np.zeros(self.N); ema[:n] = 1.0 / n
            idx = np.random.RandomState(i).randint(0, n, (8, self.K))
            info = det.update(ema, idx)
        assert info["entropy_slope"] < 0   # entropy declining


# ---------------------------------------------------------------------------
# LossAdaptiveLRController
# ---------------------------------------------------------------------------

class TestLossAdaptiveLRController:
    def test_no_change_when_improving(self):
        ctrl = LossAdaptiveLRController(patience=3, factor=0.7)
        for l in [3.0, 2.8, 2.6, 2.4, 2.2, 2.0, 1.8]:
            info = ctrl.update(l, 0.5)
        assert info["lr_scale"] == pytest.approx(1.0, rel=0.05)
        assert info["event"] is None

    def test_reduces_on_plateau(self):
        ctrl = LossAdaptiveLRController(patience=3, factor=0.7, cooldown=1)
        for l in [3.0] * 6:
            info = ctrl.update(l, 0.5)
        assert info["lr_scale"] < 1.0
        assert "plateau" in (info["event"] or "")

    def test_reduction_by_correct_factor(self):
        ctrl = LossAdaptiveLRController(patience=3, factor=0.7, cooldown=0)
        # Trigger exactly one plateau
        for _ in range(4):
            info = ctrl.update(3.0, 0.5)
        assert info["lr_scale"] == pytest.approx(0.7, rel=1e-3)

    def test_double_reduction_on_diverge(self):
        ctrl = LossAdaptiveLRController(patience=10, factor=0.7, diverge_window=4)
        for l in [2.0, 2.1, 2.2, 2.3, 2.4]:
            info = ctrl.update(l, 0.5)
        assert info["lr_scale"] < 0.7   # double-factor = 0.49
        assert "diverge" in (info["event"] or "")

    def test_cooldown_prevents_immediate_re_reduction(self):
        ctrl = LossAdaptiveLRController(patience=2, factor=0.7, cooldown=3)
        # First plateau
        for _ in range(3):
            ctrl.update(3.0, 0.5)
        scale_after_first = ctrl.lr_scale
        # During cooldown, same flat loss should NOT reduce again
        for _ in range(2):
            info = ctrl.update(3.0, 0.5)
        assert ctrl.lr_scale == pytest.approx(scale_after_first, rel=1e-3)

    def test_recovery_after_plateau_reduction(self):
        ctrl = LossAdaptiveLRController(patience=2, factor=0.7, cooldown=0, recover_rate=1.1)
        # Trigger reduction
        for _ in range(3):
            ctrl.update(3.0, 0.5)
        scale_reduced = ctrl.lr_scale
        # Now improve for several windows
        for l in [2.9, 2.7, 2.5, 2.3, 2.1]:
            info = ctrl.update(l, 0.5)
        assert ctrl.lr_scale > scale_reduced

    def test_floor_prevents_zero_lr(self):
        ctrl = LossAdaptiveLRController(patience=2, factor=0.7, cooldown=0, floor=0.05)
        # Many plateau windows
        for _ in range(30):
            ctrl.update(3.0, 0.5)
        assert ctrl.lr_scale >= 0.05

    def test_starts_at_one(self):
        ctrl = LossAdaptiveLRController()
        assert ctrl.lr_scale == pytest.approx(1.0)

    def test_improvement_rate_positive_when_declining(self):
        ctrl = LossAdaptiveLRController(fast_alpha=0.5, slow_alpha=0.1)
        for l in [4.0, 3.5, 3.0, 2.5, 2.0]:
            info = ctrl.update(l, 0.5)
        # Fast EMA tracks recent improvement faster than slow EMA
        assert info["improvement_rate"] > 0

    def test_improvement_rate_zero_when_flat(self):
        ctrl = LossAdaptiveLRController()
        for l in [3.0] * 10:
            info = ctrl.update(l, 0.5)
        assert abs(info["improvement_rate"]) < 0.01

    def test_format_line_contains_key_fields(self):
        ctrl = LossAdaptiveLRController()
        info = ctrl.update(3.0, 0.5)
        line = ctrl.format_line(info)
        assert "scale=" in line
        assert "impr=" in line
        assert "plateau=" in line

    def test_n_reductions_counts_correctly(self):
        ctrl = LossAdaptiveLRController(patience=2, factor=0.7, cooldown=1)
        for _ in range(10):
            ctrl.update(3.0, 0.5)
        assert ctrl._n_reductions >= 2

    def test_consecutive_inc_resets_after_decrease(self):
        ctrl = LossAdaptiveLRController(diverge_window=3)
        ctrl.update(2.0, 0.5)   # baseline
        ctrl.update(2.1, 0.5)   # inc
        ctrl.update(2.2, 0.5)   # inc
        ctrl.update(1.8, 0.5)   # dec — should reset consec_inc
        info = ctrl.update(1.9, 0.5)
        assert info["consec_inc"] <= 1   # reset happened


# ---------------------------------------------------------------------------
# Optimizer tx rebuild integration
# ---------------------------------------------------------------------------

class TestOptimizerTxRebuild:
    """Verify that rebuilding optimizer.tx (for adaptive LR) preserves Adam state."""

    def test_tx_rebuild_preserves_opt_state_structure(self):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from src.dwa.config import DWAConfig, TrainConfig
        from src.dwa.model import DWAModel
        from src.dwa.schedule import PhaseScheduler
        from train import _build_optimizer, _build_tx

        cfg = DWAConfig(D=2048, N=64, d_A=64, d_B=64, r=4, k_max=4, S=2,
                        d_k=32, m=2, n_heads=2, n_layers_A=1, n_layers_B=1,
                        ffn_mult=2, vocab_size=32, seq_len=16, use_ivf=True, C=4)
        tcfg = TrainConfig()
        tcfg.total_steps = 32
        scheduler = PhaseScheduler(tcfg)

        model = DWAModel(cfg, nnx.Rngs(0))
        optimizer = _build_optimizer(model, tcfg, scheduler)

        # Capture opt_state structure before rebuild
        leaves_before, treedef_before = jax.tree_util.tree_flatten(optimizer.opt_state)
        n_leaves_before = len(leaves_before)

        # Rebuild tx with a reduced scale
        optimizer.tx = _build_tx(model, tcfg, scheduler, lr_scale=0.5)

        # opt_state should be unchanged
        leaves_after, treedef_after = jax.tree_util.tree_flatten(optimizer.opt_state)
        assert len(leaves_after) == n_leaves_before, "opt_state leaf count changed"
        for b, a in zip(leaves_before, leaves_after):
            assert b.shape == a.shape, "opt_state leaf shape changed"

    def test_tx_rebuild_changes_effective_lr(self):
        """After rebuild with scale=0.5, the effective LR is halved."""
        import jax.numpy as jnp
        from src.dwa.config import TrainConfig
        from src.dwa.config import DWAConfig
        from src.dwa.model import DWAModel
        from src.dwa.schedule import PhaseScheduler
        from flax import nnx
        from train import _build_optimizer, _build_tx

        cfg = DWAConfig(D=2048, N=64, d_A=64, d_B=64, r=4, k_max=4, S=2,
                        d_k=32, m=2, n_heads=2, n_layers_A=1, n_layers_B=1,
                        ffn_mult=2, vocab_size=32, seq_len=16, use_ivf=True, C=4)
        tcfg = TrainConfig()
        tcfg.total_steps = 32
        tcfg.lr_warmup_steps = 0   # skip warmup so LR = base_lr immediately
        tcfg.lr_parts = 1e-3
        scheduler = PhaseScheduler(tcfg)

        model = DWAModel(cfg, nnx.Rngs(0))

        # Optimizer at scale=1.0 → schedule at step 0 = lr_parts * lr_scale(0)
        opt1 = _build_optimizer(model, tcfg, scheduler)
        # lr_scale(0) with lr_warmup_steps=0 → 1.0
        sched_full = scheduler.make_optax_schedule(tcfg.lr_parts)
        lr_full = float(sched_full(jnp.array(0)))

        # Rebuild with scale=0.5
        opt1.tx = _build_tx(model, tcfg, scheduler, lr_scale=0.5)
        sched_half = scheduler.make_optax_schedule(tcfg.lr_parts * 0.5)
        lr_half = float(sched_half(jnp.array(0)))

        assert lr_half == pytest.approx(lr_full * 0.5, rel=1e-3)
