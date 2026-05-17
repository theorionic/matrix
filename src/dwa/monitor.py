"""
Pool collapse detector and loss-adaptive LR controller.

PoolCollapseDetector:
    Tracks entropy, active fraction, unique retrieval fraction, and Gini
    coefficient over a rolling window.  Runs a state machine (OK → WARNING →
    CRITICAL → COLLAPSED) with hysteresis so a single bad window cannot flip
    the state, and trend-based detection catches gradual collapse before it
    becomes acute.

LossAdaptiveLRController:
    Emits a lr_scale multiplier [floor, 1.0] each window based on loss
    dynamics — not step count.  Mimics ReduceLROnPlateau: reduces on plateau,
    reduces aggressively on divergence, recovers slowly when improving.
    The multiplier is applied on top of the step-based cosine schedule so the
    two work together rather than in conflict.
"""

import math
from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gini(arr: np.ndarray) -> float:
    """Gini coefficient of arr.  0 = perfectly uniform, 1 = all mass on one item."""
    x = np.sort(np.abs(arr).ravel()).astype(np.float64)
    n = len(x)
    total = x.sum()
    if n == 0 or total < 1e-12:
        return 0.0
    i = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (i * x).sum() / (n * total)) - (n + 1.0) / n)


def _linear_slope(seq) -> float:
    """Slope of a linear fit over seq (oldest → newest).  Returns 0 if len < 3."""
    h = list(seq)
    if len(h) < 3:
        return 0.0
    x = np.arange(len(h), dtype=np.float64)
    return float(np.polyfit(x, h, 1)[0])


# ---------------------------------------------------------------------------
# Pool collapse detector
# ---------------------------------------------------------------------------

class PoolCollapseDetector:
    """
    Multi-signal pool collapse detector with state machine and hysteresis.

    Signals (all normalised to [0, 1]):
        entropy      — Shannon entropy / log(N)
        active_frac  — fraction of vectors with EMA > threshold * mean
        unique_frac  — unique indices retrieved / max possible
        gini         — Gini coefficient of pool EMA distribution
        top10_conc   — fraction of total usage in top-10% of vectors

    State machine transitions (requires hysteresis to prevent flapping):
        OK       — no action needed
        WARNING  — entropy/active declining; increase revival frequency
        CRITICAL — acute collapse; trigger immediate revival + alert
        COLLAPSED— non-recoverable; recommend stopping or resetting pool
    """

    STATES = ("OK", "WARNING", "CRITICAL", "COLLAPSED")

    def __init__(self, N: int, k_max: int, window: int = 8):
        self.N      = N
        self.k_max  = k_max
        self.window = window

        self._entropy_hist = deque(maxlen=window)
        self._active_hist  = deque(maxlen=window)
        self._unique_hist  = deque(maxlen=window)
        self._gini_hist    = deque(maxlen=window)

        self.state        = "OK"
        self._consec_warn = 0   # windows in warning zone
        self._consec_ok   = 0   # windows in healthy zone (for recovery)

    def update(
        self,
        pool_ema: np.ndarray,      # [N] float32
        last_indices: np.ndarray,  # [B, k_max] int32
        active_threshold: float = 0.01,
    ) -> dict:
        N = self.N
        ema = np.array(pool_ema, dtype=np.float64)

        # ── Compute signals ───────────────────────────────────────────────
        norm = ema / (ema.sum() + 1e-12)
        entropy = float(-np.sum(norm * np.log(norm + 1e-12))) / math.log(N)

        mean_ema     = float(ema.mean())
        active_frac  = float((ema > active_threshold * mean_ema).sum()) / N

        max_possible = min(int(last_indices.size), N)
        unique_frac  = float(np.unique(last_indices).shape[0]) / max(max_possible, 1)

        gini = _gini(norm)

        top_n      = max(1, N // 10)
        top10_conc = float(np.sort(norm)[-top_n:].sum())

        self._entropy_hist.append(entropy)
        self._active_hist.append(active_frac)
        self._unique_hist.append(unique_frac)
        self._gini_hist.append(gini)

        entropy_slope = _linear_slope(self._entropy_hist)
        active_slope  = _linear_slope(self._active_hist)

        # ── State machine ─────────────────────────────────────────────────
        prev_state = self.state

        # Hard thresholds — immediate transitions
        if unique_frac < 2.0 / max(max_possible, 1) or entropy < 0.05:
            new_state = "COLLAPSED"
        elif (entropy < 0.20 and entropy_slope < -0.008) or active_frac < 0.04:
            new_state = "CRITICAL"
        elif (
            (entropy < 0.40 and entropy_slope < -0.004)
            or (entropy_slope < -0.012 and len(self._entropy_hist) >= 5)
            or active_frac < 0.12
            or gini > 0.88
        ):
            new_state = "WARNING"
        else:
            new_state = "HEALTHY"

        # Hysteresis: need 2 consecutive windows before entering WARNING,
        # and 3 consecutive healthy windows before leaving WARNING/CRITICAL.
        if new_state in ("COLLAPSED", "CRITICAL"):
            self.state = new_state
            self._consec_warn = self.window  # saturate counter
            self._consec_ok   = 0
        elif new_state == "WARNING":
            self._consec_warn += 1
            self._consec_ok    = 0
            if self._consec_warn >= 2 or self.state in ("WARNING", "CRITICAL"):
                self.state = "WARNING"
        else:  # HEALTHY
            self._consec_ok  += 1
            self._consec_warn = max(0, self._consec_warn - 1)
            if self._consec_ok >= 3 and self.state not in ("COLLAPSED", "CRITICAL"):
                self.state    = "OK"
                self._consec_ok = 0

        # ── Actions ───────────────────────────────────────────────────────
        actions = []
        if self.state == "CRITICAL":
            actions.append("revive_now")
        elif self.state == "WARNING":
            actions.append("increase_revival_freq")
        if self.state == "COLLAPSED":
            actions.append("stop_or_reset")

        return {
            "state":          self.state,
            "prev_state":     prev_state,
            "changed":        self.state != prev_state,
            "entropy":        entropy,
            "entropy_slope":  entropy_slope,
            "active_frac":    active_frac,
            "active_slope":   active_slope,
            "unique_frac":    unique_frac,
            "gini":           gini,
            "top10_conc":     top10_conc,
            "actions":        actions,
        }

    def format_line(self, info: dict) -> str:
        tag = {
            "OK":        "",
            "WARNING":   "  ⚠ WARNING",
            "CRITICAL":  "  ❗ CRITICAL",
            "COLLAPSED": "  ☠ COLLAPSED",
        }[info["state"]]
        transition = (
            f"  (was {info['prev_state']}→{info['state']})"
            if info["changed"] and info["prev_state"] != info["state"]
            else ""
        )
        return (
            f"[Pool] entropy={info['entropy']:.3f}({info['entropy_slope']:+.4f}/w) "
            f"active={info['active_frac']:.1%}({info['active_slope']:+.4f}/w) "
            f"unique={info['unique_frac']:.1%} "
            f"gini={info['gini']:.3f} "
            f"top10={info['top10_conc']:.1%}"
            + tag + transition
        )


# ---------------------------------------------------------------------------
# Loss-adaptive LR controller
# ---------------------------------------------------------------------------

class LossAdaptiveLRController:
    """
    ReduceLROnPlateau-style LR controller driven by loss dynamics.

    Emits lr_scale ∈ [floor, 1.0] each window.  This multiplier is applied
    on top of the step-based cosine schedule so both mechanisms cooperate:
    the cosine handles the long-range shape; this handles short-range dynamics.

    Detection logic (in priority order):
        1. Divergence  — loss increasing for diverge_window consecutive windows
                         → immediate aggressive reduction (factor²)
        2. Plateau     — fast EMA not improving vs slow EMA for patience windows
                         → reduce by factor, enter cooldown
        3. Improvement — fast EMA clearly below slow EMA
                         → slowly recover lr_scale toward 1.0

    The fast/slow EMA gap is more stable than raw plateau detection because
    it filters per-window noise without requiring a fixed reference point.
    """

    def __init__(
        self,
        patience:       int   = 6,      # windows before reducing on plateau
        factor:         float = 0.70,   # LR reduction multiplier
        floor:          float = 0.05,   # minimum lr_scale
        cooldown:       int   = 4,      # windows between reductions
        recover_rate:   float = 1.03,   # per-window recovery when improving
        fast_alpha:     float = 0.15,   # fast loss EMA decay
        slow_alpha:     float = 0.03,   # slow loss EMA decay
        diverge_window: int   = 4,      # consecutive increases → diverge
        min_improve_pct: float = 0.002, # minimum relative improvement to count
    ):
        self.patience        = patience
        self.factor          = factor
        self.floor           = floor
        self.cooldown        = cooldown
        self.recover_rate    = recover_rate
        self.fast_alpha      = fast_alpha
        self.slow_alpha      = slow_alpha
        self.diverge_window  = diverge_window
        self.min_improve_pct = min_improve_pct

        self.lr_scale     = 1.0
        self._loss_fast   = None   # fast EMA
        self._loss_slow   = None   # slow EMA
        self._no_improve  = 0      # windows without improvement
        self._consec_inc  = 0      # consecutive loss increases
        self._cooldown    = 0      # remaining cooldown windows
        self._last_loss   = None
        self._gnorm_ema   = None
        self._n_reductions = 0     # total LR reductions so far

    def update(self, loss: float, grad_norm: float) -> dict:
        # ── Update EMAs ───────────────────────────────────────────────────
        if self._loss_fast is None:
            self._loss_fast = loss
            self._loss_slow = loss
        else:
            self._loss_fast = (1.0 - self.fast_alpha) * self._loss_fast + self.fast_alpha * loss
            self._loss_slow = (1.0 - self.slow_alpha) * self._loss_slow + self.slow_alpha * loss

        self._gnorm_ema = (
            grad_norm if self._gnorm_ema is None
            else 0.9 * self._gnorm_ema + 0.1 * grad_norm
        )

        # ── Improvement signal ────────────────────────────────────────────
        # Positive when fast (recent) loss is clearly below slow (lagged) loss
        improvement_rate = (
            (self._loss_slow - self._loss_fast) / (abs(self._loss_slow) + 1e-8)
        )
        is_improving = improvement_rate > self.min_improve_pct

        # Consecutive increase counter (raw loss, not EMA — catches divergence fast)
        if self._last_loss is not None and loss > self._last_loss * 1.005:
            self._consec_inc += 1
        else:
            self._consec_inc = 0
        self._last_loss = loss

        if self._cooldown > 0:
            self._cooldown -= 1

        # ── LR update logic ───────────────────────────────────────────────
        event = None

        if self._consec_inc >= self.diverge_window:
            # Divergence: double-reduce, extended cooldown
            new_scale = max(self.floor, self.lr_scale * self.factor * self.factor)
            if new_scale < self.lr_scale - 1e-6:
                event = (f"diverge({self._consec_inc}w)→"
                         f"×{self.factor**2:.2f} scale={new_scale:.4f}")
                self.lr_scale     = new_scale
                self._cooldown    = self.cooldown * 2
                self._no_improve  = 0
                self._consec_inc  = 0
                self._n_reductions += 1

        elif not is_improving and self._cooldown == 0:
            self._no_improve += 1
            if self._no_improve >= self.patience:
                new_scale = max(self.floor, self.lr_scale * self.factor)
                if new_scale < self.lr_scale - 1e-6:
                    event = (f"plateau({self._no_improve}w)→"
                             f"×{self.factor:.2f} scale={new_scale:.4f}")
                    self.lr_scale     = new_scale
                    self._cooldown    = self.cooldown
                    self._no_improve  = 0
                    self._n_reductions += 1

        elif is_improving:
            self._no_improve = 0
            # Recover slowly, cap at 1.0
            self.lr_scale = min(1.0, self.lr_scale * self.recover_rate)

        return {
            "lr_scale":         self.lr_scale,
            "loss_fast":        self._loss_fast,
            "loss_slow":        self._loss_slow,
            "improvement_rate": improvement_rate,
            "no_improve":       self._no_improve,
            "cooldown":         self._cooldown,
            "consec_inc":       self._consec_inc,
            "gnorm_ema":        self._gnorm_ema,
            "event":            event,
            "n_reductions":     self._n_reductions,
        }

    def format_line(self, info: dict) -> str:
        event_str = f"  [{info['event']}]" if info["event"] else ""
        cd_str    = f"  cd={info['cooldown']}" if info["cooldown"] > 0 else ""
        return (
            f"[LR]   scale={info['lr_scale']:.4f}  "
            f"impr={info['improvement_rate']:+.4f}  "
            f"plateau={info['no_improve']}/{self.patience}  "
            f"loss_fast={info['loss_fast']:.4f}  "
            f"loss_slow={info['loss_slow']:.4f}"
            + cd_str + event_str
        )
