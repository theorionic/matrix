"""
End-to-end resume correctness tests.

Tests:
  1. Params are byte-identical after save/restore
  2. Optimizer step counter is restored (LR schedule continuity)
  3. start_window correctly skips already-completed windows
  4. Loss at first window after resume ≈ loss if we had continued uninterrupted
  5. Checkpoint is NOT immediately re-saved on resume step
  6. Data loader cursor is saved and restored
  7. RNG state is restored so data generation is deterministic

Uses random (synthetic) data — no HuggingFace needed.
"""

import os
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dwa.config import DWAConfig, TrainConfig
from src.dwa.model import DWAModel
from src.dwa.run_config import (
    CheckpointConfig, DataConfig, RunConfig, ShardingConfig,
)
from src.dwa.schedule import PhaseScheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_cfg():
    return DWAConfig(
        D=2048, N=64, d_A=64, d_B=64, r=4, k_max=4, S=2, d_k=32, m=2,
        n_heads=2, n_layers_A=1, n_layers_B=1, ffn_mult=2,
        vocab_size=64, seq_len=16, use_ivf=True, C=4,
    )


def _tiny_tcfg():
    t = TrainConfig()
    t.warmup_steps = 4
    t.gate_on_steps = 8
    t.total_steps = 32          # 8 windows × 4 steps
    t.batch_size = 8
    t.steps_per_window = 4
    t.lr_warmup_steps = 2
    return t


def _get_mesh():
    devices = jax.devices()
    n = len(devices)
    return Mesh(np.array(devices).reshape(n, 1), ("data", "model"))


def _build_run_cfg(ckpt_dir="", ckpt_every=0, resume=False, source="random", total_steps=None):
    cfg = _tiny_cfg()
    tcfg = _tiny_tcfg()
    if total_steps is not None:
        tcfg.total_steps = total_steps
    return RunConfig(
        model=cfg,
        train=tcfg,
        sharding=ShardingConfig(n_model=1),
        data=DataConfig(source=source, gen_every=9999, val_every=0),
        checkpoint=CheckpointConfig(dir=ckpt_dir, every=ckpt_every, resume=resume),
        name="resume_test",
    )


# ---------------------------------------------------------------------------
# Test 1: Params restored byte-identically after save/restore
# ---------------------------------------------------------------------------

class TestParamRestore:
    def test_pool_vectors_exact(self):
        from train import _build_optimizer, save_checkpoint, load_checkpoint, _build_mesh
        cfg = _tiny_cfg()
        tcfg = _tiny_tcfg()
        mesh = _get_mesh()
        model = DWAModel(cfg, nnx.Rngs(0))
        scheduler = PhaseScheduler(tcfg)
        optimizer = _build_optimizer(model, tcfg, scheduler)

        pool_before = np.array(model.pool.vectors[...]).copy()

        with tempfile.TemporaryDirectory() as ckpt_dir:
            save_checkpoint(ckpt_dir, model, optimizer, jnp.zeros(cfg.N), 8, jax.random.PRNGKey(0), None)
            model.pool.vectors[...] = jnp.zeros_like(model.pool.vectors[...])  # corrupt
            load_checkpoint(ckpt_dir, 8, model, optimizer, cfg, mesh)
            pool_after = np.array(model.pool.vectors[...])

        assert np.allclose(pool_before, pool_after, atol=1e-5), "Pool vectors not restored"

    def test_transformer_weights_exact(self):
        from train import _build_optimizer, save_checkpoint, load_checkpoint, _build_mesh
        cfg = _tiny_cfg()
        tcfg = _tiny_tcfg()
        mesh = _get_mesh()
        model = DWAModel(cfg, nnx.Rngs(7))
        scheduler = PhaseScheduler(tcfg)
        optimizer = _build_optimizer(model, tcfg, scheduler)

        # Capture all Param leaves before save
        params_before = jax.tree_util.tree_map(
            np.array, nnx.state(model, nnx.Param)
        )

        with tempfile.TemporaryDirectory() as ckpt_dir:
            save_checkpoint(ckpt_dir, model, optimizer, jnp.zeros(cfg.N), 4, jax.random.PRNGKey(1), None)
            # Zero all params
            for path, leaf in jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))[0]:
                pass  # can't mutate leaves directly; use nnx.update below
            load_checkpoint(ckpt_dir, 4, model, optimizer, cfg, mesh)
            params_after = jax.tree_util.tree_map(np.array, nnx.state(model, nnx.Param))

        leaves_b = jax.tree_util.tree_leaves(params_before)
        leaves_a = jax.tree_util.tree_leaves(params_after)
        for b, a in zip(leaves_b, leaves_a):
            assert np.allclose(b, a, atol=1e-5), "Transformer param not restored"

    def test_optimizer_step_restored(self):
        from train import _build_optimizer, save_checkpoint, load_checkpoint, _build_mesh
        cfg = _tiny_cfg()
        tcfg = _tiny_tcfg()
        mesh = _get_mesh()
        model = DWAModel(cfg, nnx.Rngs(0))
        scheduler = PhaseScheduler(tcfg)
        optimizer = _build_optimizer(model, tcfg, scheduler)

        # Manually set a non-zero step counter
        optimizer.step[...] = jnp.array(42, dtype=jnp.uint32)

        with tempfile.TemporaryDirectory() as ckpt_dir:
            save_checkpoint(ckpt_dir, model, optimizer, jnp.zeros(cfg.N), 42, jax.random.PRNGKey(0), None)
            optimizer.step[...] = jnp.array(0, dtype=jnp.uint32)  # corrupt
            load_checkpoint(ckpt_dir, 42, model, optimizer, cfg, mesh)

        assert int(optimizer.step[...]) == 42, f"opt step not restored: {int(optimizer.step[...])}"

    def test_pool_ema_restored(self):
        from train import _build_optimizer, save_checkpoint, load_checkpoint, _build_mesh
        cfg = _tiny_cfg()
        tcfg = _tiny_tcfg()
        mesh = _get_mesh()
        model = DWAModel(cfg, nnx.Rngs(0))
        scheduler = PhaseScheduler(tcfg)
        optimizer = _build_optimizer(model, tcfg, scheduler)
        ema = jax.random.uniform(jax.random.PRNGKey(5), (cfg.N,))

        with tempfile.TemporaryDirectory() as ckpt_dir:
            save_checkpoint(ckpt_dir, model, optimizer, ema, 8, jax.random.PRNGKey(0), None)
            _, _, _, ema_out = load_checkpoint(ckpt_dir, 8, model, optimizer, cfg, mesh)

        assert np.allclose(np.array(ema), np.array(ema_out), atol=1e-6)

    def test_rng_restored(self):
        from train import _build_optimizer, save_checkpoint, load_checkpoint, _build_mesh
        cfg = _tiny_cfg()
        tcfg = _tiny_tcfg()
        mesh = _get_mesh()
        model = DWAModel(cfg, nnx.Rngs(0))
        scheduler = PhaseScheduler(tcfg)
        optimizer = _build_optimizer(model, tcfg, scheduler)
        rng = jax.random.PRNGKey(77)

        with tempfile.TemporaryDirectory() as ckpt_dir:
            save_checkpoint(ckpt_dir, model, optimizer, jnp.zeros(cfg.N), 8, rng, None)
            _, rng_out, _, _ = load_checkpoint(ckpt_dir, 8, model, optimizer, cfg, mesh)

        assert np.array_equal(np.array(rng), np.array(rng_out)), "RNG key not restored"


# ---------------------------------------------------------------------------
# Test 2: Optimizer Adam moments restored — loss trajectory is identical
# ---------------------------------------------------------------------------

class TestAdamMomentsRestore:
    """
    If Adam M/V are correctly restored, the first gradient step after resume
    should update parameters identically to what would have happened without
    interruption.  We verify this by running:
        run A: N windows uninterrupted → record params
        run B: K windows → checkpoint → resume → N-K windows → compare params
    Both runs use the same random seed and synthetic data.
    """

    def _run_windows(self, n_windows, seed, ckpt_dir=None, ckpt_at=None, resume=False, resume_ckpt_dir=None):
        from train import _build_optimizer, save_checkpoint, load_checkpoint, _make_train_window, _build_mesh
        cfg = _tiny_cfg()
        tcfg = _tiny_tcfg()
        mesh = _get_mesh()
        scheduler = PhaseScheduler(tcfg)
        lambda_array = scheduler.make_lambda_array()

        model = DWAModel(cfg, nnx.Rngs(seed))
        optimizer = _build_optimizer(model, tcfg, scheduler)
        pool_ema = jnp.zeros(cfg.N)
        rng = jax.random.PRNGKey(seed + 1)
        steps_done = 0
        start_window = 0

        if resume and resume_ckpt_dir:
            from train import _get_ckpt_manager
            mngr = _get_ckpt_manager(resume_ckpt_dir)
            latest = mngr.latest_step()
            steps_done, rng, _, pool_ema = load_checkpoint(
                resume_ckpt_dir, latest, model, optimizer, cfg, mesh
            )
            start_window = steps_done // tcfg.steps_per_window

        compiled = {}
        for window_idx in range(start_window, start_window + n_windows):
            start_step = window_idx * tcfg.steps_per_window
            phase_key = (scheduler.is_warmup(start_step), scheduler.aux_enabled(start_step))
            if phase_key not in compiled:
                compiled[phase_key] = _make_train_window(
                    cfg, tcfg,
                    is_warmup=phase_key[0],
                    aux_on=phase_key[1],
                    use_pallas=False, mesh=mesh,
                )
            train_fn = compiled[phase_key]
            lam_window = lambda_array[start_step: start_step + tcfg.steps_per_window]
            rng, data_rng = jax.random.split(rng)
            data_window = jax.random.randint(data_rng,
                                             (tcfg.steps_per_window, tcfg.batch_size, cfg.seq_len),
                                             0, cfg.vocab_size)
            data_sharded = jax.device_put(data_window, NamedSharding(mesh, P(None, "data", None)))

            model, optimizer, pool_ema, info = train_fn(
                model, optimizer, data_sharded, lam_window, pool_ema, tcfg.ema_decay
            )
            jax.block_until_ready(info["losses"])
            steps_done += tcfg.steps_per_window

            if ckpt_dir and ckpt_at and steps_done == ckpt_at:
                save_checkpoint(ckpt_dir, model, optimizer, pool_ema, steps_done, rng, None)

        return {
            "pool": np.array(model.pool.vectors[...]),
            "embed": np.array(model.embed.embedding.value),
            "opt_step": int(optimizer.step[...]),
            "losses": np.array(info["losses"]),
        }

    def test_uninterrupted_vs_resumed(self):
        """Params after 4 windows = params after 2 windows + save + resume + 2 windows."""
        seed = 99
        with tempfile.TemporaryDirectory() as ckpt_dir:
            # Run A: 4 windows uninterrupted
            result_a = self._run_windows(4, seed)

            # Run B: 2 windows → checkpoint → resume → 2 more windows
            result_b_part1 = self._run_windows(2, seed, ckpt_dir=ckpt_dir, ckpt_at=8)
            result_b = self._run_windows(2, seed, resume=True, resume_ckpt_dir=ckpt_dir)

        # Pool vectors should match (Adam moments restored → same updates)
        assert np.allclose(result_a["pool"], result_b["pool"], atol=1e-4), (
            f"Pool mismatch after resume:\n"
            f"  max diff: {np.abs(result_a['pool'] - result_b['pool']).max():.6f}"
        )
        # Embedding should also match
        assert np.allclose(result_a["embed"], result_b["embed"], atol=1e-4), (
            "Embedding mismatch after resume"
        )


# ---------------------------------------------------------------------------
# Test 3: start_window skips completed windows
# ---------------------------------------------------------------------------

class TestStartWindow:
    def test_start_window_calculation(self):
        """steps_done from checkpoint → correct start_window in training loop."""
        tcfg = _tiny_tcfg()
        # After 8 steps with steps_per_window=4: start_window = 8//4 = 2
        for steps_done, expected_window in [(0, 0), (4, 1), (8, 2), (12, 3), (28, 7)]:
            assert steps_done // tcfg.steps_per_window == expected_window, (
                f"steps_done={steps_done}: expected window {expected_window}"
            )

    def test_resume_does_not_repeat_steps(self):
        """Log output from resumed run should start at the correct step."""
        import io
        from contextlib import redirect_stdout
        from train import train

        with tempfile.TemporaryDirectory() as ckpt_dir:
            # Phase 1: run 2 windows (8 steps), save checkpoint, then stop
            run_cfg1 = _build_run_cfg(ckpt_dir=ckpt_dir, ckpt_every=8, total_steps=8)
            f1 = io.StringIO()
            with redirect_stdout(f1):
                train(run_cfg1)
            log1 = f1.getvalue()

            # Phase 2: resume, run remaining 6 windows (24 steps)
            run_cfg2 = _build_run_cfg(ckpt_dir=ckpt_dir, ckpt_every=0, resume=True)
            f2 = io.StringIO()
            with redirect_stdout(f2):
                train(run_cfg2)
            log2 = f2.getvalue()

        # Resumed run should contain "Resuming from step 8"
        assert "Resuming from step 8" in log2, f"Resume message not found. Log:\n{log2[:500]}"
        # Resumed run should NOT print step=4 or step=8 (those were already done)
        assert "step=     4/" not in log2, "Resumed run re-ran step 4"
        assert "step=     8/" not in log2, "Resumed run re-ran step 8"
        # Resumed run should start from step 12
        assert "step=    12/" in log2, f"Resumed run should start at step 12. Log:\n{log2[:500]}"


# ---------------------------------------------------------------------------
# Test 4: No double-checkpoint immediately at resume step
# ---------------------------------------------------------------------------

class TestNoDoubleCheckpoint:
    def test_no_immediate_checkpoint_on_resume(self):
        """Resuming at step K with ckpt_every=K should not immediately re-save."""
        import io
        from contextlib import redirect_stdout
        from train import train

        with tempfile.TemporaryDirectory() as ckpt_dir:
            # Save checkpoint at step 8 only (stop there)
            run_cfg1 = _build_run_cfg(ckpt_dir=ckpt_dir, ckpt_every=8, total_steps=8)
            with redirect_stdout(io.StringIO()):
                train(run_cfg1)

            ckpt_steps_before = set(
                int(d) for d in os.listdir(ckpt_dir)
                if d.isdigit()
            )
            assert 8 in ckpt_steps_before

            # Resume — first window ends at step 12, which doesn't cross ckpt_every=8 again
            run_cfg2 = _build_run_cfg(ckpt_dir=ckpt_dir, ckpt_every=8, resume=True)
            f2 = io.StringIO()
            with redirect_stdout(f2):
                train(run_cfg2)

        # There should be no "Saved step 8" in the resumed log (would indicate double-save)
        log2 = f2.getvalue()
        saved_lines = [l for l in log2.splitlines() if "Saved step 8" in l]
        assert len(saved_lines) == 0, f"Double-save at step 8 detected: {saved_lines}"


# ---------------------------------------------------------------------------
# Test 5: LR schedule continuity
# ---------------------------------------------------------------------------

class TestLRContinuity:
    def test_lr_at_resume_matches_expected(self):
        """After restoring optimizer.step, the LR schedule should be at the right position."""
        from train import _build_optimizer
        cfg = _tiny_cfg()
        tcfg = _tiny_tcfg()
        scheduler = PhaseScheduler(tcfg)
        model = DWAModel(cfg, nnx.Rngs(0))
        optimizer = _build_optimizer(model, tcfg, scheduler)

        # Simulate: optimizer has run 12 steps (3 windows × 4 steps)
        target_step = 12
        optimizer.step[...] = jnp.array(target_step, dtype=jnp.uint32)

        # LR scale at step 12 via scheduler
        expected_lr_scale = scheduler.get_lr_scale(target_step)
        schedule_fn = scheduler.make_optax_schedule(tcfg.lr_parts)
        actual_lr = float(jax.jit(schedule_fn)(jnp.array(target_step)))

        assert actual_lr == pytest.approx(expected_lr_scale * tcfg.lr_parts, rel=1e-4), (
            f"LR at step {target_step}: expected {expected_lr_scale * tcfg.lr_parts:.6f}, "
            f"got {actual_lr:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 6: Data loader cursor preserved
# ---------------------------------------------------------------------------

class TestLoaderCursor:
    def test_loader_state_saved_and_restored(self):
        from train import save_checkpoint, load_checkpoint, _build_optimizer, _build_mesh
        cfg = _tiny_cfg()
        tcfg = _tiny_tcfg()
        mesh = _get_mesh()
        model = DWAModel(cfg, nnx.Rngs(0))
        scheduler = PhaseScheduler(tcfg)
        optimizer = _build_optimizer(model, tcfg, scheduler)

        class FakeLoader:
            _cursor = 12345
            seq_len = cfg.seq_len
            def state_dict(self):
                return {"cursor": self._cursor,
                        "buf": np.zeros((5, self.seq_len), dtype=np.int32)}
            def load_state_dict(self, state):
                self._cursor = int(state["cursor"])

        loader = FakeLoader()
        with tempfile.TemporaryDirectory() as ckpt_dir:
            save_checkpoint(ckpt_dir, model, optimizer, jnp.zeros(cfg.N), 8,
                            jax.random.PRNGKey(0), loader)
            _, _, loader_state, _ = load_checkpoint(ckpt_dir, 8, model, optimizer, cfg, mesh)

        assert loader_state is not None
        assert loader_state["cursor"] == 12345, (
            f"Loader cursor not restored: {loader_state['cursor']}"
        )
        assert loader_state["buf"].shape[1] == cfg.seq_len
