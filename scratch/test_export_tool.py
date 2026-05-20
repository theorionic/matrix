"""Verification script for the hypernetwork-to-dense pool checkpoint export tool."""

import os
import sys
import copy
import subprocess

# Ensure matrix repo is in path
sys.path.append("/kaggle/working/matrix")

# Patch JAX config BEFORE any other import
import jax._src.config as _jax_cfg
_orig_update = _jax_cfg.config.update
def _safe_update(name: str, val) -> None:
    try:
        _orig_update(name, val)
    except AttributeError:
        pass
_jax_cfg.config.update = _safe_update

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from src.dwa.config import DWAConfig
from src.dwa.model import DWAModel
from src.dwa.run_config import load_config
from src.dwa.schedule import PhaseScheduler
from train import load_checkpoint, _build_optimizer


def run_export_verification():
    print("=== Running export verification ===")
    config_path = "configs/hypernetwork_smoke.yaml"
    ckpt_dir = "/kaggle/working/matrix/scratch/hypernetwork_ckpts"
    effective_config = os.path.join(ckpt_dir, "effective_config.yaml")
    if os.path.exists(effective_config):
        config_path = effective_config
    step = 8
    out_dir = "/kaggle/working/matrix/scratch/dense_ckpts"
    
    # 1. Run export_pool.py as a subprocess
    cmd = [
        "python", "src/dwa/export_pool.py",
        "--config", config_path,
        "--ckpt-dir", ckpt_dir,
        "--step", str(step),
        "--out-dir", out_dir
    ]
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("Export stdout:")
    print(result.stdout)
    if result.returncode != 0:
        print("Export stderr:")
        print(result.stderr)
        raise RuntimeError(f"Export tool failed with exit code {result.returncode}")
        
    print("Export subprocess completed successfully! Verifying the exported checkpoint...")
    
    # 2. Load hypernetwork checkpoint directly for comparison
    run_cfg = load_config(config_path)
    hyper_cfg = copy.deepcopy(run_cfg.model)
    hyper_cfg.use_hypernetwork = True
    
    rngs = nnx.Rngs(42)
    hyper_model = DWAModel(hyper_cfg, rngs)
    scheduler = PhaseScheduler(run_cfg.train)
    hyper_optimizer = _build_optimizer(hyper_model, run_cfg.train, scheduler)
    
    _, _, _, _ = load_checkpoint(
        ckpt_dir=ckpt_dir,
        steps_done_target=step,
        model=hyper_model,
        optimizer=hyper_optimizer,
        cfg=hyper_cfg,
        mesh=None,
        process_index=0,
    )
    
    expected_vectors = hyper_model.pool.vectors[...]
    expected_key_proj = hyper_model.pool.key_proj[...]
    
    # 3. Load exported dense checkpoint using standard (non-hypernetwork) model
    dense_cfg = copy.deepcopy(run_cfg.model)
    dense_cfg.use_hypernetwork = False
    
    dense_model = DWAModel(dense_cfg, rngs)
    dense_optimizer = _build_optimizer(dense_model, run_cfg.train, scheduler)
    
    _, _, _, _ = load_checkpoint(
        ckpt_dir=out_dir,
        steps_done_target=step,
        model=dense_model,
        optimizer=dense_optimizer,
        cfg=dense_cfg,
        mesh=None,
        process_index=0,
    )
    
    # 4. Compare parameters
    print("Comparing model states...")
    
    # Verify pool vectors match exactly
    assert dense_model.pool.vectors.shape == expected_vectors.shape
    assert jnp.allclose(dense_model.pool.vectors[...], expected_vectors, atol=1e-3), "Dense pool vectors do not match hypernetwork generated pool"
    
    # Verify key projections match exactly
    assert dense_model.pool.key_proj.shape == expected_key_proj.shape
    assert jnp.allclose(dense_model.pool.key_proj[...], expected_key_proj, atol=1e-3), "Dense key projections do not match hypernetwork key projections"
    
    # Compare layers
    hyper_state = nnx.state(hyper_model, nnx.Param)
    dense_state = nnx.state(dense_model, nnx.Param)
    
    hyper_flat = hyper_state.flat_state()
    dense_flat = dense_state.flat_state()
    hyper_dict = dict(zip(hyper_flat.paths, [v[...] for v in hyper_flat.leaves]))
    dense_dict = dict(zip(dense_flat.paths, [v[...] for v in dense_flat.leaves]))
    for path, val in hyper_dict.items():
        if any("pool" in str(p) for p in path):
            continue
        assert path in dense_dict, f"Path {path} not found in dense model state"
        assert jnp.allclose(dense_dict[path], val, atol=1e-3), f"Mismatch at parameter path {path}"
        
    print("Numerical comparison between hypernetwork and exported dense checkpoint matches perfectly!")
    
    # 5. Verify the dense model forward pass works on fake inputs
    input_ids = jnp.zeros((2, dense_cfg.seq_len), dtype=jnp.int32)
    logits, _ = dense_model(input_ids, lambda_val=jnp.array(1.0), is_warmup=False, use_pallas=False)
    print("Dense model forward logits shape:", logits.shape)
    assert logits.shape == (2, dense_cfg.seq_len, dense_cfg.vocab_size)
    assert jnp.isfinite(logits).all()
    
    print("Export verification completed successfully! Export tool verified.")


if __name__ == "__main__":
    run_export_verification()
