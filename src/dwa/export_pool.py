"""Export utility to extract standard dense pool from hypernetwork checkpoint."""

import os
import sys

# Patch JAX config BEFORE any other import (fixes optax/JAX version mismatch)
import jax._src.config as _jax_cfg
_orig_update = _jax_cfg.config.update
def _safe_update(name: str, val) -> None:
    try:
        _orig_update(name, val)
    except AttributeError:
        pass
_jax_cfg.config.update = _safe_update

import copy
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import orbax.checkpoint as ocp

# Ensure matrix repo is in path
sys.path.append("/kaggle/working/matrix")

from src.dwa.config import DWAConfig, TrainConfig
from src.dwa.model import DWAModel
from src.dwa.run_config import load_config
from src.dwa.schedule import PhaseScheduler
from train import load_checkpoint, _build_optimizer


def export_hypernetwork_checkpoint(config_path: str, ckpt_dir: str, step: int, out_dir: str):
    """
    Load a hypernetwork checkpoint, compute the full NxD dense pool,
    and save a clean inference-ready standard dense pool checkpoint.
    """
    print(f"Loading run config from {config_path}...")
    run_cfg = load_config(config_path)
    
    # 1. Force use_hypernetwork = True on the configuration to match the saved checkpoint
    hyper_cfg = copy.deepcopy(run_cfg.model)
    hyper_cfg.use_hypernetwork = True
    
    print("Initializing source (hypernetwork) model and optimizer...")
    rngs = nnx.Rngs(42)
    hyper_model = DWAModel(hyper_cfg, rngs)
    
    # Construct optimizer so load_checkpoint doesn't complain and can load optimizer state
    scheduler = PhaseScheduler(run_cfg.train)
    hyper_optimizer = _build_optimizer(hyper_model, run_cfg.train, scheduler)
    
    print(f"Restoring checkpoint at step {step} from {ckpt_dir}...")
    steps_done, rng, loader_state_dict, pool_ema = load_checkpoint(
        ckpt_dir=ckpt_dir,
        steps_done_target=step,
        model=hyper_model,
        optimizer=hyper_optimizer,
        cfg=hyper_cfg,
        mesh=None,
        process_index=0,
    )
    assert steps_done == step, f"Expected step {step}, got {steps_done}"
    
    print("Computing full [N, D] dense vector pool from hypernetwork generator...")
    # Generate all vectors and keys
    dense_vectors = hyper_model.pool.generate_all_vectors()
    key_proj = hyper_model.pool.key_proj[...]
    
    print("Initializing destination (standard/dense) model and optimizer...")
    dense_cfg = copy.deepcopy(run_cfg.model)
    dense_cfg.use_hypernetwork = False
    
    dense_model = DWAModel(dense_cfg, rngs)
    dense_optimizer = _build_optimizer(dense_model, run_cfg.train, scheduler)
    
    # Copy parameters from hyper_model to dense_model
    print("Transferring parameters to destination model state...")
    hyper_state = nnx.state(hyper_model, nnx.Param)
    dense_state = nnx.state(dense_model, nnx.Param)
    
    # In newer NNX, state is a nested dictionary.
    # Update all common weights by copying non-pool top-level keys
    for key in list(hyper_state.keys()):
        if key == "pool":
            continue
        dense_state[key] = hyper_state[key]
        
    nnx.update(dense_model, dense_state)
    
    # Direct parameter assignment for pool:
    dense_model.pool.vectors[...] = dense_vectors
    dense_model.pool.key_proj[...] = key_proj
    
    print(f"Saving compiled standard dense-pool checkpoint to {out_dir} at step {step}...")
    os.makedirs(out_dir, exist_ok=True)
    mngr = ocp.CheckpointManager(
        out_dir,
        options=ocp.CheckpointManagerOptions(max_to_keep=3),
    )
    
    model_np = jax.tree_util.tree_map(np.array, nnx.state(dense_model, nnx.Param))
    opt_leaves, _ = jax.tree_util.tree_flatten(dense_optimizer.opt_state)
    opt_dict = {f"{i:04d}": np.array(leaf) for i, leaf in enumerate(opt_leaves)}
    
    save_item = {
        "model": model_np,
        "opt":   opt_dict,
        "meta": {
            "opt_step":   np.array(int(dense_optimizer.step[...]), dtype=np.int32),
            "pool_ema":   np.array(pool_ema, dtype=np.float32),
            "steps_done": np.array(steps_done, dtype=np.int32),
            "rng":        np.array(rng),
        },
    }
    mngr.save(steps_done, args=ocp.args.StandardSave(save_item))
    mngr.wait_until_finished()
    
    # If a loader_state npz file existed at the source step directory, copy it to the target step directory
    src_step_dir = os.path.join(ckpt_dir, str(step))
    dst_step_dir = os.path.join(out_dir, str(step))
    os.makedirs(dst_step_dir, exist_ok=True)
    import shutil
    for f in os.listdir(src_step_dir) if os.path.exists(src_step_dir) else []:
        if f.endswith(".npz"):
            src_f = os.path.join(src_step_dir, f)
            dst_f = os.path.join(dst_step_dir, f)
            print(f"Copying loader state {f} to {dst_step_dir}...")
            shutil.copy2(src_f, dst_f)
            
    print(f"Successfully exported checkpoint to: {out_dir}/{step}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export hypernetwork pool generator checkpoint to standard dense pool.")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Directory containing source checkpoints")
    parser.add_argument("--step", type=int, required=True, help="Step to load/export")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to write exported checkpoint")
    args = parser.parse_args()
    
    export_hypernetwork_checkpoint(args.config, args.ckpt_dir, args.step, args.out_dir)
