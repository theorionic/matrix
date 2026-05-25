"""
restore_grain_state.py — fast-forward GrainLoader to match an old StreamingLoader checkpoint.

Reads stories_consumed from the existing StreamingLoader checkpoint (loader_state_{pid}.npz),
fast-forwards the Grain iterator by that many items without tokenizing, then saves
a proper Grain checkpoint so future resumes are exact.

Usage:
    python restore_grain_state.py \
        --ckpt-dir /kaggle/working/matrix/checkpoints \
        --config configs/700m.yaml \
        [--step 31000]        # defaults to latest step found
        [--process-index 0]   # which host shard to restore (default: 0)
        [--process-count 1]   # total hosts (default: 1)
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir",       required=True)
    parser.add_argument("--config",         required=True)
    parser.add_argument("--step",           type=int, default=None,
                        help="Checkpoint step to restore from (default: latest)")
    parser.add_argument("--process-index",  type=int, default=0)
    parser.add_argument("--process-count",  type=int, default=1)
    args = parser.parse_args()

    from src.dwa.run_config import load_config
    run_cfg = load_config(args.config)

    import orbax.checkpoint as ocp
    mngr = ocp.CheckpointManager(args.ckpt_dir)
    step = args.step if args.step is not None else mngr.latest_step()
    if step is None:
        print("[ERROR] No checkpoint found in", args.ckpt_dir)
        sys.exit(1)
    print(f"[RestoreGrain] Using checkpoint step={step}")

    # --- Read stories_consumed from old StreamingLoader state ---
    streaming_path = os.path.join(args.ckpt_dir, str(step),
                                  f"loader_state_{args.process_index}.npz")
    if not os.path.exists(streaming_path):
        print(f"[ERROR] StreamingLoader state not found: {streaming_path}")
        print("  Nothing to restore — Grain will start from beginning.")
        sys.exit(1)

    d = np.load(streaming_path)
    stories_consumed = int(d["stories_consumed"]) if "stories_consumed" in d.files else 0
    print(f"[RestoreGrain] stories_consumed from old checkpoint: {stories_consumed:,}")

    if stories_consumed == 0:
        print("[RestoreGrain] stories_consumed=0, nothing to skip.")

    # --- Detect train split ---
    from train import _detect_hf_splits, _get_grain_ckpt_manager, HFParquetSource
    _train_split, _ = _detect_hf_splits(run_cfg.data.hf_path, run_cfg.data.hf_subset)

    # --- Build Grain source + sampler + loader (no tokenizer needed) ---
    import grain.python as grain

    source = HFParquetSource(
        repo_id=run_cfg.data.hf_path,
        split=_train_split,
        text_column=run_cfg.data.hf_text_column,
        hf_subset=run_cfg.data.hf_subset,
        shard_index=args.process_index,
        shard_count=args.process_count,
    )

    sampler = grain.IndexSampler(
        num_records=len(source),
        shard_options=grain.ShardOptions(
            shard_index=0,
            shard_count=1,
            drop_remainder=True,
        ),
        shuffle=True,
        num_epochs=None,
        seed=run_cfg.train.seed,
    )
    loader = grain.DataLoader(data_source=source, sampler=sampler, worker_count=0)
    it = iter(loader)

    # --- Fast-forward: consume stories_consumed items without tokenizing ---
    print(f"[RestoreGrain] Fast-forwarding {stories_consumed:,} items …")
    from tqdm import tqdm
    for _ in tqdm(range(stories_consumed), unit="story", smoothing=0.01):
        try:
            next(it)
        except StopIteration:
            it = iter(loader)
            next(it)
    print("[RestoreGrain] Fast-forward complete.")

    # --- Save Grain checkpoint ---
    grain_mngr = _get_grain_ckpt_manager(args.ckpt_dir, args.process_index, keep=3)
    grain_mngr.save(
        step,
        args=ocp.args.Composite(grain=grain.PyGrainCheckpointSave(it)),
    )
    grain_mngr.wait_until_finished()
    print(f"[RestoreGrain] Grain state saved → {args.ckpt_dir}/grain_p{args.process_index}/")

    # --- Also save an empty buf (buffer starts fresh; Grain position is exact) ---
    step_dir = os.path.join(args.ckpt_dir, str(step))
    os.makedirs(step_dir, exist_ok=True)
    buf_path = os.path.join(step_dir, f"loader_buf_{args.process_index}.npz")
    np.savez(buf_path, buf=np.empty((0, run_cfg.model.seq_len), dtype=np.int32))
    print(f"[RestoreGrain] Empty buf saved → {buf_path}")
    print("[RestoreGrain] Done. Resume training with checkpoint.resume: true")


if __name__ == "__main__":
    main()
