"""
Rolling parquet → Grain data loader with full background prefetching.

Two-level prefetch so get_window() never blocks the TPU:
  Level 1 — file download:   background thread downloads the next parquet file
                              while the current file is being consumed.
  Level 2 — tokenization:    background thread tokenizes + packs the next batch
                              while the model trains on the current batch.

Disk management:
  Files are downloaded to a controlled cache_dir (default ~/.cache/dwa_parquet/).
  The previous file is deleted immediately after the loader rotates to the next
  one, so at most 2 parquet files exist on disk at any time: the current file
  being consumed and the pre-downloaded next file.

File rotation:
  Each parquet file is loaded with num_epochs=1 (one full pass).
  When Grain raises StopIteration the background packer thread swaps in the
  pre-downloaded next file transparently — the main thread never sees it.

get_window() flow (steady state):
  1. Pull `steps * batch_size` rows from `_buf`.
  2. If `_buf` is running low, swap in next item from pack queue (near-instant).
  → TPU trains uninterrupted; CPU/network/disk work happens in parallel.
"""

from __future__ import annotations

import os
import threading
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Single-file Grain source
# ---------------------------------------------------------------------------

class LocalParquetSource:
    """
    Grain RandomAccessDataSource backed by one locally-downloaded parquet file.

    Loads all text rows into memory once — parquet files are 10-200 MB
    compressed and fit easily in RAM.  O(1) random access by index with no
    disk I/O per item, satisfying Grain's requirement for parallel workers.
    """

    def __init__(self, path: str, text_column: str = "text"):
        import pyarrow.parquet as pq
        table = pq.read_table(path, columns=[text_column])
        self._rows: list[str] = [
            v if isinstance(v, str) else str(v)
            for v in table.column(text_column).to_pylist()
        ]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> str:
        return self._rows[idx]

    def __repr__(self) -> str:
        return f"LocalParquetSource(n={len(self._rows)})"


# ---------------------------------------------------------------------------
# File-list resolver
# ---------------------------------------------------------------------------

def _list_parquet_files(
    repo_id: str,
    split: str,
    hf_subset: str = "",
) -> list[str]:
    """
    Return sorted list of repo-relative paths for all parquet shards of a split.
    Tries several layout patterns to handle different HF Hub repo structures.
    """
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    base = f"datasets/{repo_id}"

    candidates = []
    if hf_subset:
        candidates += [
            f"{base}/data/{hf_subset}/{split}-*.parquet",
            f"{base}/data/{hf_subset}/*.parquet",
        ]
    candidates += [
        f"{base}/data/{split}-*.parquet",
        f"{base}/data/*.parquet",
        f"{base}/data/**/*.parquet",
    ]

    for pattern in candidates:
        paths = sorted(fs.glob(pattern))
        if paths:
            prefix = f"datasets/{repo_id}/"
            return [p[len(prefix):] if p.startswith(prefix) else p for p in paths]

    raise FileNotFoundError(
        f"No parquet shards found for repo={repo_id!r} split={split!r} "
        f"subset={hf_subset!r}"
    )


# ---------------------------------------------------------------------------
# Rolling loader
# ---------------------------------------------------------------------------

class RollingParquetLoader:
    """
    Streams a HuggingFace dataset parquet-file-by-parquet-file through Grain,
    with two levels of background prefetch so the TPU never waits.

    Disk usage:
        Files are downloaded to cache_dir (default ~/.cache/dwa_parquet/).
        The consumed file is deleted immediately after rotation so only 2 files
        exist on disk at a time: current + pre-downloaded next.

    Level 1 — file prefetch:
        A daemon thread downloads file[i+1] while file[i] is being consumed.

    Level 2 — pack prefetch:
        A second daemon thread tokenises + packs batches into a Queue(depth=3).
        It handles file rotation transparently via StopIteration.
        get_window() pops from the queue — near-instant in steady state.

    Usage::

        loader = RollingParquetLoader(tokenizer=tok, seq_len=512,
                                      repo_id="roneneldan/TinyStories")
        window = loader.get_window(steps=64, batch_size=8)  # [64, 8, 512]
    """

    FETCH_SIZE  = 4_000   # texts per tokenisation call
    QUEUE_DEPTH = 3       # pre-packed batches buffered ahead of consumption

    def __init__(
        self,
        tokenizer,
        seq_len: int,
        repo_id: str = "roneneldan/TinyStories",
        split: str = "train",
        hf_subset: str = "",
        text_column: str = "text",
        shard_index: int = 0,
        shard_count: int = 1,
        seed: int = 42,
        worker_count: int = 0,
        cache_dir: str = "",
    ):
        self.tokenizer   = tokenizer
        self.seq_len     = seq_len
        self.eos         = tokenizer.eos_token_id or 0
        self._repo_id    = repo_id
        self._split      = split
        self._hf_subset  = hf_subset
        self._text_col   = text_column
        self._seed       = seed
        self._worker_cnt = worker_count
        self._cache_dir  = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "dwa_parquet"
        )
        os.makedirs(self._cache_dir, exist_ok=True)

        # Resolve shard list (tiny metadata request — no content download)
        all_paths = _list_parquet_files(repo_id, split, hf_subset)
        self._paths: list[str] = all_paths[shard_index::shard_count]
        if not self._paths:
            raise ValueError(
                f"No parquet files for shard {shard_index}/{shard_count}"
            )
        print(f"[RollingParquetLoader] {len(self._paths)}/{len(all_paths)} shards "
              f"for process {shard_index}/{shard_count}  "
              f"cache={self._cache_dir}")

        # ── Shared state (protected by _state_lock) ──────────────────────────
        self._state_lock    = threading.Lock()
        self._file_idx      = 0
        self._current_local: Optional[str] = None   # path of currently-open file
        self._grain_it      = None
        self._loader        = None

        # ── Level-1 prefetch: next file download ─────────────────────────────
        self._next_local: Optional[str] = None
        self._dl_ready   = threading.Event()
        self._dl_ready.set()

        # ── Level-2 prefetch: pre-packed token batches ───────────────────────
        import queue
        self._pack_queue: "queue.Queue[np.ndarray]" = queue.Queue(
            maxsize=self.QUEUE_DEPTH
        )
        self._packer_stop = threading.Event()

        # ── Bootstrap ────────────────────────────────────────────────────────
        local0 = self._download_sync(0)
        self._current_local = local0
        self._open_file(local0, file_idx=0)
        self._start_file_prefetch(1)

        self._packer_thread = threading.Thread(
            target=self._packer_loop, daemon=True, name="grain-packer"
        )
        self._packer_thread.start()

        first_batch = self._pack_queue.get(timeout=120)
        self._buf     = first_batch
        self._buf_pos = 0
        print(f"[RollingParquetLoader] Ready: {len(self._buf)} seqs buffered, "
              f"queue depth={self.QUEUE_DEPTH}")

    # ------------------------------------------------------------------
    # Level-1: file download + deletion
    # ------------------------------------------------------------------

    def _local_path(self, file_idx: int) -> str:
        """Deterministic local filename for a given file index."""
        idx  = file_idx % len(self._paths)
        name = os.path.basename(self._paths[idx])
        return os.path.join(self._cache_dir, name)

    def _download_sync(self, file_idx: int) -> str:
        """
        Download a parquet shard to cache_dir and return its local path.
        Uses local_dir= so the file lands directly in cache_dir (not the
        HF blob store), giving us full control to delete it after use.
        Skips the download if the file is already present.
        """
        from huggingface_hub import hf_hub_download
        idx  = file_idx % len(self._paths)
        path = self._paths[idx]
        local = hf_hub_download(
            repo_id=self._repo_id,
            filename=path,
            repo_type="dataset",
            local_dir=self._cache_dir,
        )
        size_mb = os.path.getsize(local) / 1e6
        print(f"[RollingParquetLoader] file[{idx}] downloaded: "
              f"{os.path.basename(local)}  ({size_mb:.0f} MB)")
        return local

    def _delete_file(self, local_path: Optional[str]) -> None:
        """Delete a consumed parquet file. Logs the freed space."""
        if not local_path or not os.path.exists(local_path):
            return
        try:
            size_mb = os.path.getsize(local_path) / 1e6
            os.remove(local_path)
            print(f"[RollingParquetLoader] Deleted: {os.path.basename(local_path)} "
                  f"({size_mb:.0f} MB freed)")
        except OSError as e:
            print(f"[RollingParquetLoader] Could not delete {local_path}: {e}")

    def _start_file_prefetch(self, next_idx: int) -> None:
        """Launch a daemon thread to pre-download file[next_idx]."""
        self._dl_ready.clear()
        self._next_local = None

        def _worker():
            try:
                self._next_local = self._download_sync(next_idx)
            except Exception as e:
                print(f"[RollingParquetLoader] File pre-download failed: {e}")
                self._next_local = None
            self._dl_ready.set()

        threading.Thread(target=_worker, daemon=True,
                         name="grain-file-dl").start()

    def _open_file(self, local_path: str, file_idx: int) -> None:
        """Build a Grain DataLoader for a downloaded parquet file."""
        import grain.python as grain

        source  = LocalParquetSource(local_path, self._text_col)
        sampler = grain.IndexSampler(
            num_records=len(source),
            shard_options=grain.ShardOptions(
                shard_index=0,
                shard_count=1,
                drop_remainder=False,
            ),
            shuffle=True,
            num_epochs=1,
            seed=self._seed + (file_idx % len(self._paths)),
        )
        self._loader   = grain.DataLoader(
            data_source=source,
            sampler=sampler,
            worker_count=self._worker_cnt,
        )
        self._grain_it = iter(self._loader)
        print(f"[RollingParquetLoader] Opened file[{file_idx % len(self._paths)}] "
              f"({len(source)} rows)")

    def _rotate_to_next_file(self) -> None:
        """
        Called by the packer thread when Grain exhausts the current file.

        Order of operations:
          1. Increment file index.
          2. Wait for Level-1 pre-download (almost always already done).
          3. Open the next file.
          4. Delete the just-consumed file from disk.
          5. Kick off Level-1 download for the file after that.
        """
        with self._state_lock:
            prev_local       = self._current_local
            self._file_idx  += 1
            next_idx         = self._file_idx

        idx = next_idx % len(self._paths)
        print(f"[RollingParquetLoader] File exhausted → rotating to file[{idx}]")

        if not self._dl_ready.is_set():
            print("[RollingParquetLoader] Waiting for file pre-download …")
        self._dl_ready.wait()

        local = self._next_local
        if local is None:
            local = self._download_sync(next_idx)

        with self._state_lock:
            self._open_file(local, file_idx=next_idx)
            self._current_local = local

        # Delete the consumed file NOW — it's no longer referenced by any reader.
        self._delete_file(prev_local)

        self._start_file_prefetch(next_idx + 1)

    # ------------------------------------------------------------------
    # Level-2: packer thread
    # ------------------------------------------------------------------

    def _packer_loop(self) -> None:
        """
        Background loop: pull FETCH_SIZE texts from Grain → tokenise → pack
        → enqueue.  Rotates files on StopIteration.  Runs until _packer_stop.
        """
        while not self._packer_stop.is_set():
            try:
                packed = self._fetch_and_pack()
                self._pack_queue.put(packed)
            except RuntimeError:
                break   # interpreter shutdown
            except Exception as e:
                import traceback
                print(f"[RollingParquetLoader] Packer error: {e}")
                traceback.print_exc()

    def _fetch_and_pack(self) -> np.ndarray:
        """
        Pull FETCH_SIZE texts, tokenise, and pack into [N, seq_len] int32.
        Calls _rotate_to_next_file() on StopIteration (file exhausted).
        """
        texts: list[str] = []
        while len(texts) < self.FETCH_SIZE:
            try:
                with self._state_lock:
                    t = next(self._grain_it)
                texts.append(t if isinstance(t, str) else str(t))
            except StopIteration:
                self._rotate_to_next_file()
                with self._state_lock:
                    t = next(self._grain_it)
                texts.append(t if isinstance(t, str) else str(t))

        ids_list = self.tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

        total = sum(len(s) + 2 for s in ids_list)
        flat  = np.empty(total, dtype=np.int32)
        pos   = 0
        for ids in ids_list:
            n = len(ids)
            flat[pos]                  = self.eos
            flat[pos + 1: pos + 1 + n] = ids
            flat[pos + 1 + n]          = self.eos
            pos += n + 2

        n_seqs = pos // self.seq_len
        return flat[: n_seqs * self.seq_len].reshape(n_seqs, self.seq_len)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        if self._pack_queue.empty():
            print("[RollingParquetLoader] WARNING: pack queue empty — "
                  "packer thread is behind; consider increasing FETCH_SIZE")
        new_seqs  = self._pack_queue.get(timeout=120)
        leftover  = self._buf[self._buf_pos:]
        self._buf = (
            np.concatenate([leftover, new_seqs], axis=0) if len(leftover)
            else new_seqs
        )
        self._buf_pos = 0

    def get_window(self, steps: int, batch_size: int) -> np.ndarray:
        """Return [steps, batch_size, seq_len] int32 of packed token IDs."""
        needed = steps * batch_size
        while len(self._buf) - self._buf_pos < needed:
            self._refill()
        out           = self._buf[self._buf_pos: self._buf_pos + needed]
        self._buf_pos += needed
        return out.reshape(steps, batch_size, self.seq_len)

    def state_dict(self) -> dict:
        """
        Snapshot loader position for exact resume.

        Saves file_index, Grain iterator bytes, and unconsumed token buffer.
        The file is re-downloaded on restore (it may have been deleted).
        """
        with self._state_lock:
            file_idx = self._file_idx
            grain_it = self._grain_it

        grain_bytes = b""
        if grain_it is not None:
            try:
                grain_bytes = grain_it.get_state()
            except Exception:
                pass

        return {
            "file_index":  file_idx,
            "grain_bytes": grain_bytes,
            "buf":         self._buf[self._buf_pos:].copy(),
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Restore loader to a checkpointed position.
        Stops + restarts the packer thread around the state change.
        Re-downloads the target file if it was deleted since the checkpoint.
        """
        self._packer_stop.set()
        while not self._pack_queue.empty():
            try:
                self._pack_queue.get_nowait()
            except Exception:
                break
        self._packer_thread.join(timeout=10)
        self._packer_stop.clear()

        self._dl_ready.wait()

        target_idx = int(state.get("file_index", 0))
        with self._state_lock:
            self._file_idx = target_idx

        # Re-download target file (may have been deleted after consumption)
        old_local = self._current_local
        local = self._download_sync(target_idx)
        with self._state_lock:
            self._open_file(local, file_idx=target_idx)
            self._current_local = local

        # Delete the file __init__ downloaded if we're restoring to a different file
        if old_local and old_local != local:
            self._delete_file(old_local)

        grain_bytes = state.get("grain_bytes", b"")
        if grain_bytes:
            try:
                with self._state_lock:
                    new_it = iter(self._loader)
                    new_it.set_state(bytes(grain_bytes))
                    self._grain_it = new_it
            except Exception as e:
                print(f"[RollingParquetLoader] Grain state restore failed ({e})")

        buf_data = state.get("buf")
        if buf_data is not None and len(buf_data):
            arr = np.asarray(buf_data, dtype=np.int32)
            self._buf = (
                arr if arr.ndim == 2
                else arr[: (len(arr) // self.seq_len) * self.seq_len
                         ].reshape(-1, self.seq_len)
            )
        else:
            self._buf = np.empty((0, self.seq_len), dtype=np.int32)
        self._buf_pos = 0

        self._start_file_prefetch(target_idx + 1)

        self._packer_thread = threading.Thread(
            target=self._packer_loop, daemon=True, name="grain-packer"
        )
        self._packer_thread.start()

        print(f"[RollingParquetLoader] Restored: file[{target_idx % len(self._paths)}], "
              f"{len(self._buf)} seqs buffered, packer restarted")

    def stats(self) -> dict:
        """Live loader metrics for logging."""
        with self._state_lock:
            file_idx = self._file_idx
        return {
            "file_idx":   file_idx % len(self._paths),
            "n_files":    len(self._paths),
            "queue_size": self._pack_queue.qsize(),
            "queue_max":  self.QUEUE_DEPTH,
            "buf_seqs":   max(0, len(self._buf) - self._buf_pos),
            "dl_ready":   self._dl_ready.is_set(),
        }

    def close(self) -> None:
        """Stop background threads gracefully."""
        self._packer_stop.set()
        while not self._pack_queue.empty():
            try:
                self._pack_queue.get_nowait()
            except Exception:
                break
