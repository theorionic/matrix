#!/usr/bin/env python3
"""
preflight.py — Pre-training environment validator for DWA.

Checks every dependency, tool, config, and resource before training starts.
Exits 0 if everything is OK (warnings allowed); exits 1 if any hard failures.

Usage:
    python preflight.py                          # default checks
    python preflight.py --config configs/small.yaml
    python preflight.py --ckpt-dir /tmp/ckpts --resume
    python preflight.py --strict                 # treat warnings as failures
    python preflight.py --no-network             # skip HuggingFace ping
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Callable

# ─── Result model ─────────────────────────────────────────────────────────────

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"
INFO = "INFO"

_ICON  = {PASS: "✓", WARN: "⚠", FAIL: "✗", SKIP: "—", INFO: "·"}
_COLOR = {
    PASS: "\033[32m",   # green
    WARN: "\033[33m",   # yellow
    FAIL: "\033[31m",   # red
    SKIP: "\033[90m",   # dark gray
    INFO: "\033[96m",   # cyan
}
_BOLD  = "\033[1m"
_RESET = "\033[0m"

USE_COLOR = sys.stdout.isatty()


@dataclass
class Result:
    name:   str
    status: str       # PASS / WARN / FAIL / SKIP / INFO
    detail: str = ""


def _c(status: str, text: str) -> str:
    if not USE_COLOR:
        return text
    return f"{_COLOR[status]}{text}{_RESET}"


def _b(text: str) -> str:
    return f"{_BOLD}{text}{_RESET}" if USE_COLOR else text


def print_result(r: Result) -> None:
    icon = _c(r.status, _ICON[r.status])
    tag  = _c(r.status, f"[{r.status}]")
    detail = f"  {r.detail}" if r.detail else ""
    print(f"  {icon} {tag:<14}  {r.name}{detail}")


def section(title: str) -> None:
    bar = "─" * 62
    print(f"\n{bar}")
    print(f"  {_b(title)}")
    print(bar)


# ─── Package version helpers ───────────────────────────────────────────────────

# Distribution name → (import name, required version | None)
# None version means "any version is fine"
REQUIRED_PKGS: dict[str, tuple[str, str | None]] = {
    "jax":               ("jax",            "0.10.0"),
    "jaxlib":            ("jaxlib",         "0.10.0"),
    "flax":              ("flax",           "0.12.7"),
    "optax":             ("optax",          "0.2.7"),
    "orbax-checkpoint":  ("orbax.checkpoint","0.11.33"),
    "numpy":             ("numpy",          "2.4.3"),
    "datasets":          ("datasets",       "4.8.3"),
    "transformers":      ("transformers",   "4.57.1"),
    "PyYAML":            ("yaml",           "6.0.3"),
}

OPTIONAL_PKGS: dict[str, tuple[str, str | None]] = {
    "grain":     ("grain.python", "0.2.16"),
    "wandb":     ("wandb",        None),
}


def _check_pkg(dist_name: str, import_name: str, required_ver: str | None,
               optional: bool = False) -> Result:
    # 1. Check distribution metadata
    try:
        installed = _pkg_version(dist_name)
    except PackageNotFoundError:
        status = WARN if optional else FAIL
        return Result(dist_name, status,
                      f"not installed   pip install {dist_name}"
                      + ("" if not optional else "  (optional)"))

    # 2. Check import
    try:
        top = import_name.split(".")[0]
        __import__(top)
    except Exception as exc:
        return Result(dist_name, FAIL,
                      f"installed {installed} but import failed: {exc}")

    # 3. Version match
    if required_ver is None:
        return Result(dist_name, PASS, f"version {installed}")

    if installed == required_ver:
        return Result(dist_name, PASS, f"version {installed}")

    # Semantic version compare (major.minor.patch)
    def _parts(v: str):
        try:
            return tuple(int(x) for x in v.split(".")[:3])
        except ValueError:
            return (0,)

    inst_t = _parts(installed)
    req_t  = _parts(required_ver)
    status = WARN   # mismatch is a warning — might still work
    note   = (f"version {installed}  (pinned {required_ver})"
              + ("  [newer]" if inst_t > req_t else "  [older — consider upgrading]"))
    return Result(dist_name, status, note)


# ─── Individual section checkers ──────────────────────────────────────────────

def check_python() -> list[Result]:
    results = []
    maj, min_, _ = sys.version_info[:3]
    ver_str = f"Python {sys.version.split()[0]}  ({platform.python_implementation()})"
    if (maj, min_) >= (3, 10):
        results.append(Result("Python version", PASS, ver_str))
    else:
        results.append(Result("Python version", FAIL,
                               f"{ver_str}  — requires ≥ 3.10"))
    bits = "64-bit" if sys.maxsize > 2**32 else "32-bit"
    status = PASS if bits == "64-bit" else FAIL
    results.append(Result("Platform", status,
                           f"{platform.system()} {platform.machine()}  {bits}"))
    return results


def check_core_libraries() -> list[Result]:
    return [_check_pkg(d, imp, ver) for d, (imp, ver) in REQUIRED_PKGS.items()]


def check_optional_libraries() -> list[Result]:
    return [_check_pkg(d, imp, ver, optional=True)
            for d, (imp, ver) in OPTIONAL_PKGS.items()]


def check_jax_devices() -> list[Result]:
    results = []
    try:
        import jax
    except Exception as exc:
        results.append(Result("JAX import", FAIL, str(exc)))
        return results

    # Backend
    try:
        backend = jax.default_backend()
        results.append(Result("JAX backend", INFO, backend))
    except Exception as exc:
        results.append(Result("JAX backend", WARN, str(exc)))
        backend = "unknown"

    # Devices
    try:
        devices = jax.devices()
        n       = len(devices)
        kind    = devices[0].device_kind if devices else "none"
        is_tpu  = "tpu" in kind.lower()
        is_gpu  = "gpu" in kind.lower() or "cuda" in kind.lower()
        status  = PASS if (is_tpu or is_gpu) else WARN
        results.append(Result("Device count", status,
                               f"{n}× {kind}"
                               + ("  [TPU]" if is_tpu else "  [GPU]" if is_gpu
                                  else "  [CPU — no accelerator detected]")))
    except Exception as exc:
        results.append(Result("Device count", FAIL, str(exc)))
        return results

    # Local vs global
    try:
        local_n = jax.local_device_count()
        total_n = jax.device_count()
        if local_n != total_n:
            results.append(Result("Multi-host", INFO,
                                   f"local={local_n}  total={total_n}  hosts={total_n//local_n}"))
        else:
            results.append(Result("Multi-host", INFO,
                                   f"single host  {local_n} device{'s' if local_n!=1 else ''}"))
    except Exception:
        pass

    # bf16 support probe
    try:
        import jax.numpy as jnp
        x = jnp.ones((2, 2), dtype=jnp.bfloat16)
        y = x @ x
        jax.block_until_ready(y)
        results.append(Result("bfloat16 matmul", PASS, "OK"))
    except Exception as exc:
        results.append(Result("bfloat16 matmul", WARN, str(exc)))

    # XLA cache dir writable
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR",
                                os.path.join(os.path.dirname(__file__), ".jax_cache"))
    os.makedirs(cache_dir, exist_ok=True)
    probe = os.path.join(cache_dir, ".write_probe")
    try:
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        results.append(Result("XLA cache dir", PASS, cache_dir))
    except Exception as exc:
        results.append(Result("XLA cache dir", WARN,
                               f"not writable: {exc}  ({cache_dir})"))

    return results


def check_multihost_env() -> list[Result]:
    results = []
    coord   = os.environ.get("JAX_COORDINATOR_ADDRESS", "")
    n_proc  = os.environ.get("JAX_NUM_PROCESSES", "")
    proc_id = os.environ.get("JAX_PROCESS_ID", "")

    all_set  = all([coord, n_proc, proc_id])
    none_set = not any([coord, n_proc, proc_id])

    if none_set:
        results.append(Result("Multi-host env vars", SKIP,
                               "not set — single-host mode"))
        return results

    if not all_set:
        missing = [k for k, v in [
            ("JAX_COORDINATOR_ADDRESS", coord),
            ("JAX_NUM_PROCESSES", n_proc),
            ("JAX_PROCESS_ID", proc_id),
        ] if not v]
        results.append(Result("Multi-host env vars", FAIL,
                               f"partially set — missing: {', '.join(missing)}"))
        return results

    # Validate types and ranges
    errors = []
    try:
        np = int(n_proc)
        if np < 2:
            errors.append(f"JAX_NUM_PROCESSES={np} must be ≥ 2 for multi-host")
    except ValueError:
        errors.append(f"JAX_NUM_PROCESSES={n_proc!r} is not an integer")

    try:
        pid = int(proc_id)
        np  = int(n_proc) if n_proc.isdigit() else 0
        if pid < 0 or (np > 0 and pid >= np):
            errors.append(f"JAX_PROCESS_ID={pid} out of range [0, {np-1}]")
    except ValueError:
        errors.append(f"JAX_PROCESS_ID={proc_id!r} is not an integer")

    if ":" not in coord:
        errors.append(f"JAX_COORDINATOR_ADDRESS={coord!r} missing port (expected host:port)")

    if errors:
        results.append(Result("Multi-host env vars", FAIL, "; ".join(errors)))
    else:
        results.append(Result("Multi-host env vars", PASS,
                               f"coordinator={coord}  n={n_proc}  id={proc_id}"))
    return results


def check_cli_tools() -> list[Result]:
    results = []

    # ── rclone ───────────────────────────────────────────────────────────────
    rclone_path = shutil.which("rclone")
    if rclone_path is None:
        results.append(Result("rclone", WARN,
                               "not found  —  install: apt-get install -y rclone"))
        results.append(Result("rclone remotes", SKIP, "rclone not installed"))
    else:
        # Version
        try:
            out = subprocess.check_output(["rclone", "--version"],
                                           stderr=subprocess.DEVNULL, text=True, timeout=10)
            ver_line = out.splitlines()[0] if out else "?"
            results.append(Result("rclone", PASS, ver_line.strip()))
        except Exception as exc:
            results.append(Result("rclone", WARN, f"found at {rclone_path} but version check failed: {exc}"))

        # Configured remotes
        try:
            out = subprocess.check_output(["rclone", "listremotes"],
                                           stderr=subprocess.DEVNULL, text=True, timeout=15)
            remotes = [r.rstrip(":").strip() for r in out.splitlines() if r.strip()]
            if not remotes:
                results.append(Result("rclone remotes", WARN,
                                       "no remotes configured  —  run: rclone config"))
            else:
                has_gdrive = any("gdrive" in r.lower() or "drive" in r.lower() for r in remotes)
                status = PASS if has_gdrive else WARN
                note   = f"{', '.join(remotes)}"
                if not has_gdrive:
                    note += "  (no gdrive remote found — add one with: rclone config)"
                results.append(Result("rclone remotes", status, note))
        except Exception as exc:
            results.append(Result("rclone remotes", WARN, f"could not list remotes: {exc}"))

    # ── git ──────────────────────────────────────────────────────────────────
    git_path = shutil.which("git")
    if git_path is None:
        results.append(Result("git", WARN, "not found"))
    else:
        try:
            ver = subprocess.check_output(["git", "--version"],
                                           stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
            results.append(Result("git", PASS, ver))
        except Exception:
            results.append(Result("git", WARN, f"found at {git_path} but version check failed"))

    return results


def check_project_structure() -> list[Result]:
    results = []
    root = os.path.dirname(os.path.abspath(__file__))

    required_files = [
        "train.py",
        "main.py",
        "requirements.txt",
        "src/dwa/__init__.py",
        "src/dwa/assembly.py",
        "src/dwa/assembly_pallas.py",
        "src/dwa/config.py",
        "src/dwa/losses.py",
        "src/dwa/model.py",
        "src/dwa/monitor.py",
        "src/dwa/parts.py",
        "src/dwa/pool.py",
        "src/dwa/retrieval.py",
        "src/dwa/run_config.py",
        "src/dwa/schedule.py",
        "src/dwa/utils.py",
    ]

    missing = []
    for rel in required_files:
        if not os.path.isfile(os.path.join(root, rel)):
            missing.append(rel)

    if missing:
        for m in missing:
            results.append(Result(m, FAIL, "missing"))
    else:
        results.append(Result("Source files", PASS,
                               f"all {len(required_files)} required files present"))

    # configs dir
    configs_dir = os.path.join(root, "configs")
    if os.path.isdir(configs_dir):
        yamls = [f for f in os.listdir(configs_dir) if f.endswith(".yaml")]
        results.append(Result("configs/", PASS, f"{len(yamls)} YAML files: {', '.join(yamls)}"))
    else:
        results.append(Result("configs/", WARN, "directory not found"))

    # src/dwa importable
    try:
        if root not in sys.path:
            sys.path.insert(0, root)
        import importlib
        spec = importlib.util.find_spec("src.dwa.run_config")
        if spec:
            results.append(Result("src.dwa package", PASS, "importable"))
        else:
            results.append(Result("src.dwa package", FAIL, "not found in sys.path"))
    except Exception as exc:
        results.append(Result("src.dwa package", WARN, str(exc)))

    return results


def check_disk_and_memory(ckpt_dir: str | None = None) -> list[Result]:
    results = []
    root = os.path.dirname(os.path.abspath(__file__))

    # Working directory free space
    try:
        usage   = shutil.disk_usage(root)
        free_gb = usage.free / 1024**3
        total_gb = usage.total / 1024**3
        status  = PASS if free_gb >= 5 else (WARN if free_gb >= 2 else FAIL)
        results.append(Result("Disk (working dir)", status,
                               f"{free_gb:.1f} GB free / {total_gb:.1f} GB total"))
    except Exception as exc:
        results.append(Result("Disk (working dir)", WARN, str(exc)))

    # Checkpoint directory free space
    if ckpt_dir:
        ckpt_abs = os.path.abspath(ckpt_dir)
        os.makedirs(ckpt_abs, exist_ok=True)
        try:
            usage   = shutil.disk_usage(ckpt_abs)
            free_gb = usage.free / 1024**3
            status  = PASS if free_gb >= 10 else (WARN if free_gb >= 3 else FAIL)
            results.append(Result("Disk (ckpt dir)", status,
                                   f"{free_gb:.1f} GB free  ({ckpt_abs})"))
        except Exception as exc:
            results.append(Result("Disk (ckpt dir)", WARN, str(exc)))

        # Writability
        probe = os.path.join(ckpt_abs, ".write_probe")
        try:
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
            results.append(Result("Ckpt dir writable", PASS, ckpt_abs))
        except Exception as exc:
            results.append(Result("Ckpt dir writable", FAIL, str(exc)))

    # RAM
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        info = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                info[parts[0].rstrip(":")] = int(parts[1])   # kB
        total_gb = info.get("MemTotal", 0) / 1024**2
        avail_gb = info.get("MemAvailable", 0) / 1024**2
        status   = PASS if avail_gb >= 8 else (WARN if avail_gb >= 4 else FAIL)
        results.append(Result("RAM available", status,
                               f"{avail_gb:.1f} GB free / {total_gb:.1f} GB total"))
    except Exception:
        # /proc/meminfo only on Linux; skip on other platforms
        results.append(Result("RAM available", SKIP, "not available on this platform"))

    return results


def check_run_config(config_path: str, resume: bool = False) -> list[Result]:
    results = []

    # File exists
    if not os.path.isfile(config_path):
        results.append(Result("Config file", FAIL, f"not found: {config_path}"))
        return results
    results.append(Result("Config file", PASS, config_path))

    # Load and parse
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        if root not in sys.path:
            sys.path.insert(0, root)
        from src.dwa.run_config import load_config
        cfg = load_config(config_path)
        results.append(Result("Config load", PASS, f"name={cfg.name!r}"))
    except Exception as exc:
        results.append(Result("Config load", FAIL, str(exc)))
        return results

    # Sanity-check key values
    tcfg = cfg.train
    mcfg = cfg.model

    checks = [
        ("batch_size",       tcfg.batch_size,       lambda v: v > 0,           "must be > 0"),
        ("total_steps",      tcfg.total_steps,       lambda v: v > 0,           "must be > 0"),
        ("steps_per_window", tcfg.steps_per_window,  lambda v: v > 0,           "must be > 0"),
        ("warmup_steps",     tcfg.warmup_steps,      lambda v: 0 < v < tcfg.total_steps,
                                                                                 "must be in (0, total_steps)"),
        ("N (pool size)",    mcfg.N,                 lambda v: v >= 4,          "must be ≥ 4"),
        ("D (vector dim)",   mcfg.D,                 lambda v: v >= 256,        "must be ≥ 256"),
        ("d_A",              mcfg.d_A,               lambda v: v >= 32,         "must be ≥ 32"),
        ("seq_len",          mcfg.seq_len,           lambda v: v >= 32,         "must be ≥ 32"),
        ("vocab_size",       mcfg.vocab_size,        lambda v: v >= 256,        "must be ≥ 256"),
        ("k_max",            mcfg.k_max,             lambda v: v <= mcfg.N,    f"must be ≤ N={mcfg.N}"),
    ]
    for label, val, ok, msg in checks:
        if ok(val):
            results.append(Result(f"  {label}", PASS, str(val)))
        else:
            results.append(Result(f"  {label}", FAIL, f"{val}  — {msg}"))

    # Data source
    source = cfg.data.source
    if source not in ("random", "tiny_stories", "pattern"):
        results.append(Result("  data.source", FAIL,
                               f"{source!r} — must be random | tiny_stories | pattern"))
    else:
        results.append(Result("  data.source", PASS, source))

    # Checkpoint dir consistency
    ckpt_dir = cfg.checkpoint.dir
    if resume and not ckpt_dir:
        results.append(Result("  resume w/o ckpt_dir", FAIL,
                               "checkpoint.resume=true but checkpoint.dir is empty"))
    elif ckpt_dir:
        results.append(Result("  checkpoint.dir", INFO, ckpt_dir))

    # GDrive consistency
    gd = cfg.gdrive
    if gd.enabled:
        if not gd.remote_path:
            results.append(Result("  gdrive.remote_path", FAIL,
                                   "gdrive.enabled=true but remote_path is empty"))
        else:
            results.append(Result("  gdrive", PASS,
                                   f"{gd.rclone_remote}:{gd.remote_path}"))

    # WandB consistency
    wb = cfg.wandb
    if wb.enabled:
        try:
            import wandb as _wb  # noqa: F401
            results.append(Result("  wandb", PASS, f"project={wb.project!r}"))
        except ImportError:
            results.append(Result("  wandb", WARN,
                                   "wandb.enabled=true but wandb is not installed"))

    return results


def check_checkpoint(ckpt_dir: str, resume: bool = False) -> list[Result]:
    results = []
    ckpt_abs = os.path.abspath(ckpt_dir)

    if not os.path.isdir(ckpt_abs):
        if resume:
            results.append(Result("Checkpoint dir", WARN,
                                   f"does not exist yet: {ckpt_abs}"))
        else:
            results.append(Result("Checkpoint dir", INFO,
                                   f"will be created at: {ckpt_abs}"))
        return results

    results.append(Result("Checkpoint dir", PASS, ckpt_abs))

    if not resume:
        return results

    # Try to find the latest step via orbax
    try:
        import orbax.checkpoint as ocp
        mngr    = ocp.CheckpointManager(ckpt_abs)
        latest  = mngr.latest_step()
        all_steps = mngr.all_steps()
        if latest is None:
            results.append(Result("Latest checkpoint", WARN,
                                   "no checkpoints found in dir — will start fresh"))
        else:
            results.append(Result("Latest checkpoint", PASS,
                                   f"step {latest}  (available: {list(all_steps)})"))
    except Exception as exc:
        results.append(Result("Latest checkpoint", WARN,
                               f"could not probe with orbax: {exc}"))

    return results


def check_network(hf_url: str = "https://huggingface.co") -> list[Result]:
    results = []
    try:
        import urllib.request
        req = urllib.request.Request(hf_url, method="HEAD",
                                      headers={"User-Agent": "preflight/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            code = resp.status
        results.append(Result("HuggingFace reachable", PASS,
                               f"{hf_url}  HTTP {code}"))
    except Exception as exc:
        results.append(Result("HuggingFace reachable", WARN,
                               f"could not reach {hf_url}: {exc}"
                               "  (offline datasets only)"))
    return results


# ─── Summary and main ─────────────────────────────────────────────────────────

def _print_summary(all_results: list[Result], strict: bool) -> int:
    counts = {PASS: 0, WARN: 0, FAIL: 0, SKIP: 0, INFO: 0}
    for r in all_results:
        counts[r.status] += 1

    hard_fails = counts[FAIL] + (counts[WARN] if strict else 0)

    bar = "═" * 62
    print(f"\n{bar}")
    total = sum(counts.values())
    parts = [
        _c(PASS, f"{counts[PASS]} passed"),
        _c(WARN, f"{counts[WARN]} warning{'s' if counts[WARN]!=1 else ''}"),
        _c(FAIL, f"{counts[FAIL]} failed"),
    ]
    if counts[SKIP]:
        parts.append(_c(SKIP, f"{counts[SKIP]} skipped"))
    print(f"  {_b('Summary')}: {total} checks — {', '.join(parts)}")

    if hard_fails == 0:
        print(f"\n  {_c(PASS, '✓')} {_b('Environment OK — ready to train.')}")
    else:
        noun = "error" if not strict else "issue"
        print(f"\n  {_c(FAIL, '✗')} {_b(f'{hard_fails} blocking {noun}(s) — fix before training.')}")
        if strict and counts[WARN]:
            print(f"    (--strict treats warnings as failures)")

    print(f"{bar}\n")
    return 0 if hard_fails == 0 else 1


def _apply_jax_compat_patch() -> None:
    """
    Replicate the train.py JAX config patch so optax imports cleanly.

    Some optax versions call jax.config.update('jax_pmap_shmap_merge', False)
    which raises AttributeError on newer JAX.  train.py monkey-patches config.update
    to suppress this; we do the same here so the preflight import check reflects
    the actual runtime behavior.
    """
    try:
        import jax._src.config as _jax_cfg
        _orig = _jax_cfg.config.update
        def _safe(name: str, val) -> None:
            try:
                _orig(name, val)
            except AttributeError:
                pass
        _jax_cfg.config.update = _safe
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-training environment validator for DWA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",    metavar="PATH",
                        help="RunConfig YAML to validate (optional)")
    parser.add_argument("--ckpt-dir",  metavar="PATH",
                        help="Checkpoint directory to check (optional)")
    parser.add_argument("--resume",    action="store_true",
                        help="Expect a checkpoint to exist in --ckpt-dir")
    parser.add_argument("--strict",    action="store_true",
                        help="Treat warnings as failures (exit 1)")
    parser.add_argument("--no-network", action="store_true",
                        help="Skip HuggingFace connectivity check")
    args = parser.parse_args()

    # Apply the JAX compat patch BEFORE any library imports are attempted
    # (mirrors what train.py does to suppress the optax/jax_pmap_shmap_merge error)
    _apply_jax_compat_patch()

    bar = "═" * 62
    print(f"\n{bar}")
    print(f"  {_b('DWA Pre-Training Validator')}")
    print(f"  Working dir: {os.path.abspath('.')}")
    print(f"{bar}")

    all_results: list[Result] = []

    def run_section(title: str, fn: Callable[[], list[Result]]) -> None:
        section(title)
        try:
            results = fn()
        except Exception as exc:
            results = [Result(title, FAIL, f"checker crashed: {exc}")]
        for r in results:
            print_result(r)
        all_results.extend(results)

    run_section("1. Python Runtime",          check_python)
    run_section("2. Core Libraries",          check_core_libraries)
    run_section("3. Optional Libraries",      check_optional_libraries)
    run_section("4. JAX Devices",             check_jax_devices)
    run_section("5. Multi-Host Environment",  check_multihost_env)
    run_section("6. CLI Tools",               check_cli_tools)
    run_section("7. Project Structure",       check_project_structure)
    run_section("8. Disk & Memory",           lambda: check_disk_and_memory(args.ckpt_dir))

    if args.config:
        run_section("9. Run Config",
                    lambda: check_run_config(args.config, resume=args.resume))

    if args.ckpt_dir:
        run_section("10. Checkpoint Directory",
                    lambda: check_checkpoint(args.ckpt_dir, resume=args.resume))

    if not args.no_network:
        run_section("11. Network Connectivity", check_network)

    exit_code = _print_summary(all_results, strict=args.strict)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
