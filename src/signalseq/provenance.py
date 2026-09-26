"""Per-run provenance: the resolved config, package version and git commit.

Written into every output directory so a result can always be traced back to the
exact parameters and code that produced it.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import yaml

from . import __version__
from .config import Config, repo_root


def _git_describe(root: Path | None) -> dict:
    """Commit hash and dirty state of the source tree, if it is a git checkout."""
    if root is None:
        return {"available": False, "reason": "not running from a git checkout"}
    try:
        def git(*args):
            return subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()

        return {
            "available": True,
            "commit": git("rev-parse", "HEAD"),
            "commit_short": git("rev-parse", "--short", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (subprocess.SubprocessError, OSError) as exc:
        return {"available": False, "reason": f"git call failed: {exc}"}


def _dependency_versions() -> dict:
    out = {}
    for mod in ("numpy", "pandas", "scipy", "anndata", "scanpy", "numba", "matplotlib",
                "muon", "phate", "h5py"):
        try:
            import importlib.metadata as md
            out[mod] = md.version(mod)
        except Exception:
            out[mod] = None
    return out


def write(cfg: Config, out_dir: Path, command: str, extra: dict | None = None) -> dict:
    """Write resolved_config.yaml and provenance.json into `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "resolved_config.yaml", "w") as fh:
        yaml.safe_dump(_stringify(cfg.raw), fh, sort_keys=False, default_flow_style=False)

    record = {
        "run_name": cfg.name,
        "command": command,
        "signalseq_version": __version__,
        "git": _git_describe(repo_root()),
        "config_path": str(cfg.path),
        "output_dir": str(cfg.output_dir),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": _dependency_versions(),
        "read_structure": cfg.read_structure.diagram(),
    }
    if extra:
        record.update(extra)

    with open(out_dir / "provenance.json", "w") as fh:
        json.dump(record, fh, indent=2, default=str)
    return record


def _stringify(obj):
    """Make the resolved config YAML-safe (Paths -> str, tuples -> lists)."""
    if isinstance(obj, dict):
        return {str(k): _stringify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stringify(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
