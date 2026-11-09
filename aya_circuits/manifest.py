"""Run manifests: git SHA, uv.lock hash, model id + revision, seed, dtype, device, wall time,
peak memory and sha256 of every input/output artifact (brief §11). AuditKIT fingerprints the
evaluation runs themselves; this covers the non-AuditKIT scripts and never duplicates its fields."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import torch

from .config import project_root


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> tuple[str, bool | None]:
    """(HEAD sha, working tree has uncommitted changes); ("unknown", None) outside a checkout."""
    try:
        run = lambda *a: subprocess.check_output(["git", *a], cwd=project_root(), text=True)  # noqa: E731
        return run("rev-parse", "HEAD").strip(), bool(run("status", "--porcelain").strip())
    except Exception:
        return "unknown", None


def model_revision(model_id: str) -> str | None:
    """Resolved commit hash of the cached HF snapshot, if present."""
    snaps = (
        Path.home()
        / ".cache/huggingface/hub"
        / ("models--" + model_id.replace("/", "--"))
        / "snapshots"
    )
    if snaps.exists():
        hashes = sorted(p.name for p in snaps.iterdir())
        return hashes[-1] if hashes else None
    return None


class Manifest:
    def __init__(self, stage: str, cfg: dict, seed: int | None = None):
        self.t0 = time.time()
        sha, dirty = git_state()
        self.data = {
            "stage": stage,
            "git_sha": sha,
            "git_dirty": dirty,
            "uv_lock_sha256": sha256_file(project_root() / "uv.lock"),
            "model": cfg["model"],
            "model_revision": model_revision(cfg["model"]),
            "seed": seed,
            "dtype": cfg["dtype"],
            "device": cfg["device"],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "config": cfg,
            "inputs": {},
            "outputs": {},
        }

    def add_input(self, path: str | Path) -> None:
        self.data["inputs"][str(path)] = sha256_file(path)

    def add_output(self, path: str | Path) -> None:
        self.data["outputs"][str(path)] = sha256_file(path)

    def write(self, path: str | Path, **extra) -> None:
        self.data["wall_time_s"] = round(time.time() - self.t0, 1)
        self.data.update(extra)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.data, indent=2, default=str))
