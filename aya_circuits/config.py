"""YAML config loading with `key.sub=value` CLI overrides. Source of truth: configs/base.yaml in the
study checkout (`project_root()`)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """The circuit-steering checkout the CLIs run from (configs/, data/, runs/, .cache/ live there);
    `CIRCUIT_STEERING_ROOT` overrides the working directory."""
    return Path(os.environ.get("CIRCUIT_STEERING_ROOT") or Path.cwd()).resolve()


def _coerce(v: str) -> Any:
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def _set(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(argv: list[str] | None = None, path: str | Path | None = None) -> dict:
    """Load the base YAML and apply `a.b=c` overrides from argv (defaults to sys.argv[1:])."""
    path = Path(path) if path else project_root() / "configs/base.yaml"
    cfg = yaml.safe_load(Path(path).read_text())
    for arg in sys.argv[1:] if argv is None else argv:
        if "=" not in arg:
            raise SystemExit(f"override must look like key.sub=value, got {arg!r}")
        k, v = arg.split("=", 1)
        _set(cfg, k, _coerce(v))
    return cfg
