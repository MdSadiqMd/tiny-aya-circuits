"""Pure-function checks for aya_circuits (no model): overlap measures, config overrides, project
root."""

import os
from pathlib import Path

from aya_circuits.config import _coerce, _set, project_root
from aya_circuits.overlap import jaccard, weighted_jaccard


def test_overlap_measures():
    assert jaccard(["a", "b"], ["b", "c"]) == 1 / 3 and jaccard([], []) == 0.0
    sc = {f"A0.{h}": float(h) for h in range(10)} | {
        f"MLP {layer}": -float(layer) for layer in range(5)
    }
    assert abs(weighted_jaccard(sc, sc) - 1) < 1e-9
    assert weighted_jaccard(sc, {}) == 0.0


def test_config_overrides(monkeypatch, tmp_path: Path):
    d = {}
    _set(d, "a.b.c", _coerce("2"))
    _set(d, "a.x", _coerce("[1, 2]"))
    assert d == {"a": {"b": {"c": 2}, "x": [1, 2]}}
    monkeypatch.setenv("CIRCUIT_STEERING_ROOT", str(tmp_path))
    assert project_root() == tmp_path.resolve()
    monkeypatch.delenv("CIRCUIT_STEERING_ROOT")
    assert project_root() == Path(os.getcwd()).resolve()
