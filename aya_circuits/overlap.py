"""Overlap matrix between every discovered circuit (H4 input): weighted Jaccard on |signed node
scores| and plain Jaccard on top-10 % node sets, for seed 0 of every task, plus cross-seed Jaccard
per task
(stability of the node set). Writes runs/circuits/overlap.json."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

from aya_circuits.config import load_config


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def weighted_jaccard(sa: dict[str, float], sb: dict[str, float]) -> float:
    """Weighted Jaccard on |scores| (REdit's overlap measure): sum min / sum max over the union."""
    keys = set(sa) | set(sb)
    num = sum(min(abs(sa.get(k, 0.0)), abs(sb.get(k, 0.0))) for k in keys)
    den = sum(max(abs(sa.get(k, 0.0)), abs(sb.get(k, 0.0))) for k in keys)
    return num / den if den else 0.0


def load(run_root: Path, task: str, seed: int, top_fraction: float):
    n = json.loads((run_root / "circuits" / task / f"seed{seed}" / "nodes.json").read_text())
    sig = dict(zip(n["names"], n["signed_rowsum"], strict=True))
    sig.pop("input", None)
    k = max(1, round(len(sig) * top_fraction))
    return sig, sorted(sig, key=lambda x: abs(sig[x]), reverse=True)[:k]


def main() -> None:
    cfg = load_config()
    root, top = Path(cfg["out_dir"]), cfg["discovery"]["top_fraction"]
    tasks = sorted(
        p.name for p in (root / "circuits").iterdir() if (p / "seed0" / "nodes.json").exists()
    )
    circ = {t: load(root, t, 0, top) for t in tasks}
    out = {"tasks": tasks, "weighted_jaccard": {}, "top_jaccard": {}, "cross_seed_top_jaccard": {}}
    for a, b in combinations(tasks, 2):
        out["weighted_jaccard"][f"{a}|{b}"] = weighted_jaccard(circ[a][0], circ[b][0])
        out["top_jaccard"][f"{a}|{b}"] = jaccard(circ[a][1], circ[b][1])
    for t in tasks:
        seeds = [
            int(p.name[4:])
            for p in (root / "circuits" / t).glob("seed*")
            if (p / "nodes.json").exists()
        ]
        tops = {s: load(root, t, s, top)[1] for s in seeds}
        if len(seeds) > 1:
            out["cross_seed_top_jaccard"][t] = {
                f"{a}|{b}": jaccard(tops[a], tops[b]) for a, b in combinations(sorted(seeds), 2)
            }
    (root / "circuits" / "overlap.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
