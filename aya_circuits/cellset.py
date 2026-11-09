"""Faithfulness on the pre-registered circuit definition, plus necessity, for one task and seed.

Two node sets share the word "circuit" (docs/audit-2026-11-05.md B1): CircuitKIT's pillars rank the
612 nodes jointly by its absolute node scores (the *pillar set*, `faithfulness.py`), while the cells
intervene on the pre-registered per-type set (top 10 % heads and top 10 % MLPs by |signed row sum|,
`aya_circuits.nodes.circuit_nodes`, the *cell set*). This module evaluates the cell set through the
same `evaluate_at` path (CircuitKIT `_reconstruct_circuit_graph` + `run_full_faithfulness`) by
handing it a
0/1 score mask, and asserts that the realised node set is exactly the intended one.

  sufficiency   patching and zero/mean ablation raw ratios with the set kept (Pillars 1 and 2)
  necessity     zero/mean ablation raw ratio with the set ablated and everything else kept, for both
                sets: ratio 1 = the set did not matter, 0 = the model falls to the corrupt baseline
  controls      5 per-type random sets of the same size through sufficiency (zero) and necessity
                (zero and mean) at `control_n_examples`, with the real set under the identical call

Writes runs/circuits/<task>/seed<k>/cellset.json.
Run: uv run python -m aya_circuits.cellset task=b1_en seed=0
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import circuitkit as ck  # before transformer_lens
from circuitkit.api import _qkv_flags_enabled, _reconstruct_circuit_graph

from aya_circuits.config import load_config
from aya_circuits.discover import node_names
from aya_circuits.faithfulness import discovery_cfg, evaluate_at
from aya_circuits.memwatch import MemWatch
from aya_circuits.model import load_tl
from aya_circuits.nodes import circuit_nodes, sizes, split
from aya_circuits.overlap import jaccard
from aya_circuits.registry import resolve_task

N_NODES = 612


def mask(all_nodes: list[str], keep: set[str]) -> dict[str, float]:
    return {n: (1.0 if n in keep else 0.0) for n in all_nodes}


def fraction_for(n_keep: int) -> float:
    # _compute_n_topn keeps int(612 * fraction) nodes; the half keeps float error from dropping one
    return (n_keep + 0.5) / N_NODES


def realised(model, spec, scores: dict[str, float], fraction: float, cfg: dict, seed: int) -> set:
    """Node set CircuitKIT actually keeps for `scores` at `fraction` (no forward passes)."""
    dcfg = discovery_cfg(cfg, spec, seed, "zero", None)
    with _qkv_flags_enabled(model):
        graph = _reconstruct_circuit_graph(
            model,
            {"node_scores": scores},
            dcfg,
            {"target_sparsity": 1 - fraction, "scope": "both"},
            cfg["device"],
        )
    names = node_names(graph)
    return {n for n, keep in zip(names, graph.nodes_in_graph.tolist(), strict=True) if keep} - {
        "input"
    }


def summary(r: dict) -> dict:
    return {
        "patching_raw_ratio": r["metadata"].get("patching_raw_ratio"),
        "ablation_raw_ratio": r["metadata"].get("ablation_raw_ratio"),
        "n_nodes": r["n_nodes_in_circuit"],
        "n_edges": r["n_edges_in_circuit"],
    }


def main() -> None:
    cfg = load_config()
    task, seed = cfg["task"], int(cfg["seed"])
    run_dir = Path(cfg["out_dir"]) / "circuits" / task / f"seed{seed}"
    out = run_dir / "cellset.json"
    if out.exists() and not cfg.get("force"):
        print("cached", out)
        return
    t0 = time.time()
    mw = MemWatch(cfg["memory_budget_gb"])
    model = load_tl(cfg)
    spec = resolve_task(task, cfg)
    top = cfg["discovery"]["top_fraction"]
    n_ctrl = cfg["faithfulness"].get("control_n_examples", 64)
    abs_scores = ck.load_scores(str(run_dir / "circuit.pt")).scores
    nodes = json.loads((run_dir / "nodes.json").read_text())
    signed = dict(zip(nodes["names"], nodes["signed_rowsum"], strict=True))
    signed.pop("input", None)
    all_nodes = list(abs_scores)
    assert set(all_nodes) == set(signed) and len(all_nodes) == N_NODES

    cell = circuit_nodes(signed, top)
    pillar = sorted(realised(model, spec, abs_scores, top, cfg, seed))
    got = realised(model, spec, mask(all_nodes, set(cell)), fraction_for(len(cell)), cfg, seed)
    assert got == set(cell), (sorted(got - set(cell)), sorted(set(cell) - got))
    tot_abs = sum(abs(v) for v in abs_scores.values())
    tot_signed = sum(abs(v) for v in signed.values())

    def ev(keep: list[str], pillars: list[str], interv: str, n: int | None = None) -> dict:
        return summary(
            evaluate_at(
                model,
                spec,
                mask(all_nodes, set(keep)),
                fraction_for(len(keep)),
                cfg,
                seed,
                pillars,
                interv,
                1,
                n_examples=n,
            )
        )

    def complement(keep: list[str]) -> list[str]:
        return [n for n in all_nodes if n not in set(keep)]

    kh, km = sizes(signed, top)
    heads, mlps = split(all_nodes)
    draws = [
        random.Random(1000 + d).sample(heads, kh) + random.Random(2000 + d).sample(mlps, km)
        for d in range(cfg["random_draws"])
    ]
    res = {
        "task": task,
        "seed": seed,
        "top_fraction": top,
        "n_examples": cfg["faithfulness"]["n_examples"],
        "control_n_examples": n_ctrl,
        "cell_set": {"nodes": cell, "n_heads": kh, "n_mlps": km},
        "pillar_set": {"nodes": pillar, "n_mlps": len(split(pillar)[1])},
        "jaccard_pillar_cell": jaccard(pillar, cell),
        "coverage_abs_score": {
            "cell": sum(abs(abs_scores[n]) for n in cell) / tot_abs,
            "pillar": sum(abs(abs_scores[n]) for n in pillar) / tot_abs,
        },
        "coverage_signed_rowsum": {
            "cell": sum(abs(signed[n]) for n in cell) / tot_signed,
            "pillar": sum(abs(signed[n]) for n in pillar) / tot_signed,
        },
    }
    res["cell_sufficiency"] = {i: ev(cell, ["patching", "ablation"], i) for i in ("zero", "mean")}
    mw.check()
    print(f"cell set sufficiency: {res['cell_sufficiency']}", flush=True)
    res["cell_controls"] = {
        "circuit": ev(cell, ["patching", "ablation"], "zero", n_ctrl),
        "random": [ev(d, ["patching", "ablation"], "zero", n_ctrl) for d in draws],
    }
    mw.check()
    res["necessity"] = {
        "cell": {i: ev(complement(cell), ["ablation"], i) for i in ("zero", "mean")},
        "pillar": {i: ev(complement(pillar), ["ablation"], i) for i in ("zero", "mean")},
    }
    print(f"necessity: {res['necessity']}", flush=True)
    res["necessity_controls"] = {
        "circuit": ev(complement(cell), ["ablation"], "zero", n_ctrl),
        "random": [ev(complement(d), ["ablation"], "zero", n_ctrl) for d in draws],
        "circuit_mean": ev(complement(cell), ["ablation"], "mean", n_ctrl),
        "random_mean": [ev(complement(d), ["ablation"], "mean", n_ctrl) for d in draws],
    }
    res["random_draws"] = draws
    res["wall_s"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(res, indent=1, default=str))
    print("wrote", out, f"{res['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
