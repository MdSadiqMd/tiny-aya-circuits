"""Size-matched random-node control and minimality check for every discovered circuit (brief §12.2,
§12.4), built through the same `_reconstruct_circuit_graph` path as the discovered circuit.

Why not CircuitKIT's Pillar 5: at node level `_build_random_circuit` samples among *all* nodes
including `input`; unless `input` is drawn (10 % chance at our sparsity) no edge leaves the input
and the
random circuit evaluates to exactly the corrupt floor, as every run here showed (random_scores all
−11.4375). Here random circuits keep input/logits like the real circuit and only permute the scored
nodes.

Outputs per circuit (runs/circuits/<task>/seed<k>/random_baseline.json):
  random_node:  5 draws, patching ratio + signed zero-ablation raw ratio for a random 10 % node set
  minimality:   5 draws, patching ratio after dropping 20 % of the circuit's nodes at random
  circuit:      the circuit's own numbers under the identical evaluation call (same n, same seed)
Run: uv run python -m aya_circuits.random_baseline task=b1_en seed=0
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import circuitkit as ck

from aya_circuits.config import load_config
from aya_circuits.faithfulness import evaluate_at
from aya_circuits.memwatch import MemWatch
from aya_circuits.model import load_tl
from aya_circuits.registry import resolve_task


def summary(r: dict) -> dict:
    return {
        "patching": r["patching"],
        "patching_raw_ratio": r["metadata"].get("patching_raw_ratio"),
        "ablation_zero_raw": r["metadata"].get("ablation_raw_ratio"),
        "n_nodes": r["n_nodes_in_circuit"],
    }


def main() -> None:
    cfg = load_config()
    task, seed = cfg["task"], int(cfg["seed"])
    run_dir = Path(cfg["out_dir"]) / "circuits" / task / f"seed{seed}"
    out = run_dir / "random_baseline.json"
    if out.exists() and not cfg.get("force"):
        print("cached", out)
        return
    mw = MemWatch(cfg["memory_budget_gb"])
    model = load_tl(cfg)
    spec = resolve_task(task, cfg)
    scores = ck.load_scores(str(run_dir / "circuit.pt")).scores
    top = cfg["discovery"]["top_fraction"]
    n = cfg["faithfulness"].get("control_n_examples", 64)
    names = list(scores)
    res = {
        "task": task,
        "seed": seed,
        "n_examples": n,
        "circuit": summary(
            evaluate_at(
                model,
                spec,
                scores,
                top,
                cfg,
                seed,
                ["patching", "ablation"],
                "zero",
                1,
                n_examples=n,
            )
        ),
        "random_node": [],
        "minimality": [],
    }
    for draw in range(cfg["random_draws"]):
        rng = random.Random(1000 + draw)
        vals = [scores[k] for k in names]
        rng.shuffle(vals)
        res["random_node"].append(
            summary(
                evaluate_at(
                    model,
                    spec,
                    dict(zip(names, vals, strict=True)),
                    top,
                    cfg,
                    seed,
                    ["patching", "ablation"],
                    "zero",
                    1,
                    n_examples=n,
                )
            )
        )
        mw.check()
        # minimality: zero out 20 % of the circuit's top nodes so they drop out of the top-k
        # selection
        k = max(1, round(len(names) * top))
        topk = sorted(names, key=lambda x: abs(scores[x]), reverse=True)[:k]
        dropped = set(rng.sample(topk, max(1, k // 5)))
        pruned = {x: (0.0 if x in dropped else scores[x]) for x in names}
        res["minimality"].append(
            {
                **summary(
                    evaluate_at(
                        model,
                        spec,
                        pruned,
                        top * 0.8,
                        cfg,
                        seed,
                        ["patching", "ablation"],
                        "zero",
                        1,
                        n_examples=n,
                    )
                ),
                "dropped": sorted(dropped),
            }
        )
        mw.check()
        print(
            f"draw {draw}: random patching={res['random_node'][-1]['patching']:.3f} "
            f"minimality patching={res['minimality'][-1]['patching']:.3f} "
            f"(circuit {res['circuit']['patching']:.3f})",
            flush=True,
        )
    out.write_text(json.dumps(res, indent=1, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
