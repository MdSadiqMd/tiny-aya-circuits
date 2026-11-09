"""Faithfulness evaluation for one discovered circuit (brief §5, §12.1): CircuitKIT pillars
patching, ablation (zero AND mean, signed raw ratios), baselines (5 random draws + magnitude),
stability (3 re-discoveries), plus the faithfulness-vs-size curve at 1,2,5,10,20,30 % under both
ablation types. Everything runs through `run_full_faithfulness` on a graph rebuilt by CircuitKIT's
own `_reconstruct_circuit_graph`, so the numbers are the toolkit's, not ours.

Usage: uv run python -m aya_circuits.faithfulness task=b1_en seed=0
Reads runs/circuits/<task>/seed<seed>/circuit.pt (+ _scores.json); writes faithfulness.json there.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import circuitkit as ck  # before transformer_lens
from circuitkit.api import _make_eval_metric, _qkv_flags_enabled, _reconstruct_circuit_graph
from circuitkit.evaluation.full import run_full_faithfulness

from aya_circuits.config import load_config
from aya_circuits.manifest import Manifest
from aya_circuits.memwatch import MemWatch
from aya_circuits.model import load_tl
from aya_circuits.registry import resolve_task


def report_to_dict(rep) -> dict:
    d = {"patching": rep.patching_score, "ablation": rep.ablation_score, "metadata": rep.metadata}
    for k in ("stability", "baseline_comparison", "robustness", "generalization"):
        v = getattr(rep, k, None)
        if v is not None:
            d[k] = v
    return json.loads(json.dumps(d, default=lambda o: getattr(o, "tolist", lambda: str(o))()))


def discovery_cfg(cfg: dict, spec, seed: int, intervention: str, n_examples: int | None) -> dict:
    f = cfg["faithfulness"]
    return {
        "algorithm": cfg["discovery"]["algorithm"],
        "task": spec.name,
        "level": "node",
        "batch_size": f["batch_size"],
        "data_params": {"num_examples": n_examples or f["n_examples"], "seed": seed},
        "seed": seed,
        "eval_intervention": intervention,
        "ig_steps": cfg["discovery"]["ig_steps"],
        "cache_dir": cfg["cache_dir"],
    }


def evaluate_at(
    model,
    spec,
    scores: dict,
    fraction: float,
    cfg: dict,
    seed: int,
    pillars: list[str],
    intervention: str,
    n_stability: int,
    n_examples: int | None = None,
):
    dcfg = discovery_cfg(cfg, spec, seed, intervention, n_examples)
    pruning_cfg = {"target_sparsity": 1 - fraction, "scope": "both"}
    with _qkv_flags_enabled(model):
        graph = _reconstruct_circuit_graph(
            model, {"node_scores": scores}, dcfg, pruning_cfg, cfg["device"]
        )
        dl = spec.build_dataloader(model, dcfg, cfg["device"])
        rep = run_full_faithfulness(
            model=model,
            graph=graph,
            task_spec=spec,
            discovery_cfg=dcfg,
            device=cfg["device"],
            pillars=pillars,
            n_stability_runs=n_stability,
            metric_fn=_make_eval_metric(spec),
            dataloader=dl,
            intervention_dataloader=dl if intervention != "zero" else None,
            pruning_cfg=pruning_cfg,
        )
    out = report_to_dict(rep)
    out["n_nodes_in_circuit"] = int(graph.nodes_in_graph.sum().item())
    out["n_edges_in_circuit"] = int(graph.count_included_edges())
    return out


def main() -> None:
    cfg = load_config()
    task_name, seed = cfg["task"], int(cfg["seed"])
    run_dir = Path(cfg["out_dir"]) / "circuits" / task_name / f"seed{seed}"
    out = run_dir / "faithfulness.json"
    top = cfg["discovery"]["top_fraction"]
    # curve_tasks: null -> every task gets the size curve and stability (brief §12.1, §7.4)
    full_tasks = cfg["faithfulness"].get("curve_tasks") or [task_name]
    want_curve = task_name in full_tasks
    prev = json.loads(out.read_text()) if out.exists() and not cfg.get("force") else None
    if prev is not None:
        # fill mode: an existing file keeps every number it has; only absent pieces are computed
        missing_stab = want_curve and "stability" not in prev["full"].get("zero", {})
        missing_frac = [
            f for f in cfg["faithfulness"]["curve_fractions"] if str(f) not in prev["curve"]
        ]
        if not missing_stab and not (want_curve and missing_frac):
            print("cached", out)
            return
    mf, mw = Manifest("faithfulness", cfg, seed), MemWatch(cfg["memory_budget_gb"])
    model = load_tl(cfg)
    spec = resolve_task(task_name, cfg)
    circuit = ck.load_scores(str(run_dir / "circuit.pt")) if hasattr(ck, "load_scores") else None
    scores = (
        circuit.scores
        if circuit is not None
        else json.loads((run_dir / "nodes.json").read_text())["abs_scores"]
    )
    mf.add_input(run_dir / "circuit.pt")
    result = prev or {"task": task_name, "seed": seed, "top_fraction": top, "full": {}, "curve": {}}
    t0 = time.time()
    if prev is not None and want_curve and "stability" not in prev["full"].get("zero", {}):
        stab = evaluate_at(
            model,
            spec,
            scores,
            top,
            cfg,
            seed,
            ["stability"],
            "zero",
            cfg["faithfulness"]["n_stability_runs"],
        )
        result["full"]["zero"]["stability"] = stab.get("stability")
        result.setdefault("filled", []).append("full.zero.stability")
        mw.check()
        print(f"[zero] stability={stab.get('stability')}", flush=True)
    for interv in ("zero", "mean"):
        if interv in result["full"]:
            continue
        pillars = ["patching", "ablation", "baselines"] + (
            ["stability"] if interv == "zero" and want_curve else []
        )
        result["full"][interv] = evaluate_at(
            model,
            spec,
            scores,
            top,
            cfg,
            seed,
            pillars,
            interv,
            cfg["faithfulness"]["n_stability_runs"],
        )
        mw.check()
        print(
            f"[{interv}] patching={result['full'][interv]['patching']} "
            f"ablation_raw={result['full'][interv]['metadata'].get('ablation_raw_ratio')}",
            flush=True,
        )
    for frac in cfg["faithfulness"]["curve_fractions"] if want_curve else []:
        if str(frac) in result["curve"]:
            continue
        if prev is not None:
            result.setdefault("filled", []).append(f"curve.{frac}")
        result["curve"][str(frac)] = {
            interv: evaluate_at(
                model,
                spec,
                scores,
                frac,
                cfg,
                seed,
                ["patching", "ablation"],
                interv,
                1,
                n_examples=cfg["faithfulness"].get("curve_n_examples"),
            )
            for interv in ("zero", "mean")
        }
        mw.check()
        print(
            f"curve {frac}: "
            + " ".join(
                f"{i}:P={v['patching']} A={v['metadata'].get('ablation_raw_ratio')}"
                for i, v in result["curve"][str(frac)].items()
            ),
            flush=True,
        )
    result["fill_wall_s" if prev is not None else "wall_s"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(result, indent=1, default=str))
    mf.add_output(out)
    suffix = "_fill" if prev is not None else ""
    mf.write(run_dir / f"manifest_faithfulness{suffix}.json", memory=mw.summary())
    print("wrote", out)


if __name__ == "__main__":
    main()
