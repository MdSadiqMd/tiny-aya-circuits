"""Circuit discovery for one task and seed: CircuitKIT EAP-IG node discovery (official artifact +
`_scores.json`) plus a signed edge-level EAP-IG pass on the same data, saved as safetensors.

Why two passes: the artifact's node scores are absolute values (docs/notes.md), while cross-circuit
CAD (audit/l4) needs signed edge scores s_e. The edge pass reuses
`circuitkit.backends.eap.attribute`
on the same dataloader and metric. Row-sums of the signed edge matrix must match the signed node
scores up to IG noise; `check_node_edge_consistency` asserts that (Spearman > 0.9), which is also a
sanity check that both passes saw the same data.

Usage: uv run python -m aya_circuits.discover task=b1_en seed=0 [discovery.n_examples=64 ...]
Task names resolve through circuits/registry.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import circuitkit as ck  # before transformer_lens
import torch
from circuitkit.backends.eap.attribute import attribute
from circuitkit.backends.eap.graph import Graph
from safetensors.torch import save_file
from scipy.stats import spearmanr

from aya_circuits.config import load_config
from aya_circuits.manifest import Manifest
from aya_circuits.memwatch import MemWatch
from aya_circuits.model import load_tl
from aya_circuits.registry import resolve_task


def node_names(graph: Graph) -> list[str]:
    """CircuitKIT names ('A{l}.{h}', 'MLP {l}') in forward-index order (index 0 is 'input')."""
    names = ["input"]
    for layer in range(graph.cfg["n_layers"]):
        names += [f"A{layer}.{h}" for h in range(graph.cfg["n_heads"])]
        names.append(f"MLP {layer}")
    return names


def signed_edge_scores(
    model, task_spec, discovery_cfg: dict, device: str, ig_steps: int
) -> tuple[Graph, torch.Tensor]:
    """Edge-level EAP-IG (signed), same dataloader/metric as discovery."""
    graph = Graph.from_model(model, node_scores=True)
    dl = task_spec.build_dataloader(model, discovery_cfg, device)
    attribute(model, graph, dl, task_spec.metric_fn(), method="EAP-IG-inputs", ig_steps=ig_steps)
    return graph, graph.scores.detach().float().cpu()


def check_node_edge_consistency(
    edge_scores: torch.Tensor, graph: Graph, node_scores_abs: dict[str, float]
) -> float:
    """Spearman between |row-sum of signed edges| and the artifact's absolute node scores."""
    names = node_names(graph)
    rowsum = edge_scores.sum(dim=1).abs()
    xs, ys = [], []
    for i, n in enumerate(names):
        if n in node_scores_abs:
            xs.append(rowsum[i].item())
            ys.append(node_scores_abs[n])
    return float(spearmanr(xs, ys).statistic)


def main() -> None:
    cfg = load_config()
    task_name, seed = cfg["task"], int(cfg["seed"])
    d = cfg["discovery"]
    out_dir = Path(cfg["out_dir"]) / "circuits" / task_name / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "edges.safetensors").exists() and not cfg.get("force"):
        print("cached", out_dir)
        return
    mf, mw = Manifest("discover", cfg, seed), MemWatch(cfg["memory_budget_gb"])
    model = load_tl(cfg)
    spec = resolve_task(task_name, cfg)
    mf.add_input(spec.csv_path)

    t0 = time.time()
    circuit = ck.discover(
        model,
        spec.name,
        algorithm=d["algorithm"],
        level=d["level"],
        n_examples=d["n_examples"],
        batch_size=d["batch_size"],
        sparsity=1 - d["top_fraction"],
        scope=d["scope"],
        output_path=str(out_dir / "circuit.pt"),
        seed=seed,
        ig_steps=d["ig_steps"],
        cache_dir=str(Path(cfg["cache_dir"])),
    )
    t_node = time.time() - t0
    mw.check()

    t0 = time.time()
    discovery_cfg = {
        "algorithm": d["algorithm"],
        "task": spec.name,
        "level": "node",
        "batch_size": d["batch_size"],
        "data_params": {"num_examples": d["n_examples"], "seed": seed},
        "seed": seed,
        "cache_dir": cfg["cache_dir"],
    }
    graph, edges = signed_edge_scores(model, spec, discovery_cfg, cfg["device"], d["ig_steps"])
    t_edge = time.time() - t0
    rho = check_node_edge_consistency(edges, graph, circuit.scores)
    save_file(
        {"edges": edges, "node_rowsum": edges.sum(dim=1)},
        str(out_dir / "edges.safetensors"),
        metadata={
            "task": spec.name,
            "seed": str(seed),
            "n_forward": str(graph.n_forward),
            "n_backward": str(graph.n_backward),
        },
    )
    (out_dir / "nodes.json").write_text(
        json.dumps(
            {
                "names": node_names(graph),
                "abs_scores": circuit.scores,
                "signed_rowsum": edges.sum(dim=1).tolist(),
            },
            indent=1,
        )
    )
    for p in (out_dir / "edges.safetensors", out_dir / "nodes.json", out_dir / "circuit.pt"):
        mf.add_output(p)
    mf.write(
        out_dir / "manifest.json",
        memory=mw.summary(),
        timing={"node_pass_s": round(t_node, 1), "edge_pass_s": round(t_edge, 1)},
        node_edge_spearman=rho,
        n_scored=len(circuit.scores),
    )
    print(
        f"{task_name} seed{seed}: node {t_node:.0f}s edge {t_edge:.0f}s "
        f"rho(node,edge)={rho:.3f} peak {mw.peak:.1f}GB"
    )
    if rho < 0.9:
        print("WARNING: node/edge scores disagree; inspect before using", file=sys.stderr)


if __name__ == "__main__":
    main()
