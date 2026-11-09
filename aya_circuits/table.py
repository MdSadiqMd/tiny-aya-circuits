"""Summarise every runs/circuits/<task>/seed<k>/faithfulness.json into docs/results/circuits.md:
signed patching and ablation raw ratios (zero and mean), stability Jaccard, random/magnitude
baseline scores, circuit size, the size curve for target circuits, and the node/edge consistency
from
discovery manifests. Run: uv run python -m aya_circuits.table"""

from __future__ import annotations

import json
from pathlib import Path

from aya_circuits.config import load_config, project_root

ROOT = project_root()


def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def auc(xs: list[float], ys: list) -> float | None:
    """Normalised trapezoid area of a faithfulness-vs-size curve (Mueller et al. 2025, MIB)."""
    if any(y is None for y in ys) or len(xs) < 2:
        return None
    area = sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1))
    return area / (xs[-1] - xs[0])


def fmt(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, (bool, int)):
        return str(x)
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def main() -> None:
    cfg = load_config()
    root = Path(cfg["out_dir"]) / "circuits"
    rows, curves = [], []
    for fj in sorted(root.glob("*/seed*/faithfulness.json")):
        d = json.loads(fj.read_text())
        z, m = d["full"].get("zero", {}), d["full"].get("mean", {})
        mf = (
            json.loads((fj.parent / "manifest.json").read_text())
            if (fj.parent / "manifest.json").exists()
            else {}
        )
        bz = g(z, "baseline_comparison", "baselines", default={}) or {}
        rb = (
            json.loads((fj.parent / "random_baseline.json").read_text())
            if (fj.parent / "random_baseline.json").exists()
            else {}
        )
        rnd = [r["patching_raw_ratio"] for r in rb.get("random_node", [])]
        rnd_abl = [r["ablation_zero_raw"] for r in rb.get("random_node", [])]
        mini = [r["patching_raw_ratio"] for r in rb.get("minimality", [])]
        rows.append(
            {
                "task": d["task"],
                "seed": d["seed"],
                "nodes": g(z, "n_nodes_in_circuit"),
                "edges": g(z, "n_edges_in_circuit"),
                "patching": g(z, "metadata", "patching_raw_ratio", default=g(z, "patching")),
                "abl_zero_raw": g(z, "metadata", "ablation_raw_ratio"),
                "abl_mean_raw": g(m, "metadata", "ablation_raw_ratio"),
                "stability_jaccard": g(z, "stability", "mean_jaccard"),
                "stability_std": g(z, "stability", "std_jaccard"),
                "ck_random_score": g(bz, "random", "score"),
                "ck_magnitude_score": g(bz, "magnitude", "score"),
                "circuit_score": g(z, "baseline_comparison", "circuit_score"),
                "random_node_patching": (sum(rnd) / len(rnd)) if rnd else None,
                "minimality_patching": (sum(mini) / len(mini)) if mini else None,
                "control_circuit_patching": g(rb, "circuit", "patching_raw_ratio"),
                "random_node_abl_zero_raw": (sum(rnd_abl) / len(rnd_abl)) if rnd_abl else None,
                "control_circuit_abl_zero_raw": g(rb, "circuit", "ablation_zero_raw"),
                "node_edge_rho": mf.get("node_edge_spearman"),
                "wall_s": d.get("wall_s"),
            }
        )
        if d.get("curve"):
            curves.append((d["task"], d["seed"], d["curve"]))
    out = ROOT / "docs/results"
    out.mkdir(parents=True, exist_ok=True)
    cols = [
        "task",
        "seed",
        "nodes",
        "edges",
        "patching",
        "abl_zero_raw",
        "abl_mean_raw",
        "stability_jaccard",
        "circuit_score",
        "ck_random_score",
        "ck_magnitude_score",
        "control_circuit_patching",
        "random_node_patching",
        "minimality_patching",
        "control_circuit_abl_zero_raw",
        "random_node_abl_zero_raw",
        "node_edge_rho",
        "wall_s",
    ]
    lines = [
        "# Circuit faithfulness (CircuitKIT pillars, signed raw ratios; clamped scores not "
        "shown)\n",
        f"{len(rows)} circuit runs. `patching` = Pillar 1 ratio (clamped by CircuitKIT, raw "
        "in "
        "metadata when present); `abl_*_raw` = Pillar 2 unclamped signed ratio under zero / mean "
        "replacement; `ck_*` are CircuitKIT Pillar-5 raw metric values (degenerate at node level: "
        "they equal the corrupt floor because `input` is rarely sampled, docs/notes.md); "
        "`random_node_patching` / `minimality_patching` are our same-path controls "
        "(aya_circuits/random_baseline.py, 5 draws, same n as `control_circuit_patching`); "
        "`random_node_abl_zero_raw` next to `control_circuit_abl_zero_raw` is the zero-ablation "
        "floor: a symmetric logit-difference metric driven to 0 by ablating 90 % of the nodes "
        "sits at ratio 0.5 whatever is kept, so a circuit's zero-ablation ratio is only "
        "informative as a margin over this floor.\n",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    if curves:
        lines += [
            "\n## Faithfulness vs circuit size (fraction of the 612 nodes kept)\n",
            "| task | seed | fraction | patching | abl_zero_raw | abl_mean_raw |",
            "|---|---|---|---|---|---|",
        ]
        for task, seed, curve in curves:
            for frac, v in curve.items():
                lines.append(
                    f"| {task} | {seed} | {frac} | {fmt(g(v, 'zero', 'patching'))} | "
                    f"{fmt(g(v, 'zero', 'metadata', 'ablation_raw_ratio'))} | "
                    f"{fmt(g(v, 'mean', 'metadata', 'ablation_raw_ratio'))} |"
                )
        lines += [
            "\n### Area under the size curve (MIB-style, brief §12.1)\n",
            "Trapezoid over the fractions above, divided by the fraction range, so 1.0 = full "
            "faithfulness at every size and a random-level circuit sits near its floor "
            "(0 for patching, 0.5 for zero ablation on these metrics).\n",
            "| task | seed | AUC patching | AUC abl_zero_raw | AUC abl_mean_raw |",
            "|---|---|---|---|---|",
        ]
        for task, seed, curve in curves:
            fr = sorted(curve, key=float)
            xs = [float(f) for f in fr]
            cols_auc = [
                auc(xs, [g(curve[f], "zero", "patching") for f in fr]),
                auc(xs, [g(curve[f], "zero", "metadata", "ablation_raw_ratio") for f in fr]),
                auc(xs, [g(curve[f], "mean", "metadata", "ablation_raw_ratio") for f in fr]),
            ]
            lines.append(f"| {task} | {seed} | " + " | ".join(fmt(x) for x in cols_auc) + " |")
    cs_rows = []
    span_p = ROOT / "checks" / "metric_span.json"
    span = json.loads(span_p.read_text()) if span_p.exists() else {}
    for cj in sorted(root.glob("*/seed*/cellset.json")):
        c = json.loads(cj.read_text())
        suf, ctl, nec, nctl = (
            c["cell_sufficiency"],
            c["cell_controls"],
            c["necessity"],
            c["necessity_controls"],
        )

        def mean(xs, k):
            return sum(x[k] for x in xs) / len(xs)

        # zero-referenced ratio r' = (r − r0)/(1 − r0) = m/clean, with r0 = −c/(clean − c) the
        # ratio of a model whose metric is driven to 0 (checks/metric_span.json); 1 = intact,
        # 0 = destroyed
        r0 = span.get(c["task"], {}).get("ratio_at_zero_metric")

        def zero_ref(r, r0=r0):
            return None if r0 is None else (r - r0) / (1 - r0)

        cs_rows.append(
            [
                c["task"],
                c["seed"],
                f"{c['cell_set']['n_heads']}+{c['cell_set']['n_mlps']}",
                c["jaccard_pillar_cell"],
                c["coverage_abs_score"]["cell"],
                c["coverage_abs_score"]["pillar"],
                suf["zero"]["patching_raw_ratio"],
                suf["zero"]["ablation_raw_ratio"],
                suf["mean"]["ablation_raw_ratio"],
                ctl["circuit"]["patching_raw_ratio"],
                mean(ctl["random"], "patching_raw_ratio"),
                ctl["circuit"]["ablation_raw_ratio"],
                mean(ctl["random"], "ablation_raw_ratio"),
                nec["cell"]["zero"]["ablation_raw_ratio"],
                nec["cell"]["mean"]["ablation_raw_ratio"],
                nec["pillar"]["zero"]["ablation_raw_ratio"],
                nec["pillar"]["mean"]["ablation_raw_ratio"],
                nctl["circuit_mean"]["ablation_raw_ratio"],
                mean(nctl["random_mean"], "ablation_raw_ratio"),
                nctl["circuit"]["ablation_raw_ratio"],
                mean(nctl["random"], "ablation_raw_ratio"),
                r0,
                zero_ref(nctl["circuit_mean"]["ablation_raw_ratio"]),
                zero_ref(mean(nctl["random_mean"], "ablation_raw_ratio")),
                zero_ref(nec["pillar"]["mean"]["ablation_raw_ratio"]),
            ]
        )
    if cs_rows:
        lines += [
            "\n## Pre-registered circuit (per-type top 10 %: heads and MLPs ranked separately by "
            "|signed row sum|; `aya_circuits/cellset.py`)\n",
            "The table above evaluates CircuitKIT's joint-ranked set; the cells intervene on this "
            "per-type set (`jaccard` and `coverage` compare the two on CircuitKIT's absolute node "
            "scores). `patching` / `abl_*` are the same Pillar 1/2 raw ratios with the set kept "
            "(sufficiency); `ctl_*` and "
            "`random_*` are the set and 5 same-size per-type random sets under the identical call "
            "(`control_n_examples`); `nec_*` ablate the set and keep the rest (necessity: 1 = no "
            "loss, 0.5 = the metric driven to 0 on a symmetric logit difference, ≤ 0 = corrupt "
            "level), for the per-type set (`cell`) and the joint set (`pillar`), with the same "
            "random "
            "control under mean "
            "and zero replacement. `r0` = the ratio of a model whose metric is driven to 0 "
            "(−corrupt/(clean − corrupt), measured in `checks/metric_span.json`); `nec0_*` = the "
            "mean-replacement necessity re-referenced to it, (r − r0)/(1 − r0) = metric/clean: "
            "1 = the set did not matter, 0 = the behaviour is gone, < 0 = inverted.\n",
            "| task | seed | heads+mlps | jaccard | coverage cell | coverage pillar | patching | "
            "abl_zero_raw | abl_mean_raw | ctl_patching | random_patching | ctl_abl_zero | "
            "random_abl_zero | nec_cell_zero | nec_cell_mean | nec_pillar_zero | nec_pillar_mean | "
            "nec_ctl_mean | nec_random_mean | nec_ctl_zero | nec_random_zero | r0 | "
            "nec0_ctl_mean | nec0_random_mean | nec0_pillar_mean |",
            "|" + "---|" * 25,
        ]
        lines += ["| " + " | ".join(fmt(x) for x in r) + " |" for r in cs_rows]
    ov = root / "overlap.json"
    if ov.exists():
        o = json.loads(ov.read_text())
        lines += [
            "\n## Circuit overlap (seed 0): weighted Jaccard on |signed node scores| / "
            "Jaccard on top-10 % node sets\n",
            "| pair | weighted Jaccard | top-10 % Jaccard |",
            "|---|---|---|",
        ]
        for k in o["weighted_jaccard"]:
            lines.append(f"| {k} | {o['weighted_jaccard'][k]:.3f} | {o['top_jaccard'][k]:.3f} |")
        lines += ["\n### Cross-seed top-10 % Jaccard per task\n"] + [
            f"- {t}: {json.dumps({k: round(v, 3) for k, v in d.items()})}"
            for t, d in o["cross_seed_top_jaccard"].items()
        ]
    (out / "circuits.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[: 4 + len(rows)]))


if __name__ == "__main__":
    main()
