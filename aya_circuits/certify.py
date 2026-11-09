"""Certified circuit for one task and discovery seed (amendment A20, plan item 1).

Rule (sealed in docs/preregistration.md A20): rank the 612 heads+MLPs jointly by CircuitKIT's
absolute node scores of that seed (`circuit.pt`, the ranking of the reported size curve); on a fixed
grid of sizes k = round(612 f), f = 0.10, 0.12, ..., 0.50, evaluate on a held-out fold
  sufficiency  CircuitKIT patching raw ratio of the top-k set (everything else patched from the
               corrupt run; 0 = corrupt, 1 = clean)
  necessity    the set mean-ablated with the rest kept, read as r' = m/cl = (r - r0)/(1 - r0) with
               r0 = -c/(cl - c) from the clean and corrupt metric means on the same examples
               (zero ablation reported at the certified size)
and certify the smallest grid k with sufficiency >= 0.8 and mean-ablation r' <= 0.2. No grid k
passing is a reported failure, not a reason to widen the grid. The curve also gives the area under
sufficiency over f in [0, 0.5] (f = 0 scores 0 by definition), normalised by 0.5.

Held-out fold: rows order[96:96 + n] of the seed's permutation of the task CSV (the permutation
`CsvTaskSpec.build_dataloader` draws; discovery read order[:64], the pillars and size curve
order[:96]), minus any row whose clean prompt also occurs in order[:96] (the CSV has a duplicate).
Every evaluation goes through `faithfulness.evaluate_at` on a 0/1 score mask, and the realised node
set is asserted to be the intended one (`cellset.realised`).

Writes <out_dir>/circuits/<task>/seed<k>/certified.json.
Run: uv run python -m aya_circuits.certify task=b1_en seed=0 [certify.n_heldout=128]
     uv run python -m aya_circuits.certify --check   (unit checks of the rule, no model)
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from pathlib import Path

import circuitkit as ck  # before transformer_lens
import torch
from circuitkit.api import _make_eval_metric
from circuitkit.backends.eap.eap_utils import tokenize_batch_pair

from aya_circuits.cellset import N_NODES, fraction_for, mask, realised, summary
from aya_circuits.config import load_config
from aya_circuits.faithfulness import discovery_cfg, evaluate_at
from aya_circuits.memwatch import MemWatch
from aya_circuits.model import load_tl
from aya_circuits.nodes import split
from aya_circuits.registry import resolve_task
from aya_circuits.tasks import register_csv_task

GRID = [round(0.10 + 0.02 * i, 2) for i in range(21)]  # 0.10 .. 0.50
LOW = [0.01, 0.02, 0.05]  # extra curve points below the grid (area only, never certified)
SUFFICIENCY, NECESSITY = 0.8, 0.2
SEEN = 96  # rows order[:SEEN] were read by discovery (64) and the pillars / size curve (96)


def heldout_rows(csv_path: str, seed: int, n: int) -> list[dict]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed)).tolist()
    seen = {rows[i]["clean"] for i in order[:SEEN]}
    out = [rows[i] for i in order[SEEN:] if rows[i]["clean"] not in seen][:n]
    if len(out) < n:
        raise SystemExit(f"only {len(out)} held-out rows for seed {seed}, wanted {n}")
    return out


def heldout_task(task: str, seed: int, n: int, cfg: dict):
    """Register the held-out fold as its own CSV task (CircuitKIT's dataloader reads only it)."""
    src = resolve_task(task, cfg)
    rows = heldout_rows(src.csv_path, seed, n)
    p = Path(cfg["out_dir"]) / "circuits" / task / f"seed{seed}" / "certify_heldout.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clean", "corrupted", "correct_idx", "incorrect_idx"])
        w.writeheader()
        w.writerows({k: r[k] for k in w.fieldnames} for r in rows)
    return register_csv_task(f"{task}_cert_s{seed}", p)


@torch.no_grad()
def metric_means(model, spec, cfg: dict, seed: int, n: int) -> tuple[float, float]:
    """Clean and corrupt metric means on exactly the examples the pillars read (same dataloader,
    tokenisation and metric as CircuitKIT; checks/check_metric_span.py)."""
    dl = spec.build_dataloader(model, discovery_cfg(cfg, spec, seed, "zero", n), cfg["device"])
    mf = _make_eval_metric(spec)
    side, templated = getattr(dl, "pair_padding_side", None), getattr(dl, "templated", False)
    cs, ks = [], []
    for clean, corrupt, labels in dl:
        tc, tk, mc, mk, lens, _ = tokenize_batch_pair(
            model, clean, corrupt, pair_padding_side=side, templated=templated
        )
        lc, lk = model(tc, attention_mask=mc), model(tk, attention_mask=mk)
        cs.append(mf(lc, lc, lens, labels).float().cpu())
        ks.append(mf(lk, lc, lens, labels).float().cpu())
    return float(torch.cat(cs).mean()), float(torch.cat(ks).mean())


def zero_referenced(r: float | None, r0: float) -> float | None:
    return None if r is None else (r - r0) / (1 - r0)


def area(curve: dict[float, float]) -> float:
    """Trapezoidal area under sufficiency over f in [0, 0.5], with (0, 0), divided by 0.5."""
    pts = sorted({0.0: 0.0, **curve}.items())
    return sum((f2 - f1) * (s1 + s2) / 2 for (f1, s1), (f2, s2) in itertools.pairwise(pts)) / 0.5


def certify(curve: list[dict]) -> dict | None:
    """The smallest grid point passing both criteria (curve rows sorted by k)."""
    for row in curve:
        if (
            row["on_grid"]
            and row["sufficiency"] >= SUFFICIENCY
            and row["necessity_mean"] <= NECESSITY
        ):
            return row
    return None


def main() -> None:
    cfg = load_config()
    task, seed = cfg["task"], int(cfg["seed"])
    n = int(cfg.get("certify", {}).get("n_heldout", 128))
    run_dir = Path(cfg["out_dir"]) / "circuits" / task / f"seed{seed}"
    out = run_dir / "certified.json"
    if out.exists() and not cfg.get("force"):
        print("cached", out)
        return
    t0, mw = time.time(), MemWatch(cfg["memory_budget_gb"])
    model = load_tl(cfg)
    abs_scores = ck.load_scores(str(run_dir / "circuit.pt")).scores
    all_nodes = list(abs_scores)
    assert len(all_nodes) == N_NODES and "input" not in abs_scores
    ranking = sorted(all_nodes, key=lambda x: abs(abs_scores[x]), reverse=True)
    spec = heldout_task(task, seed, n, cfg)
    cl, c = metric_means(model, spec, cfg, seed, n)
    r0 = -c / (cl - c)
    print(f"{task} s{seed}: held-out n={n} clean {cl:.3f} corrupt {c:.3f} r0 {r0:.3f}", flush=True)

    def ev(keep: list[str], pillar: str, interv: str) -> dict:
        # a 0/1 mask at (k + 0.5)/612 keeps the top int(k + 0.5) = k nodes, i.e. exactly the marked
        # ones when CircuitKIT's graph holds k nodes besides `input` (asserted); the certified set
        # is also checked node by node with `realised` below
        r = summary(
            evaluate_at(
                model,
                spec,
                mask(all_nodes, set(keep)),
                fraction_for(len(keep)),
                cfg,
                seed,
                [pillar],
                interv,
                1,
                n_examples=n,
            )
        )
        assert r["n_nodes"] == len(keep) + 1, (r["n_nodes"], len(keep))  # + the input node
        return r

    curve = []
    for f in sorted(LOW + GRID):
        k = round(N_NODES * f)
        keep, rest = ranking[:k], ranking[k:]
        row = {
            "fraction": f,
            "k": k,
            "on_grid": f in GRID,
            "n_heads": len(split(keep)[0]),
            "n_mlps": len(split(keep)[1]),
            "sufficiency": ev(keep, "patching", "zero")["patching_raw_ratio"],
        }
        if row["on_grid"]:  # necessity only where the rule reads it; zero ablation at the end
            raw = ev(rest, "ablation", "mean")["ablation_raw_ratio"]
            row |= {"necessity_mean_raw": raw, "necessity_mean": zero_referenced(raw, r0)}
        curve.append(row)
        mw.check()
        print(json.dumps(row), flush=True)
    best = certify(curve)
    res = {
        "task": task,
        "seed": seed,
        "rule": f"smallest grid k: patching >= {SUFFICIENCY}, mean-ablation r' <= {NECESSITY}",
        "ranking": "CircuitKIT absolute node scores (circuit.pt), joint over heads and MLPs",
        "heldout": {
            "n": n,
            "csv": str(spec.csv_path),
            "clean_mean": cl,
            "corrupt_mean": c,
            "r0": r0,
        },
        "curve": curve,
        "area_sufficiency_0_0.5": area({r["fraction"]: r["sufficiency"] for r in curve}),
        "certified": None,
        "wall_s": None,
    }
    if best is not None:
        nodes = ranking[: best["k"]]
        got = realised(
            model, spec, mask(all_nodes, set(nodes)), fraction_for(len(nodes)), cfg, seed
        )
        assert got == set(nodes), (sorted(got - set(nodes)), sorted(set(nodes) - got))
        raw0 = ev(ranking[best["k"] :], "ablation", "zero")["ablation_raw_ratio"]
        res["certified"] = {
            **best,
            "necessity_zero_raw": raw0,
            "necessity_zero": zero_referenced(raw0, r0),
            "nodes": nodes,
        }
    res["wall_s"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}: certified k={best and best['k']} in {res['wall_s']}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        assert area({0.5: 1.0}) == 0.5 and abs(area({0.25: 1.0, 0.5: 1.0}) - 0.75) < 1e-12
        rows = [
            {"on_grid": True, "sufficiency": 0.7, "necessity_mean": 0.0, "k": 61},
            {"on_grid": False, "sufficiency": 0.9, "necessity_mean": 0.0, "k": 70},
            {"on_grid": True, "sufficiency": 0.85, "necessity_mean": 0.3, "k": 73},
            {"on_grid": True, "sufficiency": 0.81, "necessity_mean": 0.1, "k": 86},
        ]
        assert certify(rows)["k"] == 86 and zero_referenced(0.509, 0.509) == 0.0
        print("certify ok")
    else:
        main()
