"""CSV-backed CircuitKIT TaskSpec for every behaviour and bystander task in this project.

CircuitKIT's `Pipeline.from_custom_data` re-derives labels and alignment itself; we need control
over (a) the chat format (one format everywhere, docs/notes.md), (b) token-length-matched
clean/corrupt pairs, and (c) the exact label ids, so we implement the TaskSpec protocol
(`circuitkit/tasks/specs.py`) directly and register it with `ck.register_task`. The dataloader is
CircuitKIT's own `EAPDiscoveryDataset`, so discovery, faithfulness pillars and stability
re-discovery all run unchanged on it.

CSV columns: clean, corrupted, correct_idx, incorrect_idx  (already chat-wrapped text, BOS
included).
"""

from __future__ import annotations

import csv
from functools import partial
from pathlib import Path

import torch as t
from circuitkit.data.eap_dataset import EAPDiscoveryDataset
from circuitkit.tasks.registry import _TASKS, register_task


def logit_diff(logits, clean_logits, input_length, labels, mean=True, loss=False):
    """logit(correct) - logit(incorrect) at the last real position; loss=True negates
    (discovery)."""
    idx = t.arange(logits.size(0), device=logits.device)
    last = logits[idx, input_length - 1]
    good_bad = t.gather(last, -1, labels.to(logits.device))
    res = good_bad[:, 0] - good_bad[:, 1]
    if loss:
        res = -res
    return res.mean() if mean else res


class CsvTaskSpec:
    """One registered task per CSV. `templated=True` tells the EAP tokenizer the text carries its
    own BOS (CircuitKIT `tokenize_plus(templated=...)`)."""

    pair_padding_side = "right"
    chat_template_mode = "off"  # text is pre-wrapped; CircuitKIT must not wrap again

    def __init__(self, name: str, csv_path: str | Path, templated: bool = True):
        self.name = name
        self.csv_path = str(csv_path)
        self.templated = templated
        with open(self.csv_path) as f:
            self.n_rows = sum(1 for _ in f) - 1

    def validate_discovery_config(self, discovery_cfg: dict) -> None:
        if discovery_cfg.get("level", "node") not in ("node", "neuron"):
            raise ValueError("level must be node or neuron")

    def build_dataloader(self, model, discovery_cfg: dict, device: str):
        params = discovery_cfg.get("data_params", {})
        n = int(params.get("num_examples", self.n_rows))
        seed = int(params.get("seed", discovery_cfg.get("seed", 42)))
        bs = int(discovery_cfg.get("batch_size", 4))
        # Seeded row subset: different discovery seeds see different examples (stability pillar).
        with open(self.csv_path) as f:
            rows = list(csv.DictReader(f))
        g = t.Generator().manual_seed(seed)
        order = t.randperm(len(rows), generator=g).tolist()[: min(n, len(rows))]
        sub = (
            Path(discovery_cfg.get("cache_dir", ".cache"))
            / "task_subsets"
            / f"{self.name}_{n}_s{seed}.csv"
        )
        sub.parent.mkdir(parents=True, exist_ok=True)
        with sub.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["clean", "corrupted", "correct_idx", "incorrect_idx"])
            w.writeheader()
            for i in order:
                w.writerow({k: rows[i][k] for k in w.fieldnames})
        return EAPDiscoveryDataset(str(sub), templated=self.templated).to_dataloader(
            batch_size=bs, pair_padding_side=self.pair_padding_side
        )

    def metric_fn(self, metric_type: str = "logit_diff"):
        return partial(logit_diff, loss=True, mean=True)

    def artifact_metadata(self, discovery_cfg: dict) -> dict:
        return {
            "task": self.name,
            "csv": self.csv_path,
            "n_rows": self.n_rows,
            "templated": self.templated,
        }


def register_csv_task(name: str, csv_path: str | Path, templated: bool = True) -> CsvTaskSpec:
    """Register (or replace) a CSV task; CircuitKIT refuses duplicate names, so we pop first."""
    _TASKS.pop(name.lower(), None)
    spec = CsvTaskSpec(name, csv_path, templated)
    register_task(spec)
    return spec
