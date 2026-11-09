"""Task name -> registered CircuitKIT TaskSpec. Keeps every script agnostic of where data lives.

b1_<lang>        false-fact / fact-recall prompts (data/b1/eap_<lang>.csv); after filter_model.py
the discovery set is data/b1/eap_known_<lang>.csv (facts the model gets right)
b2_<lang>        refusal task (data/b2/eap_<lang>.csv), built by data/b2_eap.py
ioi              CircuitKIT built-in IOI (neutral control); uses the model tokenizer, raw text
capital_country  CircuitKIT built-in fact-recall bystander (raw text)
langid_<lang>    language-identification bystander (data/bystanders/eap_langid_<lang>.csv)
ioi_fixed        IOI with the metric at the "to" position, one BOS, length-matched single-token
                 pairs (data/ioi_fixed.py, amendment A16); the built-in `ioi` reads its metric at
                 the answer token (checks/check_ioi_position.py)
b2m_<lang>       b2_<lang> restricted to token-length-matched pairs (data/b2_matched.py)
"""

from __future__ import annotations

from aya_circuits.config import project_root
from aya_circuits.tasks import register_csv_task

ROOT = project_root()

BUILTIN = {"ioi", "capital_country"}


def resolve_task(name: str, cfg: dict):
    if name in BUILTIN:
        import circuitkit as ck

        return _BuiltinSpec(ck.get_task(name), name)
    kind, _, lang = name.partition("_")
    paths = {
        "b1": ROOT / "data/b1" / f"eap_known_{lang}.csv",
        "b1pool": ROOT / "data/b1" / f"eap_{lang}.csv",
        "b2": ROOT / "data/b2" / f"eap_{lang}.csv",
        "langid": ROOT / "data/bystanders" / f"eap_langid_{lang}.csv",
        "ioi": ROOT / "data/bystanders" / f"eap_ioi_{lang}.csv",
        "b2m": ROOT / "data/b2" / f"eap_{lang}_matched.csv",
    }
    if kind not in paths:
        raise KeyError(f"unknown task {name!r}")
    return register_csv_task(name, paths[kind])


class _BuiltinSpec:
    """Thin adapter so built-in tasks expose the two attributes our scripts read."""

    def __init__(self, spec, name: str):
        self._spec = spec
        self.name = name
        self.csv_path = ROOT / "configs" / "base.yaml"

    def __getattr__(self, item):
        return getattr(self._spec, item)
