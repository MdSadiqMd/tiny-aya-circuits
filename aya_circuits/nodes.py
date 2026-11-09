"""The pre-registered circuit definition (docs/preregistration.md: "top 10 % of the 612 heads+MLPs
by |score|, heads and MLPs ranked separately so every condition is size-matched per component type")
and
the per-type helpers every selection rule shares. Node names follow CircuitKIT: 'A{layer}.{head}',
'MLP {layer}'; 'input' is never a candidate."""

from __future__ import annotations

MLP = "MLP "


def split(nodes: list[str]) -> tuple[list[str], list[str]]:
    return [n for n in nodes if not n.startswith(MLP)], [n for n in nodes if n.startswith(MLP)]


def sizes(scores: dict[str, float], top_fraction: float) -> tuple[int, int]:
    heads, mlps = split([n for n in scores if n != "input"])
    return max(1, round(len(heads) * top_fraction)), max(1, round(len(mlps) * top_fraction))


def top_k(scores: dict[str, float], k_heads: int, k_mlps: int, reverse: bool = True) -> list[str]:
    heads, mlps = split([n for n in scores if n != "input"])
    heads.sort(key=lambda n: abs(scores[n]), reverse=reverse)
    mlps.sort(key=lambda n: abs(scores[n]), reverse=reverse)
    return heads[:k_heads] + mlps[:k_mlps]


def circuit_nodes(signed_scores: dict[str, float], top_fraction: float) -> list[str]:
    """The intervention circuit of one discovered task: per-type top fraction by |signed row
    sum|."""
    return top_k(signed_scores, *sizes(signed_scores, top_fraction))
