# tiny-aya-circuits

EAP-IG circuits on `CohereLabs/tiny-aya-global` built on Lexsi's CircuitKIT 0.2.0, with the controls a
circuit needs before anyone intervenes on it: signed node and edge scores, the faithfulness pillars under
zero and mean ablation (reported signed, never clamped), stability across seeds, a size curve, and
same-path random node sets evaluated under the same patching and ablation (the random margin is what
separates a localised circuit from a floor effect). Package `aya_circuits`; it also carries the model,
config and manifest layer that the other two repos of the study share
([circuit-steering](../circuit-steering), [nothing-else-broke](../nothing-else-broke)).

Modules (`python -m aya_circuits.<name> key.sub=value ...`, run from a study checkout that holds
`configs/base.yaml`, `data/` and `runs/`, or set `CIRCUIT_STEERING_ROOT`):

| module | does |
|---|---|
| `discover` | EAP-IG node + signed edge attribution for one task and seed → `runs/circuits/<task>/seed<k>/` |
| `faithfulness` | CircuitKIT pillars (patching, zero/mean ablation, stability, size curve) with raw ratios |
| `random_baseline` | size-matched random node sets through the same patching and ablation path |
| `overlap` | weighted Jaccard between every circuit, cross-seed Jaccard per task |
| `table` | `docs/results/circuits.md` from every faithfulness and control file |
| `registry`, `tasks` | task name → CircuitKIT `TaskSpec` (CSV tasks in `data/`, built-in IOI) |
| `model`, `config`, `manifest`, `memwatch`, `hub` | shared loading (TransformerLens and HF from one checkpoint, one chat format), YAML config with CLI overrides, run manifests, memory ceiling, Hub helpers |

```bash
uv sync && uv run pytest -q
```

Circuit discovery rules used by the study: `eap-ig` only (`acdc` is impractically slow on this model,
`ibcircuit` at aggressive settings OOMs above 3B), batch 2 for the long-prompt tasks.
