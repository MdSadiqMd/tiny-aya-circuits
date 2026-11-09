"""Model loading and the one prompt format shared by discovery, interventions and audit.

TransformerLens path (CircuitKIT discovery, steering, editing): `load_tl` wraps `ck.load_model`
with an explicit device (ck defaults to cpu without CUDA). HF path (LoRA, generation-based audit
layers): `load_hf`. Both read the same checkpoint so weights are identical.

Chat format: Tiny Aya's template always injects a ~366-token system preamble. `chat_wrap` builds
the minimal user-turn format; `chat_format="full"` uses the tokenizer template instead. Decision
and parity check: docs/notes.md 2026-11-02.
"""

from __future__ import annotations

import torch

# Same turn tokens as the official template (user turn; assistant turn wrapped in
# START/END_RESPONSE),
# minus the system turn. Verified against tokenizer_config.json on 2026-11-02.
MINIMAL_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<|START_OF_TURN_TOKEN|>"
    "{% if m['role'] == 'user' %}<|USER_TOKEN|>{{ m['content'] }}"
    "{% else %}<|CHATBOT_TOKEN|><|START_RESPONSE|>{{ m['content'] }}<|END_RESPONSE|>{% endif %}"
    "<|END_OF_TURN_TOKEN|>{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{% endif %}"
)


def cap_mps(budget_gb: float) -> None:
    """Cap the MPS allocator at the granted budget so it evicts cached blocks instead of growing
    (driver-allocated memory reached 33 GB in the fp32 edit stage with ~15 GB live)."""
    if torch.backends.mps.is_available():
        rec = torch.mps.recommended_max_memory()
        torch.mps.set_per_process_memory_fraction(min(2.0, budget_gb * 2**30 / rec))


def is_cohere(name: str) -> bool:
    return "aya" in name.lower() or "cohere" in name.lower() or "command" in name.lower()


def full_template(tokenizer) -> str:
    """The tokenizer's own template as a string (transformers 5 may store a dict of named
    templates)."""
    t = tokenizer.chat_template
    if isinstance(t, dict):
        t = t.get("default") or next(iter(t.values()))
    if isinstance(t, list):  # list of {name, template}
        t = next((x["template"] for x in t if x.get("name") == "default"), t[0]["template"])
    return t


def chat_wrap(
    tokenizer, prompt: str, chat_format: str = "minimal", assistant_prefix: str = ""
) -> str:
    """Return the full text (BOS included) the model sees for one user prompt.

    assistant_prefix is appended at the start of the assistant turn so a single-token answer is the
    immediate next token (CircuitKIT `wrap_prompt` uses the same trick).
    """
    if getattr(tokenizer, "chat_template", None) is None or chat_format == "raw":
        return prompt + assistant_prefix  # base models / gpt2 smoke: raw text
    if chat_format == "minimal":
        tpl = MINIMAL_TEMPLATE
    elif chat_format == "full":
        tpl = full_template(tokenizer)
    else:
        raise ValueError(chat_format)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        chat_template=tpl,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text + assistant_prefix


def set_chat_format(tokenizer, chat_format: str) -> None:
    """Make `tokenizer.apply_chat_template` itself produce the chosen format, so SafeTune /
    AuditKIT / PEFT code that calls the tokenizer's template sees the same prompts we discovered
    on."""
    if chat_format == "minimal" and getattr(tokenizer, "chat_template", None) is not None:
        tokenizer.chat_template = MINIMAL_TEMPLATE


def load_tl(cfg: dict, name: str | None = None):
    """CircuitKIT HookedTransformer with EAP flags, on cfg['device'] in cfg['dtype']."""
    import circuitkit as ck  # (must precede transformer_lens)

    cap_mps(cfg.get("memory_budget_gb", 30))
    model = ck.load_model(name or cfg["model"], dtype=cfg["dtype"], device=cfg["device"])
    set_chat_format(model.tokenizer, cfg.get("chat_format", "minimal"))
    if getattr(model.tokenizer, "chat_template", None) is not None:
        # Every prompt we feed carries its own <BOS_TOKEN>; CircuitKIT's ROME/MEMIT tokenisers read
        # cfg.default_prepend_bos, so a True default would double the BOS and shift every position.
        model.cfg.default_prepend_bos = False
    model.eval()
    return model


def load_hf(cfg: dict, name: str | None = None, dtype: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = name or cfg["model"]
    cap_mps(cfg.get("memory_budget_gb", 30))
    tok = AutoTokenizer.from_pretrained(name)
    set_chat_format(tok, cfg.get("chat_format", "minimal"))
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=getattr(torch, dtype or cfg["dtype"]), attn_implementation="sdpa"
    ).to(cfg["device"])
    model.eval()
    return model, tok


@torch.no_grad()
def generate(
    model, tok, prompts: list[str], max_new_tokens: int = 48, batch_size: int = 8
) -> list[str]:
    """Deterministic batched generation with left padding; prompts are already chat-wrapped."""
    outs: list[str] = []
    tok.padding_side = "left"
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(
            model.device
        )
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
        )
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    return outs
