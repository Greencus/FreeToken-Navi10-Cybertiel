"""Hot-set residency gates for the Vulkan GGUF decoder (model-dependent).

Opt-in via FREETOKEN_TEST_MODEL_GGUF (the CyberTiel GGUF; mmap only). One
``HotsetResidentGGUF`` is built per session (construction ~1 min: dense
pins + hot-set upload) and shared by the tests, which run in this order:

1. pin state: experts/layer, entry count, budget ledger;
2. greedy token identity: "Name three primary colors." rendered through
   the CHAT TEMPLATE exactly like cli.py's default (no --raw-prompt)
   must produce the known-good ids [5844, 11, 13358, 11]
   ("Red, yellow,") on BOTH the sequential and the batched decode path -
   the same ids the engine without the hot set produces (argmax
   identity; the in-shader unpack is fp32 and the host dequant BF16, so
   bit-equality is NOT the bar);
3. eviction stress: flood the LRU with non-hot experts and assert no
   pinned key ever left ``_exp_cache``.

The hot set is small here (K=4, 512 MiB cap) so the gate is fast; the
production defaults (K=24, 4096 MiB) are exercised by
tests/vulkan/bench_gguf_hotset.py.
"""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

torch.set_num_threads(8)

# Greedy reference for "Name three primary colors." --max-tokens 4 at
# temperature 0 on the packed CyberTiel path (cli.py defaults: chat
# template ON, temperature 0). The RAW text encodes to just
# [657, 2250, 5839, 7736, 13] and greedily continues [271, 248068, ...]
# on BOTH the hot-set and the plain path - the reference ids only match
# when the prompt is rendered the way the CLI renders it.
REFERENCE_PROMPT = "Name three primary colors."
REFERENCE_IDS = [5844, 11, 13358, 11]

HOTSET_ENV = {
    "FTV_GGUF_HOTSET": "1",
    "FTV_GGUF_HOTSET_MB": "512",
    "FTV_GGUF_HOTSET_K": "4",
    "FTV_GGUF_HOTSET_KEYS": "both",
    "FTV_GGUF_PACKED": "1",
    "FTV_EXP_CACHE_MB": "768",
}


def _identity_prompt_ids(tokenizer) -> list[int]:
    """cli.py build_prompt: chat template + generation prompt (no thinking)."""
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": REFERENCE_PROMPT}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
    )
    return [int(i) for i in out["input_ids"]]


@pytest.fixture(scope="session")
def hotset_model():
    gguf = os.environ.get("FREETOKEN_TEST_MODEL_GGUF", "")
    if not gguf or not os.path.isfile(gguf):
        pytest.skip("set FREETOKEN_TEST_MODEL_GGUF to the CyberTiel GGUF")
    pytest.importorskip("freetoken.models.qwen3_5_moe.vulkan")
    saved = {k: os.environ.get(k) for k in HOTSET_ENV}
    os.environ.update(HOTSET_ENV)
    try:
        from freetoken.models.qwen3_5_moe.vulkan.hotset_resident import (
            HotsetResidentGGUF,
        )

        model = HotsetResidentGGUF(gguf, None, max_len=2048)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if not model.hotset_info()["enabled"]:
        pytest.skip("hot set did not pin (missing table?)")
    yield model


def test_pin_state(hotset_model):
    info = hotset_model.hotset_info()
    assert info["layers_pinned"] == 40
    assert info["per_layer"] == [4] * 40
    # both key styles: 3 per-matrix keys + 1 fused gate+up key per expert
    assert info["pinned_experts"] == 4 * 40
    assert info["pinned_entries"] == 4 * 40 * 4
    mib = 2**20
    assert 0 < info["pinned_bytes"] <= info["pin_budget"] <= 768 * mib
    assert info["lru_region"] >= 0
    # every pinned key is resident right after init
    cache = hotset_model._exp_cache
    assert all(k in cache for k in hotset_model._hotset_keys)


def test_greedy_identity(hotset_model):
    from freetoken.models.gguf.tokenizer import load_gguf_tokenizer

    tok = load_gguf_tokenizer(
        os.environ["FREETOKEN_TEST_MODEL_GGUF"])
    ids = _identity_prompt_ids(tok)
    out = hotset_model.generate(ids, max_new=4, temperature=0.0, seed=1234)
    assert out == REFERENCE_IDS


def test_greedy_identity_batched(hotset_model):
    """Same reference through the batched decode path (FTV_GGUF_BATCHED=1):
    this is the path that consumes the pinned FUSED gate+up bundles."""
    from freetoken.models.gguf.tokenizer import load_gguf_tokenizer

    if not getattr(hotset_model, "_gguf_batched", False):
        pytest.skip("batched decode disabled")
    tok = load_gguf_tokenizer(
        os.environ["FREETOKEN_TEST_MODEL_GGUF"])
    ids = _identity_prompt_ids(tok)
    out = hotset_model.generate(
        ids, max_new=4, temperature=0.0, seed=1234, batched=True)
    assert out == REFERENCE_IDS


def test_eviction_never_touches_pinned(hotset_model):
    import itertools

    model = hotset_model
    pinned = set(model._hotset_keys)
    assert pinned
    gate_mats = [
        model.expert_mats[f"language_model.model.layers.{l}.mlp.switch_mlp.gate_proj"]
        for l in range(0, 40, 8)
    ]
    # ~700 non-hot per-matrix misses >> the ~288 MiB LRU region
    # (768 MiB budget - ~480 MiB pinned), forcing evictions
    cold = itertools.count(128)  # hot table holds the top ranks; 128+ is cold
    for _ in range(700):
        e = next(cold) % 250
        model._expert_handle(gate_mats[e % len(gate_mats)], e)
    evicts = model.stat_calls.get("cache_evict", 0)
    assert evicts > 0, "stress produced no evictions - LRU region too big?"
    missing = [k for k in pinned if k not in model._exp_cache]
    assert not missing, f"pinned keys evicted under stress: {missing[:5]}"
