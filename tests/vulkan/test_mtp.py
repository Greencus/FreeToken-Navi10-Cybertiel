"""MTP (nextn) speculative decoding gates.

The gate that matters: the speculative path must emit the SAME greedy
tokens as the non-speculative ``generate`` path on a fixed prompt set. That
is what the llama.cpp reference promises (draft + verify + exact-match
accept) and what a wrong port would break first - a wrong ``pending_h``
shift, a MTP KV cell fed with the wrong hidden row, or a verify row read
from the wrong position all show up as token divergence long before they
show up as garbage.

Gates:

- ``test_mtp_cli_guards`` - ``--mtp`` refuses a non-GGUF run and a
  non-greedy target during argument parsing (no checkpoint, no device).
- ``test_accept_greedy_rule`` - the accept rule itself (no checkpoint
  needed): stop at the first mismatch, one bonus sample on a full match.
- ``test_prefill_with_h_matches_prefill_chunked`` - the MTP prefill runs the
  production ``prefill_chunked``; the extra hidden-row capture must not
  perturb it (bit-identical last-row logits).
- ``test_mtp_identity`` - 3 prompts x ``MAX_NEW`` greedy tokens, MTP output
  identical to ``generate``. Prints the accept rate so a slow-but-correct
  run still reports whether drafting was useful.

Model-dependent gates skip without ``FREETOKEN_TEST_MODEL_GGUF``.
"""
from __future__ import annotations

import os

import pytest
import torch

from freetoken.models.qwen3_5_moe.vulkan.mtp import accept_greedy, generate_mtp

# Prompts encoded with the GGUF's own tokenizer
# (``load_gguf_tokenizer``), no special tokens; kept as raw ids so this gate
# needs no transformers install.
PROMPTS = [
    ("short factual", [760, 6511, 314, 9338, 369]),
    ("short instruction", [7734, 264, 6185, 36974, 883, 34343, 13]),
    (
        "multi-sentence",
        [14917, 557, 264, 1248, 1834, 13, 353, 14428, 2046, 13569, 321, 1801, 14799, 13],
    ),
]
# At ~30 s/token on the current host-dequant path keep the gate short by
# default; raise with FREETOKEN_TEST_MTP_TOKENS for a longer run.
MAX_NEW = int(os.environ.get("FREETOKEN_TEST_MTP_TOKENS", "6"))
TRUNK_LAYERS = 40


def _snapshot(model):
    return {
        "gdn": {i: s.clone() for i, s in model.gdn_state.items()},
        "conv": {i: c.clone() for i, c in model.conv_buf.items()},
        "k": {i: k.clone() for i, k in model.k_cache.items()},
        "v": {i: v.clone() for i, v in model.v_cache.items()},
        "seq_len": model._seq_len,
    }


def _restore(model, snap):
    for i, s in snap["gdn"].items():
        model.gdn_state[i].copy_(s)
    for i, c in snap["conv"].items():
        model.conv_buf[i].copy_(c)
    for i, k in snap["k"].items():
        model.k_cache[i].copy_(k)
    for i, v in snap["v"].items():
        model.v_cache[i].copy_(v)
    model._seq_len = snap["seq_len"]
    model._gdn_gpu_dirty = True


def test_mtp_cli_guards(monkeypatch):
    """--mtp rejects its two unusable combinations before loading weights.

    Both checks run during argument parsing, so this needs no checkpoint and
    no device - which is the point: the guard must not cost a 30 s pin to
    discover it cannot run.
    """
    import sys

    from freetoken.models.qwen3_5_moe.vulkan import cli

    def _run(argv):
        monkeypatch.setattr(sys, "argv", ["cli", *argv])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        return str(exc.value)

    assert "pass --gguf" in _run(["--mtp", "--max-tokens", "1"])
    assert "greedy-only" in _run(
        ["--mtp", "--gguf", "/nonexistent/x.gguf", "--temperature", "0.7"]
    )


def test_accept_greedy_rule():
    """accept_greedy matches common_sampler_sample_and_accept_n (greedy)."""
    rows = torch.tensor(
        [
            [5.0, 1.0, 0.0],  # argmax 0
            [0.0, 4.0, 1.0],  # argmax 1
            [1.0, 0.0, 3.0],  # argmax 2
            [9.0, 1.0, 0.0],  # argmax 0 (bonus row)
        ]
    )
    assert accept_greedy(rows, []) == [0]
    assert accept_greedy(rows, [0]) == [0, 1]
    assert accept_greedy(rows, [0, 1]) == [0, 1, 2]
    # all drafts matched -> the bonus row is sampled too
    assert accept_greedy(rows, [0, 1, 2]) == [0, 1, 2, 0]
    # mismatch on the first draft -> one token, no bonus
    assert accept_greedy(rows, [1]) == [0]
    # mismatch later -> accepted prefix + the rejecting token
    assert accept_greedy(rows, [0, 0]) == [0, 1]


def test_prefill_with_h_matches_prefill_chunked(gguf_model):
    """The h-row capture leaves the production prefill bit-identical."""
    from freetoken.models.qwen3_5_moe.vulkan.mtp import prefill_with_h

    ids = PROMPTS[0][1]
    model = gguf_model
    model.reset()
    # Snapshot the pristine post-reset state: the GDN state and conv history
    # are cumulative, so a snapshot taken after run 1 would make run 2 replay
    # the prompt on top of run 1's recurrence.
    snap = _snapshot(model)
    logits_a = model.prefill_chunked(ids, chunk_size=64).clone()
    _restore(model, snap)
    logits_b, h_rows = prefill_with_h(model, ids, chunk_size=64)
    assert torch.equal(logits_a, logits_b), (
        f"prefill_with_h drifted from prefill_chunked: "
        f"{(logits_a - logits_b).abs().max().item():.3e}"
    )
    assert h_rows.shape == (len(ids), model.config.hidden_size)
    assert torch.isfinite(h_rows).all()


def test_mtp_walk_parity(gguf_model):
    """The batched MTP walk == chained decode_step_batched (bit-exact),
    and lands within the fork2 1e-5 bar of the sequential decode_step.

    Every arm starts from the SAME pristine snapshot, restored at the top of
    the arm - the GDN recurrence and the KV caches accumulate, so restoring
    between rows inside an arm would desynchronize the chain (the harness
    gotcha that produced the false "divergence at layer 1"). All arms see
    the same embedding rows at the same positions ``pos..pos+T-1``.
    """
    from freetoken.models.qwen3_5_moe.vulkan.mtp import enable_mtp

    model = gguf_model
    head = enable_mtp(model)
    if not head._batched:
        pytest.skip(
            "walk parity needs the batched MTP lane "
            "(FTV_MTP_BATCHED=1 with the batched decode contract)"
        )
    t_rows = 4
    model.reset()
    logits = model.prefill_chunked(PROMPTS[0][1], chunk_size=64)
    pos = model._seq_len
    row_ids: list[int] = []
    for _ in range(t_rows):
        tid = int(logits.argmax().item())
        row_ids.append(tid)
        logits = model.decode_step(model.embed_row(tid), pos)
        pos += 1
        model._seq_len = pos
    snap = _snapshot(model)
    x_rows = torch.stack([model.embed_row(t) for t in row_ids])

    _restore(model, snap)
    w1_logits, w1_h = head.verify(pos, row_ids)
    _restore(model, snap)
    w2_logits, w2_h = head.verify(pos, row_ids)
    _restore(model, snap)
    bat_logits = torch.stack(
        [model.decode_step_batched(x_rows[t], pos + t)
         for t in range(t_rows)]
    )
    _restore(model, snap)
    seq_logits = torch.stack(
        [model.decode_step(x_rows[t], pos + t) for t in range(t_rows)]
    )

    d_walk = (w1_logits - w2_logits).abs().max().item()
    d_walk_h = (w1_h - w2_h).abs().max().item()
    assert torch.equal(w1_logits, w2_logits) and torch.equal(w1_h, w2_h), (
        f"walk not bit-exact across restores: logits {d_walk:.3e} "
        f"hidden {d_walk_h:.3e}"
    )
    d_bat = (w1_logits - bat_logits).abs().max().item()
    # Expected 0.0: both arms run the same recorders row-for-row; only the
    # lm cross differs (one T-row batch_bf16 call vs T single-row ones) and
    # the shader is row-parallel in T.
    assert torch.equal(w1_logits, bat_logits), (
        f"walk vs chained decode_step_batched drifted {d_bat:.3e} "
        "(if small, the lm cross is the suspect: move lm into the walk)"
    )
    d_seq = (w1_logits - seq_logits).abs().max().item()
    print(
        f"[walk parity] walk-walk {d_walk:.3e} (0), "
        f"walk-batched {d_bat:.3e} (0), "
        f"walk-sequential {d_seq:.3e} (<= 1e-5)",
        flush=True,
    )
    assert d_seq <= 1e-5, (
        f"walk vs sequential decode_step maxdiff {d_seq:.3e} > 1e-5 "
        "(fork2 bar: lane rounding between the fused and host paths)"
    )


def test_mtp_identity(gguf_model):
    """MTP output tokens == the non-speculative greedy tokens."""
    model = gguf_model
    for name, ids in PROMPTS:
        baseline = model.generate(ids, max_new=MAX_NEW, temperature=0.0)
        speculative = generate_mtp(
            model, ids, max_new=MAX_NEW, n_max=3, p_min=0.0, prefill_chunk=64
        )
        stats = dict(model.mtp.stats)
        drafted, accepted = stats["drafted"], stats["accepted"]
        rate = accepted / drafted if drafted else 0.0
        # Full-model passes per committed token, counting a draft step and a
        # catch-up block as 1/TRUNK_LAYERS of a pass (one layer each).
        # Denominator = tokens the executed rounds produced (1 prefill sample
        # + accepted + rounds): the final round may over-deliver past max_new,
        # and counting only max_new would overstate the cost per token.
        rounds, rollbacks = stats["rounds"], stats["rollbacks"]
        passes = rounds + rollbacks + (drafted + rounds) / TRUNK_LAYERS
        produced = 1 + accepted + rounds
        per_tok = passes / max(produced, 1)
        print(
            f"[{name}] tokens {len(speculative)} drafted {drafted:.0f} "
            f"accepted {accepted:.0f} rate {rate:.2f} "
            f"rounds {rounds:.0f} rollbacks {rollbacks:.0f} "
            f"passes/token {per_tok:.3f} (baseline 1.000, "
            f"{(1 - per_tok) * 100:+.1f}% full passes)",
            flush=True,
        )
        print(
            f"[{name}] cost ms: verify {stats['ms_verify']:.0f} "
            f"draft {stats['ms_draft']:.0f} catchup {stats['ms_catchup']:.0f} "
            f"rollback {stats['ms_rollback']:.0f} prefill {stats['ms_prefill']:.0f} "
            f"snapshot {stats.get('ms_snapshot', 0):.0f}",
            flush=True,
        )
        print(f"[{name}] ids {baseline}", flush=True)
        assert speculative == baseline, (
            f"{name}: MTP diverged from the greedy baseline at "
            f"first mismatch {next(i for i, (a, b) in enumerate(zip(speculative, baseline)) if a != b) if speculative != baseline else '-'}\n"
            f"  baseline     {baseline}\n"
            f"  speculative  {speculative}"
        )
        assert drafted >= accepted >= 0


def test_mtp_skip_kv_snapshot_identity(gguf_model, monkeypatch):
    """FTV_MTP_SKIP_KV_SNAPSHOT=1 (KV-eliding rollback) stays token-identical.

    The eliding snapshot skips the ~190 MB trunk-KV host copies on the
    theory that every KV cell is written before it is read, so the copies
    only preserve unread garbage. That theory is only as good as the tokens:
    this gate runs one short identity with the flag on and asserts the same
    baseline.
    """
    monkeypatch.setenv("FTV_MTP_SKIP_KV_SNAPSHOT", "1")
    model = gguf_model
    # enable_mtp is idempotent on an existing head; drop it so the flag is
    # re-read in the constructor.
    model.mtp = None
    ids = PROMPTS[0][1]
    baseline = model.generate(ids, max_new=MAX_NEW, temperature=0.0)
    speculative = generate_mtp(
        model, ids, max_new=MAX_NEW, n_max=3, p_min=0.0, prefill_chunk=64
    )
    assert model.mtp._skip_kv_snap
    assert speculative == baseline, (
        f"skip-KV-snapshot MTP diverged: {speculative} != {baseline}"
    )
    model.mtp = None  # later tests get the default (full-copy) head back
