"""Port of the fork's test_resident_moe.py: resident vs non-resident gpt-oss MoE.

Verifies ``moe_mlp_gpt_oss_resident`` (weights uploaded once, reused by
handle) produces output identical to ``moe_mlp_gpt_oss`` (weights uploaded
per call) on real gpt-oss layer-0 weights, then frees the handles
(fork gate: maxdiff < 1e-4).

Opt-in via FREETOKEN_TEST_MODEL_GPT_OSS (gpt-oss checkpoint directory);
skips otherwise. The extension comes from the package's _runtime helper
instead of the fork's in-process torch cpp build.
"""
from __future__ import annotations

import json

import torch

from freetoken.models.qwen3_5_moe.vulkan._runtime import get_ext


def test_resident_moe_matches_nonresident(gptoss_model_dir):
    ext = get_ext()
    model_dir = gptoss_model_dir

    with open(model_dir / "model.safetensors.index.json") as fh:
        index = json.load(fh)["weight_map"]

    def get(name: str) -> torch.Tensor:
        from safetensors import safe_open

        with safe_open(model_dir / index[name], framework="pt", device="cpu") as fh:
            return fh.get_tensor(name)

    # Layer-0 MoE weights (gpt-oss MXFP4 layout).
    gu_blocks = get("model.layers.0.mlp.experts.gate_up_proj_blocks").contiguous()
    gu_scales = get("model.layers.0.mlp.experts.gate_up_proj_scales").contiguous()
    gu_bias = get("model.layers.0.mlp.experts.gate_up_proj_bias").float().contiguous()
    d_blocks = get("model.layers.0.mlp.experts.down_proj_blocks").contiguous()
    d_scales = get("model.layers.0.mlp.experts.down_proj_scales").contiguous()
    d_bias = get("model.layers.0.mlp.experts.down_proj_bias").float().contiguous()

    e, two_dff = gu_blocks.shape[0], gu_blocks.shape[1]
    dff = two_dff // 2
    d = 2880
    k = 4

    torch.manual_seed(0xF00D)
    t = 8
    x = (torch.randn(t, d) * 0.05).float().contiguous()
    indices = torch.randint(0, e, (t, k)).contiguous()
    weights = torch.softmax(torch.randn(t, k), dim=-1).contiguous()

    # Pass 1: non-resident - upload weights every call.
    y_nonres = ext.moe_mlp_gpt_oss(
        x, indices, weights, gu_blocks, gu_scales, gu_bias,
        d_blocks, d_scales, d_bias,
    )

    # Pass 2: resident - upload once, reuse by handle.
    baseline = ext.resident_bytes_total()
    handles = [
        ext.upload_resident(gu_blocks),
        ext.upload_resident(gu_scales),
        ext.upload_resident(gu_bias),
        ext.upload_resident(d_blocks),
        ext.upload_resident(d_scales),
        ext.upload_resident(d_bias),
    ]
    try:
        y_res = ext.moe_mlp_gpt_oss_resident(
            x, indices, weights, *handles, e, d, dff,
        )
    finally:
        for handle in handles:
            ext.free_resident(handle)

    diff = (y_res - y_nonres).abs().max().item()
    print(f"resident-vs-nonresident maxdiff {diff:.3e}", flush=True)
    assert torch.isfinite(y_res).all()
    assert diff < 1e-4, f"resident MoE maxdiff {diff:.3e}"
    assert ext.resident_bytes_total() == baseline, "resident pool not restored"
