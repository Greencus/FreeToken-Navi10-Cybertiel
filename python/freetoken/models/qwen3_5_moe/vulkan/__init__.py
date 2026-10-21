"""Qwen3.5-MoE inference on the FreeToken Vulkan backend (port of the fork).

Consolidates the fork's ``qwen35`` (TorchAO-affine CPU reference + hybrid
GPU/CPU ``ResidentQwen35`` decoder) and ``qwen35_nvfp4`` (Ressl ModelOpt
NVFP4 loader + ``ResidentNVFP4`` decoder) packages, plus the runtime
resolver (``_runtime``) and the top-level ``engine``/``cli`` entry points.

The native extension is never imported at package import time; call
``get_ext()`` (or construct a decoder, which resolves it lazily) to reach
the Vulkan device.
"""
from ._runtime import ext_or_none, get_ext, runtime_info, shaders_dir
from .config import Qwen35MoeConfig, load_qwen35_config
from .loader import (
    AffineQuantMatrix,
    Qwen35MoeLayerWeights,
    Qwen35MoeModelWeights,
    ShardedSafetensors,
    dequantize_affine,
    load_qwen35_model,
    load_qwen35_moe_layer,
    unpack_affine_u32,
)
from .layer import decoder_layer_forward
from .linear_attn import gated_delta_forward, l2norm
from .moe import moe_forward, moe_router
from .attention import (
    apply_partial_rotary,
    compute_text_rope,
    full_attn_forward,
    rms_norm_head,
)
from .resident import ResidentQwen35
from .nvfp4 import MODEL_DIR, PREFIX, dequantize_nvfp4, pack_scales_u32, unpack_e2m1
from .nvfp4_resident import ResidentNVFP4
from .gguf_provider import ExpertBank, GgufWeightProvider, gguf_config
from .gguf_resident import ResidentGGUF
from .hotset_resident import HotsetResidentGGUF
from .engine import EOS_IDS, generate, is_nvfp4_checkpoint, load_engine

__all__ = [
    # config / loader
    "Qwen35MoeConfig",
    "load_qwen35_config",
    "AffineQuantMatrix",
    "Qwen35MoeLayerWeights",
    "Qwen35MoeModelWeights",
    "ShardedSafetensors",
    "dequantize_affine",
    "load_qwen35_model",
    "load_qwen35_moe_layer",
    "unpack_affine_u32",
    # reference forwards
    "decoder_layer_forward",
    "gated_delta_forward",
    "l2norm",
    "moe_forward",
    "moe_router",
    "apply_partial_rotary",
    "compute_text_rope",
    "full_attn_forward",
    "rms_norm_head",
    # resident decoders
    "ResidentQwen35",
    "ResidentNVFP4",
    "ResidentGGUF",
    "HotsetResidentGGUF",
    # GGUF weight source (CyberTiel host-dequant path)
    "ExpertBank",
    "GgufWeightProvider",
    "gguf_config",
    # NVFP4 helpers
    "MODEL_DIR",
    "PREFIX",
    "dequantize_nvfp4",
    "pack_scales_u32",
    "unpack_e2m1",
    # engine API
    "EOS_IDS",
    "generate",
    "is_nvfp4_checkpoint",
    "load_engine",
    # runtime resolution
    "ext_or_none",
    "get_ext",
    "runtime_info",
    "shaders_dir",
]
