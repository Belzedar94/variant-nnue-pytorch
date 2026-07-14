"""SFNNv15 fake quantization and export scales."""

from __future__ import annotations

import torch


FT_ONE = 256.0
FT_MAX = 255.0
HIDDEN_ONE = 128.0
HIDDEN_MAX = 127.0
NNUE_TO_SCORE = 600.0
OUTPUT_SCALE = 16.0

FC0_WEIGHT_SCALE = 128.0
FC1_WEIGHT_SCALE = 64.0
FC2_WEIGHT_SCALE = 128.0
FC0_BIAS_SCALE = FC0_WEIGHT_SCALE * HIDDEN_ONE
FC1_BIAS_SCALE = FC1_WEIGHT_SCALE * HIDDEN_ONE
FC2_BIAS_SCALE = FC2_WEIGHT_SCALE * HIDDEN_ONE
PSQT_WEIGHT_SCALE = NNUE_TO_SCORE * OUTPUT_SCALE


def fake_quantize_weight(value: torch.Tensor, scale: float) -> torch.Tensor:
    hard = value.mul(scale).round().div(scale).detach()
    return hard + (value - value.detach())


def fake_quantize_activation(value: torch.Tensor, scale: float = HIDDEN_ONE) -> torch.Tensor:
    # Engine activations truncate; the epsilon protects exact grid values from
    # floating point representation noise.
    hard = value.mul(scale).add(1e-5).floor().div(scale).detach()
    return hard + (value - value.detach())


def clip_feature_activation(value: torch.Tensor) -> torch.Tensor:
    return value.clamp(0.0, FT_MAX / FT_ONE)


def clip_hidden_activation(value: torch.Tensor) -> torch.Tensor:
    return value.clamp(0.0, HIDDEN_MAX / HIDDEN_ONE)


def fake_quantize_output(value: torch.Tensor) -> torch.Tensor:
    multiplier = int(NNUE_TO_SCORE * OUTPUT_SCALE)
    denominator = int(HIDDEN_ONE * FC2_WEIGHT_SCALE * 2.0)
    integer_value = torch.round(value * denominator).to(torch.int64)
    quantized = torch.div(integer_value * multiplier, denominator, rounding_mode="trunc")
    hard = quantized.to(value.dtype) / float(multiplier)
    return hard.detach() + (value - value.detach())


def fake_quantize_psqt_output(value: torch.Tensor) -> torch.Tensor:
    """Match C++ signed integer PSQT division while preserving gradients."""

    # Recover the pre-division integer. Truncating `value * 9600` directly is
    # not stable in float32 (for example, -49 may be represented as -48.999996).
    twice_raw = torch.round(value * (2.0 * PSQT_WEIGHT_SCALE)).to(torch.int64)
    raw = torch.div(twice_raw, 2, rounding_mode="trunc")
    hard = raw.to(value.dtype) / PSQT_WEIGHT_SCALE
    return hard.detach() + (value - value.detach())
