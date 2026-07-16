"""AtomicNNUEV3 mixed feature and byte-identical SFNNv15 quantization."""

from __future__ import annotations

import math
from functools import lru_cache

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
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


@lru_cache(maxsize=None)
def inward_float32_i32_bounds(scale: float) -> tuple[float, float]:
    """Return float32 parameter bounds that remain i32-safe when exported.

    ``INT32_MAX / scale`` commonly rounds outward when stored as float32.  Move
    each endpoint toward the interval until multiplication and rounding in
    either float32 or float64 stays in the signed-i32 domain.
    """

    if (
        not isinstance(scale, (int, float))
        or isinstance(scale, bool)
        or not math.isfinite(float(scale))
        or scale <= 0
    ):
        raise ValueError("i32 export scale must be positive")
    lower = torch.tensor(INT32_MIN / float(scale), dtype=torch.float32)
    upper = torch.tensor(INT32_MAX / float(scale), dtype=torch.float32)

    def exportable(value: torch.Tensor) -> bool:
        integers = (
            torch.round(value * value.new_tensor(scale)).item(),
            torch.round(value.to(torch.float64) * float(scale)).item(),
        )
        return all(INT32_MIN <= integer <= INT32_MAX for integer in integers)

    positive = lower.new_tensor(float("inf"))
    negative = lower.new_tensor(float("-inf"))
    while not exportable(lower):
        lower = torch.nextafter(lower, positive)
    while not exportable(upper):
        upper = torch.nextafter(upper, negative)
    return float(lower.item()), float(upper.item())


FC0_BIAS_EXPORT_MIN, FC0_BIAS_EXPORT_MAX = inward_float32_i32_bounds(FC0_BIAS_SCALE)
FC1_BIAS_EXPORT_MIN, FC1_BIAS_EXPORT_MAX = inward_float32_i32_bounds(FC1_BIAS_SCALE)
FC2_BIAS_EXPORT_MIN, FC2_BIAS_EXPORT_MAX = inward_float32_i32_bounds(FC2_BIAS_SCALE)
# Float32 cannot represent ``INT32_MAX / 9600`` without rounding out of the
# signed-i32 domain.  These are the adjacent inward float32 values, used at the
# persistent parameter boundary so a later exporter remains safe whether it
# multiplies in float32 or float64.
PSQT_EXPORT_MIN = -223696.203125
PSQT_EXPORT_MAX = 223696.203125


def fake_quantize_integer(
    value: torch.Tensor, *, scale: float, minimum: int, maximum: int
) -> torch.Tensor:
    if scale <= 0:
        raise ValueError("quantization scale must be positive")
    if minimum >= maximum:
        raise ValueError("quantization integer interval is empty")
    # Promote the integer-domain clamp: near i32 limits a float32 clamp bound
    # would itself round INT32_MAX to 2**31.
    hard = (
        value.to(torch.float64)
        .mul(scale)
        .round()
        .clamp(minimum, maximum)
        .div(scale)
        .to(value.dtype)
        .detach()
    )
    return hard + (value - value.detach())


def fake_quantize_i16_feature(value: torch.Tensor) -> torch.Tensor:
    return fake_quantize_integer(value, scale=FT_ONE, minimum=-32768, maximum=32767)


def fake_quantize_i8_feature(value: torch.Tensor) -> torch.Tensor:
    return fake_quantize_integer(value, scale=FT_ONE, minimum=-128, maximum=127)


def fake_quantize_psqt_weight(value: torch.Tensor) -> torch.Tensor:
    exportable = value.clamp(PSQT_EXPORT_MIN, PSQT_EXPORT_MAX)
    return fake_quantize_integer(
        exportable, scale=PSQT_WEIGHT_SCALE, minimum=-(1 << 31), maximum=(1 << 31) - 1
    )


def fake_quantize_weight(value: torch.Tensor, scale: float) -> torch.Tensor:
    return fake_quantize_integer(value, scale=scale, minimum=-127, maximum=127)


def fake_quantize_activation(value: torch.Tensor, scale: float = HIDDEN_ONE) -> torch.Tensor:
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
    twice_raw = torch.round(value * (2.0 * PSQT_WEIGHT_SCALE)).to(torch.int64)
    raw = torch.div(twice_raw, 2, rounding_mode="trunc")
    hard = raw.to(value.dtype) / PSQT_WEIGHT_SCALE
    return hard.detach() + (value - value.detach())
