"""Private byte-identical SFNNv15 dense tail for AtomicNNUEV3."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .contract import LAYER_STACKS
from .quantization import (
    FC0_BIAS_SCALE,
    FC0_WEIGHT_SCALE,
    FC1_BIAS_SCALE,
    FC1_WEIGHT_SCALE,
    FC2_BIAS_SCALE,
    FC2_WEIGHT_SCALE,
    INT32_MAX,
    INT32_MIN,
    clip_hidden_activation,
    fake_quantize_activation,
    fake_quantize_integer,
    fake_quantize_output,
    fake_quantize_weight,
    inward_float32_i32_bounds,
)


class StackedLinear(nn.Module):
    def __init__(
        self, inputs: int, outputs: int, count: int, weight_scale: float, bias_scale: float
    ):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.count = count
        self.weight_scale = weight_scale
        self.bias_scale = bias_scale
        self.linear = nn.Linear(inputs, outputs * count)
        self._repeat_first_bucket()

    @torch.no_grad()
    def _repeat_first_bucket(self) -> None:
        weight = self.linear.weight[: self.outputs]
        bias = self.linear.bias[: self.outputs]
        self.linear.weight.copy_(weight.repeat(self.count, 1))
        self.linear.bias.copy_(bias.repeat(self.count))

    def _merged_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.linear.weight, self.linear.bias

    def forward(
        self,
        value: torch.Tensor,
        bucket_indices: torch.Tensor,
        fake_quantize_weights: bool,
        *,
        validate_bucket_range: bool = True,
    ) -> torch.Tensor:
        if value.ndim != 2 or bucket_indices.shape != (value.shape[0],):
            raise ValueError("stacked linear requires [batch, inputs] and [batch] buckets")
        if bucket_indices.dtype != torch.long:
            raise TypeError("layer-stack indices must use torch.long")
        if validate_bucket_range and torch.any(
            (bucket_indices < 0) | (bucket_indices >= self.count)
        ):
            raise ValueError("layer-stack index outside [0, 7]")
        weight, bias = self._merged_parameters()
        if fake_quantize_weights:
            weight = fake_quantize_weight(weight, self.weight_scale)
            bias = fake_quantize_integer_bias(bias, self.bias_scale)
        broad = F.linear(value, weight, bias)
        rows = broad.reshape(-1, self.outputs)
        offsets = torch.arange(
            0, bucket_indices.numel() * self.count, self.count, device=broad.device
        )
        return rows[bucket_indices.reshape(-1) + offsets]


def fake_quantize_integer_bias(value: torch.Tensor, scale: float) -> torch.Tensor:
    # Dense biases are signed i32.  Clamp to an inward float32 boundary before
    # quantization: INT32_MAX / scale can otherwise round outward merely by
    # storing the quantized parameter back in float32.
    minimum, maximum = inward_float32_i32_bounds(scale)
    exportable = value.clamp(minimum, maximum)
    return fake_quantize_integer(
        exportable, scale=scale, minimum=INT32_MIN, maximum=INT32_MAX
    )


class FactorizedStackedLinear(StackedLinear):
    def __init__(
        self, inputs: int, outputs: int, count: int, weight_scale: float, bias_scale: float
    ):
        super().__init__(inputs, outputs, count, weight_scale, bias_scale)
        self.factorized_linear = nn.Linear(inputs, outputs)
        with torch.no_grad():
            self.factorized_linear.weight.zero_()
            self.factorized_linear.bias.zero_()

    def _merged_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.linear.weight + self.factorized_linear.weight.repeat(self.count, 1),
            self.linear.bias + self.factorized_linear.bias.repeat(self.count),
        )


class AtomicV3LayerStacks(nn.Module):
    """Eight SFNNv15 stacks with the frozen squared paths and short skip."""

    def __init__(self):
        super().__init__()
        self.fc0 = FactorizedStackedLinear(
            1024, 32, LAYER_STACKS, FC0_WEIGHT_SCALE, FC0_BIAS_SCALE
        )
        self.fc1 = StackedLinear(64, 32, LAYER_STACKS, FC1_WEIGHT_SCALE, FC1_BIAS_SCALE)
        self.fc2 = StackedLinear(128, 1, LAYER_STACKS, FC2_WEIGHT_SCALE, FC2_BIAS_SCALE)
        with torch.no_grad():
            self.fc2.linear.bias.zero_()

    def forward(
        self,
        value: torch.Tensor,
        bucket_indices: torch.Tensor,
        fake_quantize_activations: bool = True,
        fake_quantize_weights: bool = True,
        validate_bucket_range: bool = True,
    ) -> torch.Tensor:
        if value.ndim != 2 or value.shape[1] != 1024:
            raise ValueError("SFNNv15 layer stacks require [batch, 1024] input")
        if bucket_indices.shape != (value.shape[0],):
            raise ValueError("layer-stack indices must have shape [batch]")
        if bucket_indices.dtype != torch.long:
            raise TypeError("layer-stack indices must use torch.long")
        if validate_bucket_range and torch.any(
            (bucket_indices < 0) | (bucket_indices >= LAYER_STACKS)
        ):
            raise ValueError("layer-stack index outside [0, 7]")

        fc0 = self.fc0(
            value,
            bucket_indices,
            fake_quantize_weights,
            validate_bucket_range=False,
        )
        short_skip = fc0[:, -2:-1] - fc0[:, -1:]
        fc0_squared = fc0.square()
        fc0_linear = fc0
        if fake_quantize_activations:
            fc0_squared = fake_quantize_activation(fc0_squared)
            fc0_linear = fake_quantize_activation(fc0_linear)
        activated0 = clip_hidden_activation(torch.cat((fc0_squared, fc0_linear), dim=1))

        fc1 = self.fc1(
            activated0,
            bucket_indices,
            fake_quantize_weights,
            validate_bucket_range=False,
        )
        fc1_squared = fc1.square()
        fc1_linear = fc1
        if fake_quantize_activations:
            fc1_squared = fake_quantize_activation(fc1_squared)
            fc1_linear = fake_quantize_activation(fc1_linear)
        activated1 = clip_hidden_activation(torch.cat((fc1_squared, fc1_linear), dim=1))

        output = self.fc2(
            torch.cat((activated0, activated1), dim=1),
            bucket_indices,
            fake_quantize_weights,
            validate_bucket_range=False,
        ) + short_skip
        if fake_quantize_activations:
            output = fake_quantize_output(output)
        return output


def pairwise_multiply(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2 or value.shape[1] != 2048:
        raise ValueError("pairwise multiply requires [batch, 2048] input")
    first_us, second_us, first_them, second_them = value.split(512, dim=1)
    return torch.cat((first_us * second_us, first_them * second_them), dim=1)
