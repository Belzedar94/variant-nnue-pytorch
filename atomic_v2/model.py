"""Exact HalfKAv2Atomic / SFNNv15 training graph for AtomicNNUEV2."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .contract import (
    ACCUMULATOR_DIMENSIONS,
    FEATURE_DIMENSIONS,
    FEATURE_NAME,
    LAYER_STACKS,
    PSQT_BUCKETS,
)
from .quantization import (
    FC0_BIAS_SCALE,
    FC0_WEIGHT_SCALE,
    FC1_BIAS_SCALE,
    FC1_WEIGHT_SCALE,
    FC2_BIAS_SCALE,
    FC2_WEIGHT_SCALE,
    FT_ONE,
    PSQT_WEIGHT_SCALE,
    clip_feature_activation,
    clip_hidden_activation,
    fake_quantize_activation,
    fake_quantize_output,
    fake_quantize_psqt_output,
    fake_quantize_weight,
)


def _validate_sparse_inputs(indices: torch.Tensor, values: torch.Tensor) -> None:
    if indices.ndim != 2 or values.shape != indices.shape:
        raise ValueError("feature indices and values must be equally shaped matrices")
    if indices.dtype != torch.int32:
        raise TypeError("feature indices must use torch.int32")
    if values.dtype != torch.float32:
        raise TypeError("feature values must use torch.float32")
    if indices.device != values.device:
        raise ValueError("feature indices and values must use the same device")
    if torch.any(indices < -1):
        raise ValueError("feature indices may only use -1 as the empty sentinel")


def sparse_feature_sum(
    indices: torch.Tensor,
    values: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    sparse_gradient: bool = False,
) -> torch.Tensor:
    _validate_sparse_inputs(indices, values)
    if weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError("feature-transformer parameters must use torch.float32")
    if weight.device != indices.device or bias.device != indices.device:
        raise ValueError("feature inputs and parameters must use the same device")

    valid = torch.cumprod((indices != -1).to(torch.int32), dim=1).to(torch.bool)
    safe_indices = indices.clamp_min(0).to(torch.long)
    scaled_values = torch.where(valid, values, torch.zeros_like(values))
    return F.embedding_bag(
        safe_indices,
        weight,
        mode="sum",
        per_sample_weights=scaled_values,
        # Sparse gradients are safe only when `weight` is the leaf FT matrix.
        # Fake quantization creates views/operations whose SparseCPU backward
        # is unsupported, so that path deliberately requests a dense gradient.
        sparse=sparse_gradient,
    ) + bias


class SparseFeatureTransformer(nn.Module):
    num_features = FEATURE_DIMENSIONS
    accumulator_dimensions = ACCUMULATOR_DIMENSIONS
    psqt_buckets = PSQT_BUCKETS

    def __init__(self, initialize: bool = True):
        super().__init__()
        outputs = self.accumulator_dimensions + self.psqt_buckets
        self.weight = nn.Parameter(torch.empty(self.num_features, outputs, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(self.accumulator_dimensions, dtype=torch.float32))
        self.reset_parameters(initialize)

    def reset_parameters(self, initialize: bool = True) -> None:
        with torch.no_grad():
            if initialize:
                bound = math.sqrt(1.0 / self.num_features)
                self.weight.uniform_(-bound, bound)
                self.bias.uniform_(-bound, bound)
            else:
                self.weight.zero_()
                self.bias.zero_()
            # Start PSQT from a neutral state. It is trained through the same
            # authenticated HalfKAv2Atomic indices as the main accumulator.
            self.weight[:, self.accumulator_dimensions :].zero_()

    def _merged_parameters(
        self, fake_quantize_weights: bool
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        if fake_quantize_weights:
            main_weight = self.weight[:, : self.accumulator_dimensions]
            psqt_weight = self.weight[:, self.accumulator_dimensions :]
            bias = self.bias
            main_weight = fake_quantize_weight(main_weight, FT_ONE)
            psqt_weight = fake_quantize_weight(psqt_weight, PSQT_WEIGHT_SCALE)
            bias = fake_quantize_weight(bias, FT_ONE)
            merged = torch.cat((main_weight, psqt_weight), dim=1)
            sparse_gradient = False
        else:
            merged = self.weight
            bias = self.bias
            sparse_gradient = True
        psqt_bias = torch.zeros(self.psqt_buckets, dtype=bias.dtype, device=bias.device)
        return merged, torch.cat((bias, psqt_bias)), sparse_gradient

    def forward(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        fake_quantize_weights: bool,
    ) -> torch.Tensor:
        merged, bias, sparse_gradient = self._merged_parameters(fake_quantize_weights)
        return sparse_feature_sum(
            indices,
            values,
            merged,
            bias,
            sparse_gradient=sparse_gradient,
        )

    def forward_pair(
        self,
        white_indices: torch.Tensor,
        white_values: torch.Tensor,
        black_indices: torch.Tensor,
        black_values: torch.Tensor,
        fake_quantize_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform both perspectives with one quantized parameter graph."""

        merged, bias, sparse_gradient = self._merged_parameters(fake_quantize_weights)
        white = sparse_feature_sum(
            white_indices,
            white_values,
            merged,
            bias,
            sparse_gradient=sparse_gradient,
        )
        black = sparse_feature_sum(
            black_indices,
            black_values,
            merged,
            bias,
            sparse_gradient=sparse_gradient,
        )
        return white, black


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
    ) -> torch.Tensor:
        weight, bias = self._merged_parameters()
        if fake_quantize_weights:
            weight = fake_quantize_weight(weight, self.weight_scale)
            bias = fake_quantize_weight(bias, self.bias_scale)
        broad_output = F.linear(value, weight, bias)
        rows = broad_output.reshape(-1, self.outputs)
        offsets = torch.arange(
            0,
            bucket_indices.numel() * self.count,
            self.count,
            device=broad_output.device,
        )
        return rows[bucket_indices.reshape(-1) + offsets]

    @torch.no_grad()
    def bucket_parameters(self, bucket: int) -> tuple[torch.Tensor, torch.Tensor]:
        if bucket < 0 or bucket >= self.count:
            raise IndexError(bucket)
        begin = bucket * self.outputs
        end = begin + self.outputs
        weight, bias = self._merged_parameters()
        return weight[begin:end].detach(), bias[begin:end].detach()


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
        weight = self.linear.weight + self.factorized_linear.weight.repeat(self.count, 1)
        bias = self.linear.bias + self.factorized_linear.bias.repeat(self.count)
        return weight, bias


class AtomicLayerStacks(nn.Module):
    """Eight exact SFNNv15 stacks; squared paths have no wire-hash node."""

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
    ) -> torch.Tensor:
        if value.ndim != 2 or value.shape[1] != 1024:
            raise ValueError("SFNNv15 layer stacks require [batch, 1024] input")
        if bucket_indices.shape != (value.shape[0],):
            raise ValueError("layer-stack indices must have shape [batch]")
        if bucket_indices.dtype != torch.long:
            raise TypeError("layer-stack indices must use torch.long")
        if torch.any((bucket_indices < 0) | (bucket_indices >= LAYER_STACKS)):
            raise ValueError("layer-stack index outside [0, 7]")

        fc0 = self.fc0(value, bucket_indices, fake_quantize_weights)
        short_skip = fc0[:, -2:-1] - fc0[:, -1:]
        # Deliberately square the signed pre-activation before clipping. This
        # is the exact SqrClippedReLU order used by the engine and the pinned
        # official trainer; negative inputs therefore contribute on this path.
        fc0_squared = fc0.square()
        fc0_linear = fc0
        if fake_quantize_activations:
            fc0_squared = fake_quantize_activation(fc0_squared)
            fc0_linear = fake_quantize_activation(fc0_linear)
        activated0 = clip_hidden_activation(torch.cat((fc0_squared, fc0_linear), dim=1))

        fc1 = self.fc1(activated0, bucket_indices, fake_quantize_weights)
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
        ) + short_skip
        if fake_quantize_activations:
            output = fake_quantize_output(output)
        return output


def pairwise_multiply(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2 or value.shape[1] != 2048:
        raise ValueError("pairwise multiply requires [batch, 2048] input")
    first_us, second_us, first_them, second_them = value.split(512, dim=1)
    return torch.cat((first_us * second_us, first_them * second_them), dim=1)


class AtomicNNUEV2(nn.Module):
    feature_name = FEATURE_NAME
    num_features = FEATURE_DIMENSIONS
    accumulator_dimensions = ACCUMULATOR_DIMENSIONS
    psqt_buckets = PSQT_BUCKETS
    layer_stacks = LAYER_STACKS

    def __init__(self, initialize: bool = True):
        super().__init__()
        self._allocate_feature_parameters(initialize)
        self.network = AtomicLayerStacks()

    def _allocate_feature_parameters(self, initialize: bool = True) -> None:
        self.feature_transformer = SparseFeatureTransformer(initialize=initialize)

    @torch.no_grad()
    def clip_weights(self) -> None:
        """Keep every coalesced hidden weight inside the signed int8 grid."""
        fc0_limit = 127.0 / FC0_WEIGHT_SCALE
        virtual = self.network.fc0.factorized_linear.weight.repeat(LAYER_STACKS, 1)
        base = self.network.fc0.linear.weight
        base.copy_(
            torch.maximum(
                torch.minimum(base, base.new_full((), fc0_limit) - virtual),
                base.new_full((), -fc0_limit) - virtual,
            )
        )
        self.network.fc1.linear.weight.clamp_(
            -127.0 / FC1_WEIGHT_SCALE, 127.0 / FC1_WEIGHT_SCALE
        )
        self.network.fc2.linear.weight.clamp_(
            -127.0 / FC2_WEIGHT_SCALE, 127.0 / FC2_WEIGHT_SCALE
        )

    def forward(
        self,
        us: torch.Tensor,
        them: torch.Tensor,
        white_indices: torch.Tensor,
        white_values: torch.Tensor,
        black_indices: torch.Tensor,
        black_values: torch.Tensor,
        psqt_indices: torch.Tensor,
        layer_stack_indices: torch.Tensor,
        fake_quantize_activations: bool = True,
        fake_quantize_weights: bool = True,
    ) -> torch.Tensor:
        if us.ndim != 2 or us.shape[1] != 1 or them.shape != us.shape:
            raise ValueError("us and them must have shape [batch, 1]")
        batch_size = us.shape[0]
        if psqt_indices.shape != (batch_size,) or layer_stack_indices.shape != (
            batch_size,
        ):
            raise ValueError("bucket indices must have shape [batch]")
        if psqt_indices.dtype != torch.long or layer_stack_indices.dtype != torch.long:
            raise TypeError("bucket indices must use torch.long")
        if torch.any((psqt_indices < 0) | (psqt_indices >= self.psqt_buckets)):
            raise ValueError("PSQT bucket index outside [0, 7]")

        white, black = self.feature_transformer.forward_pair(
            white_indices,
            white_values,
            black_indices,
            black_values,
            fake_quantize_weights,
        )
        white_main, white_psqt = white.split(
            (self.accumulator_dimensions, self.psqt_buckets), dim=1
        )
        black_main, black_psqt = black.split(
            (self.accumulator_dimensions, self.psqt_buckets), dim=1
        )

        white_bucket = white_psqt.gather(1, psqt_indices.reshape(-1, 1))
        black_bucket = black_psqt.gather(1, psqt_indices.reshape(-1, 1))
        side_ordered = us * torch.cat((white_main, black_main), dim=1) + them * torch.cat(
            (black_main, white_main), dim=1
        )
        transformed = pairwise_multiply(clip_feature_activation(side_ordered))
        if fake_quantize_activations:
            transformed = fake_quantize_activation(transformed)

        positional = fake_quantize_psqt_output(
            (white_bucket - black_bucket) * (us - 0.5)
        )
        return self.network(
            transformed,
            layer_stack_indices,
            fake_quantize_activations,
            fake_quantize_weights,
        ) + positional
