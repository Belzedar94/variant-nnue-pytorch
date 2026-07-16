"""Isolated mixed-slice AtomicNNUEV3 training graph."""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .contract import (
    ACCUMULATOR_DIMENSIONS,
    BACKEND_NAME,
    BLAST_RING_DIMENSIONS,
    CAPTURE_PAIR_DIMENSIONS,
    FEATURE_NAME,
    HM_OUTPUT_DIMENSIONS,
    HM_TRAINING_DIMENSIONS,
    HM_VIRTUAL_DIMENSIONS,
    KING_BLAST_EP_DIMENSIONS,
    LAYER_STACKS,
    PSQT_BUCKETS,
)
from .dataset import AtomicV3Batch, SparseSliceBatch, validate_batch
from .dense import AtomicV3LayerStacks, pairwise_multiply
from .quantization import (
    FC0_BIAS_SCALE,
    FC0_BIAS_EXPORT_MAX,
    FC0_BIAS_EXPORT_MIN,
    FC0_WEIGHT_SCALE,
    FC1_BIAS_EXPORT_MAX,
    FC1_BIAS_EXPORT_MIN,
    FC1_WEIGHT_SCALE,
    FC2_BIAS_EXPORT_MAX,
    FC2_BIAS_EXPORT_MIN,
    FC2_WEIGHT_SCALE,
    FT_ONE,
    INT32_MAX,
    INT32_MIN,
    PSQT_EXPORT_MAX,
    PSQT_EXPORT_MIN,
    PSQT_WEIGHT_SCALE,
    clip_feature_activation,
    fake_quantize_activation,
    fake_quantize_i16_feature,
    fake_quantize_i8_feature,
    fake_quantize_psqt_output,
    fake_quantize_psqt_weight,
)


@torch.no_grad()
def _clip_factorized_i32_sums_(
    base: torch.Tensor,
    expanded_factor: torch.Tensor,
    *,
    scale: float,
    minimum: float,
    maximum: float,
    label: str,
) -> None:
    """Constrain persisted ``base + factor`` in both accumulation widths.

    Computing ``maximum - factor`` in float32 is not sufficient.  When the two
    operands have opposite signs, that subtraction can round the persisted
    base outward; adding the operands again can then produce an i32 overflow.
    Form the target in float64, write a coordinated float32 base, and correct
    it with bounded ``nextafter`` steps until the *actual wire rounding* is in
    range from both a float32 and float64 view of the persisted operands.
    """

    if base.dtype != torch.float32 or expanded_factor.dtype != torch.float32:
        raise TypeError(f"Atomic V3 {label} factors must use float32 parameters")
    if base.shape != expanded_factor.shape or base.device != expanded_factor.device:
        raise ValueError(f"Atomic V3 {label} factor shapes/devices are inconsistent")
    if not torch.all(torch.isfinite(base)) or not torch.all(torch.isfinite(expanded_factor)):
        raise FloatingPointError(f"Atomic V3 {label} factor is non-finite")

    target = (base.to(torch.float64) + expanded_factor.to(torch.float64)).clamp(
        float(minimum), float(maximum)
    )
    base.copy_((target - expanded_factor.to(torch.float64)).to(base.dtype))

    positive = base.new_full((), float("inf"))
    negative = base.new_full((), float("-inf"))
    correction_target = base + expanded_factor
    for _ in range(4):
        merged32 = base + expanded_factor
        merged64 = base.to(torch.float64) + expanded_factor.to(torch.float64)
        # Promote after the float32 arithmetic but before comparing with the
        # asymmetric i32 endpoint.  Comparing INT32_MAX directly in float32
        # rounds the threshold itself to 2**31 and masks the overflow.
        integer32 = torch.round(merged32 * merged32.new_tensor(scale)).to(torch.float64)
        integer32_wide = torch.round(merged32.to(torch.float64) * float(scale))
        integer64 = torch.round(merged64 * float(scale))
        too_low = (
            (integer32 < INT32_MIN)
            | (integer32_wide < INT32_MIN)
            | (integer64 < INT32_MIN)
        )
        too_high = (
            (integer32 > INT32_MAX)
            | (integer32_wide > INT32_MAX)
            | (integer64 > INT32_MAX)
        )
        # Correct in the coalesced value's ULP domain.  Moving the base itself
        # is not bounded when cancellation makes the base much smaller than
        # the factor: one base ULP can then be far too small to change the
        # persisted sum.  Re-targeting one merged float32 ULP inward and
        # solving for the base in float64 is bounded because |base| is at most
        # twice the clamped coalesced/factor envelope.
        correction_target = torch.where(
            too_low,
            torch.nextafter(correction_target, positive),
            torch.where(
                too_high,
                torch.nextafter(correction_target, negative),
                correction_target,
            ),
        )
        corrected_base = (
            correction_target.to(torch.float64) - expanded_factor.to(torch.float64)
        ).to(base.dtype)
        base.copy_(torch.where(too_low | too_high, corrected_base, base))

    merged32 = base + expanded_factor
    merged64 = base.to(torch.float64) + expanded_factor.to(torch.float64)
    integer32 = torch.round(merged32 * merged32.new_tensor(scale)).to(torch.float64)
    integer32_wide = torch.round(merged32.to(torch.float64) * float(scale))
    integer64 = torch.round(merged64 * float(scale))
    valid = (
        torch.isfinite(merged32)
        & torch.isfinite(merged64)
        & (integer32 >= INT32_MIN)
        & (integer32 <= INT32_MAX)
        & (integer32_wide >= INT32_MIN)
        & (integer32_wide <= INT32_MAX)
        & (integer64 >= INT32_MIN)
        & (integer64 <= INT32_MAX)
    )
    if not torch.all(valid):
        raise FloatingPointError(
            f"Atomic V3 {label} could not be made signed-i32 exportable"
        )


@torch.no_grad()
def _clip_factorized_i32_biases_(
    base: torch.Tensor,
    factor: torch.Tensor,
    *,
    count: int,
    scale: float,
    minimum: float,
    maximum: float,
) -> None:
    """Constrain every persisted factorized dense bias."""

    if base.ndim != 1 or factor.ndim != 1 or base.numel() != factor.numel() * count:
        raise ValueError("Atomic V3 factorized dense bias shapes are inconsistent")
    factor.clamp_(minimum, maximum)
    _clip_factorized_i32_sums_(
        base,
        factor.repeat(count),
        scale=scale,
        minimum=minimum,
        maximum=maximum,
        label="FC0 bias",
    )


def _sparse_sum(
    features: SparseSliceBatch,
    weight: torch.Tensor,
    quantize: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> torch.Tensor:
    indices, values = features.indices, features.values
    valid = indices != -1
    safe = indices.clamp_min(0).to(torch.long)
    # Quantize after gathering.  Besides avoiding a dense fake-quantized copy,
    # this preserves sparse gradients for the 77,900-row production tables.
    rows = F.embedding(safe, weight, sparse=True)
    if quantize is not None:
        rows = quantize(rows)
    scales = torch.where(valid, values, torch.zeros_like(values)).unsqueeze(-1)
    return (rows * scales).sum(dim=1)


class AtomicV3FeatureTransformer(nn.Module):
    """Four frozen slices; only HM owns factor rows and PSQT columns."""

    def __init__(self, initialize: bool = True):
        super().__init__()
        self.bias = nn.Parameter(torch.empty(ACCUMULATOR_DIMENSIONS, dtype=torch.float32))
        self.hm_bucket_weight = nn.Parameter(
            torch.empty(HM_TRAINING_DIMENSIONS, HM_OUTPUT_DIMENSIONS, dtype=torch.float32)
        )
        self.hm_virtual_weight = nn.Parameter(
            torch.empty(HM_VIRTUAL_DIMENSIONS, HM_OUTPUT_DIMENSIONS, dtype=torch.float32)
        )
        self.capture_pair_weight = nn.Parameter(
            torch.empty(CAPTURE_PAIR_DIMENSIONS, ACCUMULATOR_DIMENSIONS, dtype=torch.float32)
        )
        self.king_blast_ep_weight = nn.Parameter(
            torch.empty(KING_BLAST_EP_DIMENSIONS, ACCUMULATOR_DIMENSIONS, dtype=torch.float32)
        )
        self.blast_ring_weight = nn.Parameter(
            torch.empty(BLAST_RING_DIMENSIONS, ACCUMULATOR_DIMENSIONS, dtype=torch.float32)
        )
        self.reset_parameters(initialize)

    @torch.no_grad()
    def reset_parameters(self, initialize: bool = True) -> None:
        if initialize:
            bound = math.sqrt(1.0 / (HM_TRAINING_DIMENSIONS + HM_VIRTUAL_DIMENSIONS))
            self.hm_bucket_weight.uniform_(-bound, bound)
            # Begin with an unfactorized model; the shared factor learns only
            # signal supported across king buckets.
            self.hm_virtual_weight.zero_()
            self.bias.uniform_(-bound, bound)
            self.capture_pair_weight.zero_()
            self.king_blast_ep_weight.zero_()
            self.blast_ring_weight.zero_()
        else:
            for parameter in self.parameters():
                parameter.zero_()
        # V3 relations have no hidden zero-initialized PSQT parameters.  HM's
        # real PSQT columns start neutral.
        self.hm_bucket_weight[:, ACCUMULATOR_DIMENSIONS:].zero_()
        self.hm_virtual_weight[:, ACCUMULATOR_DIMENSIONS:].zero_()

    def _hm_sum(self, features: SparseSliceBatch, fake_quantize_weights: bool) -> torch.Tensor:
        safe = features.indices.clamp_min(0).to(torch.long)
        valid = features.indices != -1
        bucket_rows = F.embedding(safe, self.hm_bucket_weight, sparse=True)
        virtual_rows = F.embedding(
            torch.remainder(safe, HM_VIRTUAL_DIMENSIONS), self.hm_virtual_weight, sparse=True
        )
        coalesced = bucket_rows + virtual_rows
        if fake_quantize_weights:
            main = fake_quantize_i16_feature(coalesced[..., :ACCUMULATOR_DIMENSIONS])
            psqt = fake_quantize_psqt_weight(coalesced[..., ACCUMULATOR_DIMENSIONS:])
            coalesced = torch.cat((main, psqt), dim=-1)
        scales = torch.where(valid, features.values, torch.zeros_like(features.values)).unsqueeze(-1)
        return (coalesced * scales).sum(dim=1)

    def forward(self, perspective, fake_quantize_weights: bool) -> tuple[torch.Tensor, torch.Tensor]:
        hm = self._hm_sum(perspective.hm, fake_quantize_weights)
        main, psqt = hm.split((ACCUMULATOR_DIMENSIONS, PSQT_BUCKETS), dim=1)
        bias = fake_quantize_i16_feature(self.bias) if fake_quantize_weights else self.bias
        main = main + bias
        main = main + _sparse_sum(
            perspective.capture_pair,
            self.capture_pair_weight,
            fake_quantize_i8_feature if fake_quantize_weights else None,
        )
        main = main + _sparse_sum(
            perspective.king_blast_ep,
            self.king_blast_ep_weight,
            fake_quantize_i16_feature if fake_quantize_weights else None,
        )
        main = main + _sparse_sum(
            perspective.blast_ring,
            self.blast_ring_weight,
            fake_quantize_i8_feature if fake_quantize_weights else None,
        )
        return main, psqt

    @torch.no_grad()
    def clip_weights(self) -> None:
        i16_min, i16_max = -32768.0 / FT_ONE, 32767.0 / FT_ONE
        i8_min, i8_max = -128.0 / FT_ONE, 127.0 / FT_ONE
        psqt_min = PSQT_EXPORT_MIN
        psqt_max = PSQT_EXPORT_MAX
        hm_psqt_base = self.hm_bucket_weight[:, ACCUMULATOR_DIMENSIONS:]
        hm_psqt_factor = self.hm_virtual_weight[:, ACCUMULATOR_DIMENSIONS:]
        if not torch.all(torch.isfinite(hm_psqt_base)) or not torch.all(
            torch.isfinite(hm_psqt_factor)
        ):
            raise FloatingPointError("Atomic V3 HM PSQT factor is non-finite")
        self.bias.clamp_(i16_min, i16_max)
        self.capture_pair_weight.clamp_(i8_min, i8_max)
        self.king_blast_ep_weight.clamp_(i16_min, i16_max)
        self.blast_ring_weight.clamp_(i8_min, i8_max)

        # The wire stores bucket+virtual, so clipping either factor alone is
        # insufficient.  Process one king bucket at a time to avoid a second
        # production-size tensor.
        self.hm_virtual_weight[:, :ACCUMULATOR_DIMENSIONS].clamp_(i16_min, i16_max)
        self.hm_virtual_weight[:, ACCUMULATOR_DIMENSIONS:].clamp_(psqt_min, psqt_max)
        if self.hm_bucket_weight.shape[0] % self.hm_virtual_weight.shape[0] != 0:
            raise ValueError("Atomic V3 HM factor shapes are inconsistent")
        factor_rows = self.hm_virtual_weight.shape[0]
        factor_count = self.hm_bucket_weight.shape[0] // factor_rows
        for bucket in range(factor_count):
            begin, end = bucket * factor_rows, (bucket + 1) * factor_rows
            base = self.hm_bucket_weight[begin:end]
            virtual = self.hm_virtual_weight
            base_main = base[:, :ACCUMULATOR_DIMENSIONS]
            virtual_main = virtual[:, :ACCUMULATOR_DIMENSIONS]
            base_main.copy_(
                torch.maximum(
                    torch.minimum(base_main, base_main.new_tensor(i16_max) - virtual_main),
                    base_main.new_tensor(i16_min) - virtual_main,
                )
            )
        _clip_factorized_i32_sums_(
            hm_psqt_base,
            hm_psqt_factor.repeat(factor_count, 1),
            scale=PSQT_WEIGHT_SCALE,
            minimum=psqt_min,
            maximum=psqt_max,
            label="HM PSQT weight",
        )


class AtomicNNUEV3(nn.Module):
    backend_name = BACKEND_NAME
    feature_name = FEATURE_NAME
    accumulator_dimensions = ACCUMULATOR_DIMENSIONS
    psqt_buckets = PSQT_BUCKETS
    layer_stacks = LAYER_STACKS

    def __init__(self, initialize: bool = True):
        super().__init__()
        self.feature_transformer = AtomicV3FeatureTransformer(initialize=initialize)
        self.network = AtomicV3LayerStacks()

    @torch.no_grad()
    def clip_weights(self) -> None:
        self.feature_transformer.clip_weights()
        fc0_limit = 127.0 / FC0_WEIGHT_SCALE
        fc0_factor = self.network.fc0.factorized_linear.weight
        fc0_factor.clamp_(-fc0_limit, fc0_limit)
        virtual = fc0_factor.repeat(LAYER_STACKS, 1)
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

        # The dense wire stores signed-i32 biases.  FC0 is factorized during
        # training but serialized as base+factor, so constrain the coalesced
        # value rather than merely clipping each factor independently.
        dense_biases = (
            ("FC0 base", self.network.fc0.linear.bias),
            ("FC0 factor", self.network.fc0.factorized_linear.bias),
            ("FC1", self.network.fc1.linear.bias),
            ("FC2", self.network.fc2.linear.bias),
        )
        for name, bias in dense_biases:
            if not torch.all(torch.isfinite(bias)):
                raise FloatingPointError(f"Atomic V3 {name} bias is non-finite")

        _clip_factorized_i32_biases_(
            self.network.fc0.linear.bias,
            self.network.fc0.factorized_linear.bias,
            count=LAYER_STACKS,
            scale=FC0_BIAS_SCALE,
            minimum=FC0_BIAS_EXPORT_MIN,
            maximum=FC0_BIAS_EXPORT_MAX,
        )
        self.network.fc1.linear.bias.clamp_(FC1_BIAS_EXPORT_MIN, FC1_BIAS_EXPORT_MAX)
        self.network.fc2.linear.bias.clamp_(FC2_BIAS_EXPORT_MIN, FC2_BIAS_EXPORT_MAX)

    def forward(
        self,
        batch: AtomicV3Batch,
        *,
        fake_quantize_activations: bool = True,
        fake_quantize_weights: bool = True,
        validate: bool = True,
    ) -> torch.Tensor:
        if validate:
            validate_batch(batch)
        white_main, white_psqt = self.feature_transformer.forward(
            batch.white, fake_quantize_weights
        )
        black_main, black_psqt = self.feature_transformer.forward(
            batch.black, fake_quantize_weights
        )
        white_bucket = white_psqt.gather(1, batch.bucket_indices.reshape(-1, 1))
        black_bucket = black_psqt.gather(1, batch.bucket_indices.reshape(-1, 1))
        us = batch.side_to_move_white
        them = 1.0 - us
        side_ordered = us * torch.cat((white_main, black_main), dim=1) + them * torch.cat(
            (black_main, white_main), dim=1
        )
        transformed = pairwise_multiply(clip_feature_activation(side_ordered))
        if fake_quantize_activations:
            transformed = fake_quantize_activation(transformed)
        stm_psqt_difference = us * (white_bucket - black_bucket) + them * (
            black_bucket - white_bucket
        )
        positional = fake_quantize_psqt_output(stm_psqt_difference * 0.5)
        return self.network(
            transformed,
            batch.bucket_indices,
            fake_quantize_activations,
            fake_quantize_weights,
        ) + positional
