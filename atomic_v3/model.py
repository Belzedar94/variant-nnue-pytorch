"""Isolated mixed-slice AtomicNNUEV3 training graph."""

from __future__ import annotations

import math

import torch
from torch import nn

from .contract import (
    ACCUMULATOR_DIMENSIONS,
    BACKEND_NAME,
    BLAST_RING_DIMENSIONS,
    CAPTURE_PAIR_DIMENSIONS,
    FEATURE_NAME,
    FC0_OUTPUTS,
    FC1_OUTPUTS,
    FC2_OUTPUTS,
    HM_OUTPUT_DIMENSIONS,
    HM_PHYSICAL_DIMENSIONS,
    HM_TRAINING_DIMENSIONS,
    HM_VIRTUAL_DIMENSIONS,
    KING_BLAST_EP_DIMENSIONS,
    LAYER_STACKS,
    PSQT_BUCKETS,
)
from .dataset import AtomicV3Batch, validate_batch
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
    fake_quantize_psqt_output,
    fake_quantize_psqt_weight,
)
from .sparse_transformer import double_sparse_linear, sparse_linear


@torch.no_grad()
def _clip_factorized_training_sums_(
    base: torch.Tensor,
    factor: torch.Tensor,
    *,
    count: int,
    minimum: float,
    maximum: float,
    label: str,
) -> None:
    """Fast FP32 per-step clamp for a persisted ``base + factor``.

    This deliberately performs no reductions, host synchronizations, FP64
    widening, or production-sized ``repeat``.  The exact signed-wire audit is
    deferred to :func:`_clip_factorized_i32_sums_` at persistence boundaries.
    """

    if base.dtype != torch.float32 or factor.dtype != torch.float32:
        raise TypeError(f"Atomic V3 {label} factors must use float32 parameters")
    expected_shape = (factor.shape[0] * count, *factor.shape[1:])
    if base.shape != expected_shape or base.device != factor.device:
        raise ValueError(f"Atomic V3 {label} factor shapes/devices are inconsistent")
    factor.clamp_(minimum, maximum)
    grouped_base = base.view(count, *factor.shape)
    lower = minimum - factor
    upper = maximum - factor
    torch.minimum(grouped_base, upper.unsqueeze(0), out=grouped_base)
    torch.maximum(grouped_base, lower.unsqueeze(0), out=grouped_base)


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


def _fake_quantize_training_fp32(
    value: torch.Tensor, *, scale: float, minimum: float, maximum: float
) -> torch.Tensor:
    """Fast float32 QAT boundary with a straight-through gradient.

    The exact widening/range checks remain at the persistent clipping and
    serialization boundaries.  The training graph only needs the quantized
    float32 grid; promoting every active row to float64 was both unnecessary
    and prohibitively expensive.
    """

    if value.dtype != torch.float32:
        raise TypeError("Atomic V3 training QAT requires float32 parameters")
    hard = value.mul(scale).round().clamp(minimum, maximum).div(scale).detach()
    return hard + (value - value.detach())


def _fake_quantize_i16_training(value: torch.Tensor) -> torch.Tensor:
    return _fake_quantize_training_fp32(
        value, scale=FT_ONE, minimum=-32768.0, maximum=32767.0
    )


def _fake_quantize_i8_training(value: torch.Tensor) -> torch.Tensor:
    return _fake_quantize_training_fp32(
        value, scale=FT_ONE, minimum=-128.0, maximum=127.0
    )


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

    def _prepared_weights(
        self, fake_quantize_weights: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.hm_bucket_weight.shape[0] % self.hm_virtual_weight.shape[0] != 0:
            raise ValueError("Atomic V3 HM factor shapes are inconsistent")
        factor_count = self.hm_bucket_weight.shape[0] // self.hm_virtual_weight.shape[0]
        hm = self.hm_bucket_weight + self.hm_virtual_weight.repeat(factor_count, 1)
        bias = self.bias
        capture_pair = self.capture_pair_weight
        king_blast_ep = self.king_blast_ep_weight
        blast_ring = self.blast_ring_weight
        if fake_quantize_weights:
            hm = torch.cat(
                (
                    _fake_quantize_i16_training(hm[:, :ACCUMULATOR_DIMENSIONS]),
                    # PSQT's 9600 scale is not exactly representable through
                    # every float32 half-step.  Preserve the serializer's
                    # float64 ties-to-even wire semantics on these 8 columns.
                    fake_quantize_psqt_weight(hm[:, ACCUMULATOR_DIMENSIONS:]),
                ),
                dim=1,
            )
            bias = _fake_quantize_i16_training(bias)
            capture_pair = _fake_quantize_i8_training(capture_pair)
            king_blast_ep = _fake_quantize_i16_training(king_blast_ep)
            blast_ring = _fake_quantize_i8_training(blast_ring)
        hm_bias = torch.cat((bias, bias.new_zeros(PSQT_BUCKETS)), dim=0)
        return hm, hm_bias, capture_pair, king_blast_ep, blast_ring

    @staticmethod
    def _zero_bias(weight: torch.Tensor) -> torch.Tensor:
        return weight.new_zeros(weight.shape[1])

    @staticmethod
    def _split_hm(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return value.split((ACCUMULATOR_DIMENSIONS, PSQT_BUCKETS), dim=1)

    def forward(self, perspective, fake_quantize_weights: bool) -> tuple[torch.Tensor, torch.Tensor]:
        hm_weight, hm_bias, capture_pair, king_blast_ep, blast_ring = self._prepared_weights(
            fake_quantize_weights
        )
        hm = sparse_linear(
            perspective.hm.indices,
            perspective.hm.values,
            hm_weight,
            hm_bias,
            unit_values=True,
        )
        main, psqt = self._split_hm(hm)
        for features, weight in (
            (perspective.capture_pair, capture_pair),
            (perspective.king_blast_ep, king_blast_ep),
            (perspective.blast_ring, blast_ring),
        ):
            main = main + sparse_linear(
                features.indices,
                features.values,
                weight,
                self._zero_bias(weight),
                unit_values=True,
            )
        return main, psqt

    def forward_pair(
        self, white, black, fake_quantize_weights: bool
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Transform both perspectives while sharing QAT and dense gradients."""

        hm_weight, hm_bias, capture_pair, king_blast_ep, blast_ring = self._prepared_weights(
            fake_quantize_weights
        )
        white_hm, black_hm = double_sparse_linear(
            white.hm.indices,
            white.hm.values,
            black.hm.indices,
            black.hm.values,
            hm_weight,
            hm_bias,
            unit_values=True,
        )
        white_main, white_psqt = self._split_hm(white_hm)
        black_main, black_psqt = self._split_hm(black_hm)
        for white_features, black_features, weight in (
            (white.capture_pair, black.capture_pair, capture_pair),
            (white.king_blast_ep, black.king_blast_ep, king_blast_ep),
            (white.blast_ring, black.blast_ring, blast_ring),
        ):
            white_relation, black_relation = double_sparse_linear(
                white_features.indices,
                white_features.values,
                black_features.indices,
                black_features.values,
                weight,
                self._zero_bias(weight),
                unit_values=True,
            )
            white_main = white_main + white_relation
            black_main = black_main + black_relation
        return (white_main, white_psqt), (black_main, black_psqt)

    @torch.no_grad()
    def clip_training_weights(self) -> None:
        """Clamp the QAT domain without exact wire-boundary overhead."""

        i16_min, i16_max = -32768.0 / FT_ONE, 32767.0 / FT_ONE
        i8_min, i8_max = -128.0 / FT_ONE, 127.0 / FT_ONE
        self.bias.clamp_(i16_min, i16_max)
        self.capture_pair_weight.clamp_(i8_min, i8_max)
        self.king_blast_ep_weight.clamp_(i16_min, i16_max)
        self.blast_ring_weight.clamp_(i8_min, i8_max)

        if self.hm_bucket_weight.shape[0] % self.hm_virtual_weight.shape[0] != 0:
            raise ValueError("Atomic V3 HM factor shapes are inconsistent")
        factor_count = self.hm_bucket_weight.shape[0] // self.hm_virtual_weight.shape[0]
        _clip_factorized_training_sums_(
            self.hm_bucket_weight[:, :ACCUMULATOR_DIMENSIONS],
            self.hm_virtual_weight[:, :ACCUMULATOR_DIMENSIONS],
            count=factor_count,
            minimum=i16_min,
            maximum=i16_max,
            label="HM main weight",
        )
        _clip_factorized_training_sums_(
            self.hm_bucket_weight[:, ACCUMULATOR_DIMENSIONS:],
            self.hm_virtual_weight[:, ACCUMULATOR_DIMENSIONS:],
            count=factor_count,
            minimum=PSQT_EXPORT_MIN,
            maximum=PSQT_EXPORT_MAX,
            label="HM PSQT weight",
        )

    @torch.no_grad()
    def clip_weights(self) -> None:
        """Exact persistence-boundary clamp and signed-i32 PSQT audit."""

        hm_psqt_base = self.hm_bucket_weight[:, ACCUMULATOR_DIMENSIONS:]
        hm_psqt_factor = self.hm_virtual_weight[:, ACCUMULATOR_DIMENSIONS:]
        if not torch.all(torch.isfinite(hm_psqt_base)) or not torch.all(
            torch.isfinite(hm_psqt_factor)
        ):
            raise FloatingPointError("Atomic V3 HM PSQT factor is non-finite")
        self.clip_training_weights()
        factor_count = self.hm_bucket_weight.shape[0] // self.hm_virtual_weight.shape[0]
        _clip_factorized_i32_sums_(
            hm_psqt_base,
            hm_psqt_factor.repeat(factor_count, 1),
            scale=PSQT_WEIGHT_SCALE,
            minimum=PSQT_EXPORT_MIN,
            maximum=PSQT_EXPORT_MAX,
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
        # A valid V3 wire can contain i32 values whose dequantized float32
        # image is shared by more than one integer.  Keep the authenticated
        # source integers as persistent, non-trainable buffers so an imported
        # network remains exactly serializable after a normal state_dict
        # checkpoint/resume.  These buffers are deliberately absent from the
        # forward graph and add no optimizer state.
        self.register_buffer(
            "_atomic_v3_imported_i32_valid",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "_atomic_v3_imported_hm_psqt_i32",
            torch.zeros(
                (HM_PHYSICAL_DIMENSIONS, PSQT_BUCKETS), dtype=torch.int32
            ),
            persistent=True,
        )
        self.register_buffer(
            "_atomic_v3_imported_fc0_bias_i32",
            torch.zeros((LAYER_STACKS, FC0_OUTPUTS), dtype=torch.int32),
            persistent=True,
        )
        self.register_buffer(
            "_atomic_v3_imported_fc1_bias_i32",
            torch.zeros((LAYER_STACKS, FC1_OUTPUTS), dtype=torch.int32),
            persistent=True,
        )
        self.register_buffer(
            "_atomic_v3_imported_fc2_bias_i32",
            torch.zeros((LAYER_STACKS, FC2_OUTPUTS), dtype=torch.int32),
            persistent=True,
        )

    @torch.no_grad()
    def _clip_training_dense_weights(self) -> None:
        fc0_limit = 127.0 / FC0_WEIGHT_SCALE
        fc0_factor = self.network.fc0.factorized_linear.weight
        _clip_factorized_training_sums_(
            self.network.fc0.linear.weight,
            fc0_factor,
            count=LAYER_STACKS,
            minimum=-fc0_limit,
            maximum=fc0_limit,
            label="FC0 weight",
        )
        self.network.fc1.linear.weight.clamp_(
            -127.0 / FC1_WEIGHT_SCALE, 127.0 / FC1_WEIGHT_SCALE
        )
        self.network.fc2.linear.weight.clamp_(
            -127.0 / FC2_WEIGHT_SCALE, 127.0 / FC2_WEIGHT_SCALE
        )
        _clip_factorized_training_sums_(
            self.network.fc0.linear.bias,
            self.network.fc0.factorized_linear.bias,
            count=LAYER_STACKS,
            minimum=FC0_BIAS_EXPORT_MIN,
            maximum=FC0_BIAS_EXPORT_MAX,
            label="FC0 bias",
        )
        self.network.fc1.linear.bias.clamp_(FC1_BIAS_EXPORT_MIN, FC1_BIAS_EXPORT_MAX)
        self.network.fc2.linear.bias.clamp_(FC2_BIAS_EXPORT_MIN, FC2_BIAS_EXPORT_MAX)

    @torch.no_grad()
    def clip_training_weights(self) -> None:
        """Low-overhead clamp intended for every optimizer step."""

        self.feature_transformer.clip_training_weights()
        self._clip_training_dense_weights()

    @torch.no_grad()
    def clip_weights(self) -> None:
        """Exact checkpoint/export clamp with signed-wire verification."""

        dense_biases = (
            ("FC0 base", self.network.fc0.linear.bias),
            ("FC0 factor", self.network.fc0.factorized_linear.bias),
            ("FC1", self.network.fc1.linear.bias),
            ("FC2", self.network.fc2.linear.bias),
        )
        for name, bias in dense_biases:
            if not torch.all(torch.isfinite(bias)):
                raise FloatingPointError(f"Atomic V3 {name} bias is non-finite")
        self.feature_transformer.clip_weights()
        self._clip_training_dense_weights()

        # The dense wire stores signed-i32 biases.  FC0 is factorized during
        # training but serialized as base+factor, so constrain the coalesced
        # value rather than merely clipping each factor independently.
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
        (white_main, white_psqt), (black_main, black_psqt) = self.feature_transformer.forward_pair(
            batch.white, batch.black, fake_quantize_weights
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
            validate_bucket_range=False,
        ) + positional
