from types import SimpleNamespace

import pytest
import torch

import atomic_v3.training as v3_training
from atomic_v3.dataset import SparseSliceBatch, load_canonical_fixture
from atomic_v3.dense import AtomicV3LayerStacks, fake_quantize_integer_bias
from atomic_v3.model import AtomicNNUEV3, AtomicV3FeatureTransformer
from atomic_v3.quantization import (
    FC0_BIAS_EXPORT_MAX,
    FC0_BIAS_EXPORT_MIN,
    FC0_BIAS_SCALE,
    FC0_WEIGHT_SCALE,
    FC1_BIAS_EXPORT_MAX,
    FC1_BIAS_EXPORT_MIN,
    FC1_BIAS_SCALE,
    FC2_BIAS_EXPORT_MAX,
    FC2_BIAS_EXPORT_MIN,
    FC2_BIAS_SCALE,
    FT_ONE,
    PSQT_EXPORT_MAX,
    PSQT_EXPORT_MIN,
    PSQT_WEIGHT_SCALE,
    fake_quantize_i16_feature,
    fake_quantize_i8_feature,
    fake_quantize_psqt_output,
    fake_quantize_psqt_weight,
)
from atomic_v3.training import (
    atomic_loss,
    create_core_optimizer,
    deterministic_cpu_one_step,
    optimizer_step,
)


def _slice(index=0):
    return SparseSliceBatch(
        torch.tensor([[index]], dtype=torch.int32),
        torch.tensor([[1.0]], dtype=torch.float32),
    )


def _tiny_transformer():
    transformer = AtomicV3FeatureTransformer.__new__(AtomicV3FeatureTransformer)
    torch.nn.Module.__init__(transformer)
    transformer.bias = torch.nn.Parameter(torch.zeros(1024))
    transformer.hm_bucket_weight = torch.nn.Parameter(torch.zeros((1, 1032)))
    transformer.hm_virtual_weight = torch.nn.Parameter(torch.zeros((1, 1032)))
    transformer.capture_pair_weight = torch.nn.Parameter(torch.zeros((1, 1024)))
    transformer.king_blast_ep_weight = torch.nn.Parameter(torch.zeros((1, 1024)))
    transformer.blast_ring_weight = torch.nn.Parameter(torch.zeros((1, 1024)))
    return transformer


def _tiny_model():
    model = AtomicNNUEV3.__new__(AtomicNNUEV3)
    torch.nn.Module.__init__(model)
    model.feature_transformer = _tiny_transformer()
    model.network = AtomicV3LayerStacks()
    return model


def _fc0_serialized_integers(model, dtype):
    base = model.network.fc0.linear.bias.to(dtype)
    factor = model.network.fc0.factorized_linear.bias.repeat(8).to(dtype)
    rounded = torch.round((base + factor) * FC0_BIAS_SCALE).to(torch.float64)
    if torch.any(rounded < -(1 << 31)) or torch.any(rounded > (1 << 31) - 1):
        raise OverflowError("FC0 bias is outside signed-i32 wire range")
    return rounded.to(torch.int64)


def _assert_fc0_serializable_in_both_widths(model):
    for dtype in (torch.float32, torch.float64):
        integers = _fc0_serialized_integers(model, dtype)
        assert torch.all(integers >= -(1 << 31))
        assert torch.all(integers <= (1 << 31) - 1)


def test_v3_composed_transformer_factorizes_hm_and_gives_relations_no_psqt():
    transformer = _tiny_transformer()
    with torch.no_grad():
        transformer.bias.fill_(0.25)
        transformer.hm_bucket_weight[:, :1024].fill_(0.50)
        transformer.hm_bucket_weight[:, 1024:].fill_(0.10)
        transformer.hm_virtual_weight[:, :1024].fill_(0.25)
        transformer.hm_virtual_weight[:, 1024:].fill_(0.20)
        transformer.capture_pair_weight.fill_(0.10)
        transformer.king_blast_ep_weight.fill_(-0.20)
        transformer.blast_ring_weight.fill_(0.30)
    perspective = SimpleNamespace(
        hm=_slice(), capture_pair=_slice(), king_blast_ep=_slice(), blast_ring=_slice()
    )

    main, psqt = transformer.forward(perspective, fake_quantize_weights=False)

    torch.testing.assert_close(main, torch.full((1, 1024), 1.20))
    torch.testing.assert_close(psqt, torch.full((1, 8), 0.30))
    assert transformer.capture_pair_weight.shape[1] == 1024
    assert transformer.king_blast_ep_weight.shape[1] == 1024
    assert transformer.blast_ring_weight.shape[1] == 1024
    assert transformer.hm_bucket_weight.shape[1] == 1032


def test_factorized_and_relation_tables_keep_sparse_active_row_gradients():
    transformer = _tiny_transformer()
    perspective = SimpleNamespace(
        hm=_slice(), capture_pair=_slice(), king_blast_ep=_slice(), blast_ring=_slice()
    )
    main, psqt = transformer.forward(perspective, fake_quantize_weights=True)
    (main.sum() + psqt.sum()).backward()

    for parameter in (
        transformer.hm_bucket_weight,
        transformer.hm_virtual_weight,
        transformer.capture_pair_weight,
        transformer.king_blast_ep_weight,
        transformer.blast_ring_weight,
    ):
        assert parameter.grad is not None
        assert parameter.grad.is_sparse
        assert parameter.grad.coalesce().indices().shape[1] == 1


def test_mixed_feature_quantizers_use_i8_only_for_cp_and_ring():
    values = torch.tensor([-200.0, -128.0 / 256.0, 127.0 / 256.0, 200.0])
    i8 = fake_quantize_i8_feature(values)
    i16 = fake_quantize_i16_feature(values)
    assert i8.tolist() == pytest.approx([-0.5, -0.5, 127.0 / 256.0, 127.0 / 256.0])
    assert i16[0] == pytest.approx(-128.0)
    assert i16[-1] == pytest.approx(32767.0 / 256.0)


def test_clip_weights_makes_mixed_tables_and_factor_sums_exportable():
    model = _tiny_model()
    transformer = model.feature_transformer
    with torch.no_grad():
        transformer.bias[0:2] = torch.tensor([1.0e9, -1.0e9])
        transformer.capture_pair_weight[0, 0:2] = torch.tensor([1.0e9, -1.0e9])
        transformer.king_blast_ep_weight[0, 0:2] = torch.tensor([1.0e9, -1.0e9])
        transformer.blast_ring_weight[0, 0:2] = torch.tensor([1.0e9, -1.0e9])
        transformer.hm_virtual_weight[0, 0:2] = torch.tensor([1.0e9, -1.0e9])
        transformer.hm_bucket_weight[0, 0:2] = torch.tensor([1.0e9, -1.0e9])
        transformer.hm_virtual_weight[0, 1024:1026] = torch.tensor([1.0e9, -1.0e9])
        transformer.hm_bucket_weight[0, 1024:1026] = torch.tensor([1.0e9, -1.0e9])
        model.network.fc0.factorized_linear.weight[0, 0:2] = torch.tensor(
            [1.0e9, -1.0e9]
        )
        model.network.fc0.linear.weight[:, 0:2] = torch.tensor([1.0e9, -1.0e9])
        model.network.fc0.factorized_linear.bias[0:2] = torch.tensor([1.0e30, -1.0e30])
        for bucket in range(8):
            offset = bucket * 32
            model.network.fc0.linear.bias[offset : offset + 2] = torch.tensor(
                [1.0e30, -1.0e30]
            )
            model.network.fc1.linear.bias[offset : offset + 2] = torch.tensor(
                [1.0e30, -1.0e30]
            )
        model.network.fc2.linear.bias[0:2] = torch.tensor([1.0e30, -1.0e30])

    model.clip_weights()

    assert transformer.bias[0].item() == 32767.0 / FT_ONE
    assert transformer.bias[1].item() == -32768.0 / FT_ONE
    for table in (transformer.capture_pair_weight, transformer.blast_ring_weight):
        assert table[0, 0].item() == 127.0 / FT_ONE
        assert table[0, 1].item() == -128.0 / FT_ONE
    assert transformer.king_blast_ep_weight[0, 0].item() == 32767.0 / FT_ONE
    assert transformer.king_blast_ep_weight[0, 1].item() == -32768.0 / FT_ONE

    hm_main = transformer.hm_bucket_weight[:, :1024] + transformer.hm_virtual_weight[:, :1024]
    assert torch.all(hm_main >= -32768.0 / FT_ONE)
    assert torch.all(hm_main <= 32767.0 / FT_ONE)
    hm_psqt = transformer.hm_bucket_weight[:, 1024:] + transformer.hm_virtual_weight[:, 1024:]
    assert torch.all(hm_psqt >= PSQT_EXPORT_MIN)
    assert torch.all(hm_psqt <= PSQT_EXPORT_MAX)
    scaled_psqt = torch.round(hm_psqt.to(torch.float64) * PSQT_WEIGHT_SCALE)
    assert torch.all(scaled_psqt >= -(1 << 31))
    assert torch.all(scaled_psqt <= (1 << 31) - 1)

    fc0_factor = model.network.fc0.factorized_linear.weight
    fc0_export = model.network.fc0.linear.weight + fc0_factor.repeat(8, 1)
    fc0_limit = 127.0 / FC0_WEIGHT_SCALE
    assert torch.all(fc0_factor >= -fc0_limit)
    assert torch.all(fc0_factor <= fc0_limit)
    assert torch.all(fc0_export >= -fc0_limit)
    assert torch.all(fc0_export <= fc0_limit)

    _assert_fc0_serializable_in_both_widths(model)
    for dtype in (torch.float32, torch.float64):
        fc0_bias = model.network.fc0.linear.bias.to(
            dtype
        ) + model.network.fc0.factorized_linear.bias.repeat(8).to(dtype)
        assert torch.all(fc0_bias >= FC0_BIAS_EXPORT_MIN)
        assert torch.all(fc0_bias <= FC0_BIAS_EXPORT_MAX)

    bias_cases = (
        (
            model.network.fc1.linear.bias,
            FC1_BIAS_SCALE,
            FC1_BIAS_EXPORT_MIN,
            FC1_BIAS_EXPORT_MAX,
        ),
        (
            model.network.fc2.linear.bias,
            FC2_BIAS_SCALE,
            FC2_BIAS_EXPORT_MIN,
            FC2_BIAS_EXPORT_MAX,
        ),
    )
    for bias, scale, minimum, maximum in bias_cases:
        assert torch.all(bias >= minimum)
        assert torch.all(bias <= maximum)
        for dtype in (torch.float32, torch.float64):
            integers = torch.round(bias.to(dtype) * scale).to(torch.float64)
            assert torch.all(integers >= -(1 << 31))
            assert torch.all(integers <= (1 << 31) - 1)


@pytest.mark.parametrize("scale", [FC0_BIAS_SCALE, FC1_BIAS_SCALE, FC2_BIAS_SCALE])
def test_dense_bias_fake_quantizer_is_inward_i32_safe_for_extreme_float32(scale):
    values = torch.tensor([-torch.finfo(torch.float32).max, torch.finfo(torch.float32).max])
    quantized = fake_quantize_integer_bias(values, scale)
    assert torch.all(torch.isfinite(quantized))
    for dtype in (torch.float32, torch.float64):
        integers = torch.round(quantized.to(dtype) * scale).to(torch.float64)
        assert torch.all(integers >= -(1 << 31))
        assert torch.all(integers <= (1 << 31) - 1)


def test_fc0_opposite_sign_half_ulp_regressions_are_clipped_before_serialization():
    model = _tiny_model()
    with torch.no_grad():
        # float32 coalescing rounds this exact safe sum outward to 2**31.
        model.network.fc0.linear.bias[0] = 131072.0
        model.network.fc0.factorized_linear.bias[0] = -1.0 / 256.0
        # float32 hides this half-ulp, but float64 exposes -2147483776.
        model.network.fc0.linear.bias[1] = -131072.015625
        model.network.fc0.factorized_linear.bias[1] = 1.0 / 128.0

    base = model.network.fc0.linear.bias
    factor = model.network.fc0.factorized_linear.bias.repeat(8)
    before32 = torch.round((base + factor) * FC0_BIAS_SCALE)
    before64 = torch.round(
        (base.to(torch.float64) + factor.to(torch.float64)) * FC0_BIAS_SCALE
    )
    assert before32[0].item() == 2147483648
    assert before64[1].item() == -2147483776
    with pytest.raises(OverflowError):
        _fc0_serialized_integers(model, torch.float32)
    with pytest.raises(OverflowError):
        _fc0_serialized_integers(model, torch.float64)

    model.clip_weights()
    _assert_fc0_serializable_in_both_widths(model)


def test_fc0_coordinated_clamp_fuzzes_nextafter_edges_in_both_widths():
    model = _tiny_model()
    generator = torch.Generator().manual_seed(0xA70C)
    with torch.no_grad():
        model.network.fc0.linear.bias.uniform_(-1.0e9, 1.0e9, generator=generator)
        model.network.fc0.factorized_linear.bias.uniform_(
            -1.0e9, 1.0e9, generator=generator
        )
        lower = torch.tensor(FC0_BIAS_EXPORT_MIN, dtype=torch.float32)
        upper = torch.tensor(FC0_BIAS_EXPORT_MAX, dtype=torch.float32)
        toward_negative = torch.tensor(float("-inf"), dtype=torch.float32)
        toward_positive = torch.tensor(float("inf"), dtype=torch.float32)
        edge_values = []
        for edge in (lower, upper):
            edge_values.extend(
                (
                    torch.nextafter(edge, toward_negative),
                    edge,
                    torch.nextafter(edge, toward_positive),
                )
            )
        model.network.fc0.linear.bias[: len(edge_values)] = torch.stack(edge_values)
        model.network.fc0.factorized_linear.bias[: len(edge_values)] = -torch.stack(
            edge_values
        )

    model.clip_weights()
    _assert_fc0_serializable_in_both_widths(model)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_clip_weights_rejects_nonfinite_dense_biases(nonfinite):
    model = _tiny_model()
    with torch.no_grad():
        model.network.fc1.linear.bias[0] = nonfinite
    with pytest.raises(FloatingPointError, match="non-finite"):
        model.clip_weights()


def test_psqt_fake_quantizer_stays_inside_i32_after_adversarial_float32_input():
    values = torch.tensor([[-1.0e30, 1.0e30]], dtype=torch.float32)
    quantized = fake_quantize_psqt_weight(values)
    integers = torch.round(quantized.to(torch.float64) * PSQT_WEIGHT_SCALE)
    assert torch.all(integers >= -(1 << 31))
    assert torch.all(integers <= (1 << 31) - 1)


def test_optimizer_step_clips_both_the_forward_boundary_and_persistent_state(monkeypatch):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0]))
            self.clip_calls = 0

        @torch.no_grad()
        def clip_weights(self):
            self.clip_calls += 1
            self.weight.clamp_(-1.0, 1.0)

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=10.0)
    monkeypatch.setattr(
        v3_training,
        "batch_loss",
        lambda candidate, batch, lambda_=0.5: candidate.weight.square().sum(),
    )
    reported = optimizer_step(model, optimizer, object())
    assert reported == pytest.approx(1.0)
    assert model.clip_calls == 2
    assert model.weight.item() == -1.0


def test_optimizer_step_keeps_all_dense_biases_i32_safe_before_and_after_update(monkeypatch):
    model = _tiny_model()
    with torch.no_grad():
        model.network.fc0.linear.bias[0:2] = torch.tensor([1.0e30, -1.0e30])
        model.network.fc0.factorized_linear.bias[0:2] = torch.tensor([1.0e30, -1.0e30])
        model.network.fc0.linear.bias[2:4] = torch.tensor(
            [131072.0, -131072.015625]
        )
        model.network.fc0.factorized_linear.bias[2:4] = torch.tensor(
            [-1.0 / 256.0, 1.0 / 128.0]
        )
        model.network.fc1.linear.bias[0:2] = torch.tensor([1.0e30, -1.0e30])
        model.network.fc2.linear.bias[0:2] = torch.tensor([1.0e30, -1.0e30])

    raw_base = model.network.fc0.linear.bias
    raw_factor = model.network.fc0.factorized_linear.bias.repeat(8)
    assert torch.round((raw_base + raw_factor) * FC0_BIAS_SCALE)[2].item() == 2147483648
    assert torch.round(
        (raw_base.to(torch.float64) + raw_factor.to(torch.float64)) * FC0_BIAS_SCALE
    )[3].item() == -2147483776

    observed_forward = []

    def assert_exportable(candidate):
        _assert_fc0_serializable_in_both_widths(candidate)
        cases = (
            (candidate.network.fc1.linear.bias, FC1_BIAS_SCALE),
            (candidate.network.fc2.linear.bias, FC2_BIAS_SCALE),
        )
        for bias, scale in cases:
            integers = torch.round(bias.to(torch.float64) * scale)
            assert torch.all(integers >= -(1 << 31))
            assert torch.all(integers <= (1 << 31) - 1)

    def adversarial_loss(candidate, batch, lambda_=0.5):
        assert_exportable(candidate)
        observed_forward.append(True)
        positive = (
            candidate.network.fc0.linear.bias[0]
            + candidate.network.fc0.factorized_linear.bias[0]
            + candidate.network.fc1.linear.bias[0]
            + candidate.network.fc2.linear.bias[0]
        )
        negative = (
            candidate.network.fc0.linear.bias[1]
            + candidate.network.fc0.factorized_linear.bias[1]
            + candidate.network.fc1.linear.bias[1]
            + candidate.network.fc2.linear.bias[1]
        )
        return positive - negative

    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e9)
    monkeypatch.setattr(v3_training, "batch_loss", adversarial_loss)
    optimizer_step(model, optimizer, object())

    assert observed_forward == [True]
    assert_exportable(model)
    merged = model.network.fc0.linear.bias + model.network.fc0.factorized_linear.bias.repeat(8)
    assert merged[0] < 0.0 and merged[1] > 0.0
    assert model.network.fc1.linear.bias[0] < 0.0
    assert model.network.fc1.linear.bias[1] > 0.0
    assert model.network.fc2.linear.bias[0] < 0.0
    assert model.network.fc2.linear.bias[1] > 0.0


def test_sfnnv15_dense_tail_shapes_and_short_skip_are_frozen():
    stacks = AtomicV3LayerStacks()
    assert tuple(stacks.fc0.linear.weight.shape) == (8 * 32, 1024)
    assert tuple(stacks.fc0.factorized_linear.weight.shape) == (32, 1024)
    assert tuple(stacks.fc1.linear.weight.shape) == (8 * 32, 64)
    assert tuple(stacks.fc2.linear.weight.shape) == (8, 128)

    with torch.no_grad():
        for parameter in stacks.parameters():
            parameter.zero_()
        for bucket in range(8):
            stacks.fc0.linear.bias[bucket * 32 + 30] = 3.0
            stacks.fc0.linear.bias[bucket * 32 + 31] = 1.0
    output = stacks(
        torch.zeros((3, 1024)),
        torch.tensor([0, 3, 7]),
        fake_quantize_activations=False,
        fake_quantize_weights=False,
    )
    torch.testing.assert_close(output, torch.full((3, 1), 2.0))


def test_signed_psqt_half_uses_cpp_truncation_and_keeps_ste_gradient():
    raw = torch.tensor([[-3.0], [-1.0], [1.0], [3.0]], requires_grad=True)
    output = fake_quantize_psqt_output(raw / (2.0 * PSQT_WEIGHT_SCALE))
    torch.testing.assert_close(
        output * PSQT_WEIGHT_SCALE,
        torch.tensor([[-1.0], [0.0], [0.0], [1.0]]),
    )
    output.sum().backward()
    assert torch.all(torch.isfinite(raw.grad))


def test_atomic_loss_rejects_ambiguous_shapes_and_nonfinite_is_not_hidden():
    with pytest.raises(ValueError, match="equal shape"):
        atomic_loss(torch.zeros((2, 1)), torch.zeros((1, 1)), torch.zeros((2, 1)))
    value = atomic_loss(
        torch.tensor([[float("nan")]]), torch.tensor([[0.5]]), torch.tensor([[0.0]])
    )
    assert torch.isnan(value)


def test_production_shape_cpu_one_step_is_finite_and_deterministic():
    # The fixture loader checks train and validation independently before the
    # production-size model is allocated.
    fixture = load_canonical_fixture()
    assert fixture.batch("train").batch_size == 2
    assert fixture.batch("validation").batch_size == 1

    first = deterministic_cpu_one_step(seed=20260716)
    second = deterministic_cpu_one_step(seed=20260716)

    assert first == second
    assert first.train_loss_after < first.train_loss_before
    assert first.validation_loss_before >= 0.0
    assert first.validation_loss_after >= 0.0
    assert len(first.parameter_sha256) == 64


@pytest.mark.parametrize("learning_rate", [float("nan"), float("inf"), -float("inf"), 0.0])
def test_core_optimizer_rejects_nonfinite_or_nonpositive_learning_rate(learning_rate):
    model = _tiny_model()
    with pytest.raises(ValueError, match="finite and positive"):
        create_core_optimizer(model, learning_rate)


@pytest.mark.parametrize("learning_rate", [True, "0.001"])
def test_core_optimizer_rejects_non_real_learning_rate(learning_rate):
    model = _tiny_model()
    with pytest.raises(TypeError, match="real number"):
        create_core_optimizer(model, learning_rate)


@pytest.mark.parametrize("seed", [True, 1.5, "1"])
def test_deterministic_step_rejects_non_integer_seed_before_allocating(seed):
    with pytest.raises(TypeError, match="seed must be an integer"):
        deterministic_cpu_one_step(seed=seed)


@pytest.mark.parametrize("seed", [-1, 1 << 64])
def test_deterministic_step_rejects_seed_outside_nonnegative_uint64(seed):
    with pytest.raises(ValueError, match="non-negative uint64"):
        deterministic_cpu_one_step(seed=seed)
