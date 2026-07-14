import torch

from atomic_v2.contract import FEATURE_DIMENSIONS, FEATURE_NAME
from atomic_v2.model import (
    AtomicLayerStacks,
    AtomicNNUEV2,
    SparseFeatureTransformer,
    pairwise_multiply,
)
from atomic_v2.quantization import PSQT_WEIGHT_SCALE, fake_quantize_psqt_output


def test_sfnnv15_layer_stack_has_the_physical_v2_shapes():
    stacks = AtomicLayerStacks()

    assert tuple(stacks.fc0.linear.weight.shape) == (8 * 32, 1024)
    assert tuple(stacks.fc0.factorized_linear.weight.shape) == (32, 1024)
    assert tuple(stacks.fc1.linear.weight.shape) == (8 * 32, 64)
    assert tuple(stacks.fc2.linear.weight.shape) == (8, 128)


def test_sfnnv15_short_skip_is_fc0_minus_two_minus_minus_one():
    stacks = AtomicLayerStacks()
    with torch.no_grad():
        for parameter in stacks.parameters():
            parameter.zero_()
        for bucket in range(8):
            offset = bucket * 32
            stacks.fc0.linear.bias[offset + 30] = 3.0
            stacks.fc0.linear.bias[offset + 31] = 1.0

    output = stacks(
        torch.zeros((3, 1024), dtype=torch.float32),
        torch.tensor([0, 3, 7], dtype=torch.long),
        fake_quantize_activations=False,
        fake_quantize_weights=False,
    )

    torch.testing.assert_close(output, torch.full((3, 1), 2.0))


def test_pairwise_multiply_is_per_perspective_then_concatenated():
    x = torch.zeros((1, 2048), dtype=torch.float32)
    x[:, :512] = 2.0
    x[:, 512:1024] = 3.0
    x[:, 1024:1536] = 5.0
    x[:, 1536:] = 7.0

    output = pairwise_multiply(x)

    assert tuple(output.shape) == (1, 1024)
    torch.testing.assert_close(output[:, :512], torch.full((1, 512), 6.0))
    torch.testing.assert_close(output[:, 512:], torch.full((1, 512), 35.0))


def test_hidden_weight_clipping_applies_to_coalesced_fc0_weights():
    model = AtomicNNUEV2.__new__(AtomicNNUEV2)
    torch.nn.Module.__init__(model)
    model.network = AtomicLayerStacks()
    with torch.no_grad():
        model.network.fc0.factorized_linear.weight.fill_(0.75)
        model.network.fc0.linear.weight.fill_(2.0)
        model.network.fc1.linear.weight.fill_(-3.0)
        model.network.fc2.linear.weight.fill_(3.0)

    model.clip_weights()

    merged, _ = model.network.fc0._merged_parameters()
    assert float(merged.max()) <= 127.0 / 128.0
    assert float(merged.min()) >= -127.0 / 128.0
    assert float(model.network.fc1.linear.weight.min()) >= -127.0 / 64.0
    assert float(model.network.fc2.linear.weight.max()) <= 127.0 / 128.0


def test_production_model_exposes_only_halfkav2atomic(monkeypatch):
    monkeypatch.setattr(AtomicNNUEV2, "_allocate_feature_parameters", lambda self: None)
    model = AtomicNNUEV2.__new__(AtomicNNUEV2)

    assert model.feature_name == FEATURE_NAME == "HalfKAv2Atomic"
    assert model.num_features == FEATURE_DIMENSIONS == 45056
    assert "Threat" not in model.feature_name


def test_feature_transformer_quantizes_once_for_both_perspectives(monkeypatch):
    transformer = SparseFeatureTransformer.__new__(SparseFeatureTransformer)
    torch.nn.Module.__init__(transformer)
    transformer.weight = torch.nn.Parameter(torch.zeros((2, 1032), dtype=torch.float32))
    transformer.bias = torch.nn.Parameter(torch.zeros(1024, dtype=torch.float32))
    calls = []
    original = transformer._merged_parameters

    def counted(fake_quantize_weights):
        calls.append(fake_quantize_weights)
        return original(fake_quantize_weights)

    monkeypatch.setattr(transformer, "_merged_parameters", counted)
    white_indices = torch.tensor([[0, -1]], dtype=torch.int32)
    black_indices = torch.tensor([[1, -1]], dtype=torch.int32)
    values = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    white, black = transformer.forward_pair(
        white_indices,
        values,
        black_indices,
        values,
        True,
    )

    assert calls == [True]
    assert white.shape == black.shape == (1, 1032)


def test_psqt_signed_halves_truncate_toward_zero_with_ste_gradient():
    raw_halves = torch.tensor(
        [[0.5], [-0.5], [1.0], [-1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    output = fake_quantize_psqt_output(raw_halves / PSQT_WEIGHT_SCALE)

    torch.testing.assert_close(
        output * PSQT_WEIGHT_SCALE,
        torch.tensor([[0.0], [0.0], [1.0], [-1.0]]),
    )
    output.sum().backward()
    torch.testing.assert_close(
        raw_halves.grad,
        torch.full_like(raw_halves, 1.0 / PSQT_WEIGHT_SCALE),
    )


def test_psqt_float32_rounding_matches_cpp_for_signed_delta_sweep():
    delta = torch.arange(-10_000, 10_001, dtype=torch.int64)
    soft_half = delta.to(torch.float32) / PSQT_WEIGHT_SCALE * 0.5

    output = fake_quantize_psqt_output(soft_half)
    actual = torch.round(output * PSQT_WEIGHT_SCALE).to(torch.int64)
    expected = torch.div(delta, 2, rounding_mode="trunc")

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
