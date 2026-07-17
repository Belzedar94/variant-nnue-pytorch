import copy
from types import SimpleNamespace

import pytest
import torch

import atomic_v3.training as v3_training
from atomic_v3.dataset import SparseSliceBatch, load_canonical_fixture
from atomic_v3.dense import AtomicV3LayerStacks, fake_quantize_integer_bias
from atomic_v3.executor import (
    PersistenceFiniteStateError,
    audit_persistence_finite_state,
    create_production_optimizer,
)
from atomic_v3.model import (
    AtomicNNUEV3,
    AtomicV3FeatureTransformer,
    _clip_factorized_i32_sums_,
)
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
    INT32_MAX,
    INT32_MIN,
    PSQT_EXPORT_MAX,
    PSQT_EXPORT_MIN,
    PSQT_WEIGHT_SCALE,
    fake_quantize_i16_feature,
    fake_quantize_i8_feature,
    fake_quantize_psqt_output,
    fake_quantize_psqt_weight,
)
from atomic_v3.serialization import _quantized_numpy
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


def _tiny_forward_batch(device):
    active = SparseSliceBatch(
        torch.tensor([[0]], dtype=torch.int32, device=device),
        torch.tensor([[1.0]], dtype=torch.float32, device=device),
    )
    perspective = SimpleNamespace(
        hm=active,
        capture_pair=active,
        king_blast_ep=active,
        blast_ring=active,
    )
    return SimpleNamespace(
        white=perspective,
        black=perspective,
        side_to_move_white=torch.ones((1, 1), dtype=torch.float32, device=device),
        bucket_indices=torch.zeros(1, dtype=torch.long, device=device),
    )


def _rng_snapshot():
    cpu = torch.random.get_rng_state().clone()
    cuda = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_initialized()
        else None
    )
    return cpu, cuda


def _assert_rng_snapshot_equal(expected):
    cpu, cuda = _rng_snapshot()
    assert torch.equal(cpu, expected[0])
    if expected[1] is None:
        assert cuda is None
    else:
        assert cuda is not None and len(cuda) == len(expected[1])
        assert all(torch.equal(actual, saved) for actual, saved in zip(cuda, expected[1]))


def _factorized_i32_serialized_integers(base, factor, dtype, scale, label):
    """Future serializer boundary: range-check before the signed-i32 cast."""

    merged = base.to(dtype) + factor.to(dtype)
    # The mixed-width serializer widens the persisted sum before applying the
    # integer scale; it must reject overflow before any signed cast.
    rounded = torch.round(merged.to(torch.float64) * float(scale))
    if not torch.all(torch.isfinite(rounded)) or torch.any(rounded < INT32_MIN) or torch.any(
        rounded > INT32_MAX
    ):
        raise OverflowError(f"{label} is outside signed-i32 wire range")
    return rounded.to(torch.int64)


def _fc0_serialized_integers(model, dtype):
    return _factorized_i32_serialized_integers(
        model.network.fc0.linear.bias,
        model.network.fc0.factorized_linear.bias.repeat(8),
        dtype,
        FC0_BIAS_SCALE,
        "FC0 bias",
    )


def _assert_fc0_serializable_in_both_widths(model):
    for dtype in (torch.float32, torch.float64):
        integers = _fc0_serialized_integers(model, dtype)
        assert torch.all(integers >= -(1 << 31))
        assert torch.all(integers <= (1 << 31) - 1)


def _hm_psqt_serialized_integers(transformer, dtype):
    base = transformer.hm_bucket_weight[:, 1024:]
    factor = transformer.hm_virtual_weight[:, 1024:]
    if base.shape[0] % factor.shape[0] != 0:
        raise ValueError("test HM factor shapes are inconsistent")
    return _factorized_i32_serialized_integers(
        base,
        factor.repeat(base.shape[0] // factor.shape[0], 1),
        dtype,
        PSQT_WEIGHT_SCALE,
        "HM PSQT weight",
    )


def _assert_hm_psqt_serializable_in_both_widths(transformer):
    for dtype in (torch.float32, torch.float64):
        integers = _hm_psqt_serialized_integers(transformer, dtype)
        assert torch.all(integers >= INT32_MIN)
        assert torch.all(integers <= INT32_MAX)


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


def test_factorized_and_relation_tables_produce_dense_gradients():
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
        assert parameter.grad.is_sparse is False
        assert parameter.grad.shape == parameter.shape
        assert torch.all(torch.isfinite(parameter.grad))
        assert torch.count_nonzero(parameter.grad) > 0


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
    _assert_hm_psqt_serializable_in_both_widths(transformer)

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


def test_training_clip_avoids_reductions_and_keeps_all_coalesced_ranges(monkeypatch):
    model = _tiny_model()
    transformer = model.feature_transformer
    with torch.no_grad():
        for parameter in transformer.parameters():
            parameter.fill_(1.0e9)
        for parameter in model.network.parameters():
            parameter.fill_(-1.0e9)

    def forbidden_reduction(*_args, **_kwargs):
        raise AssertionError("per-step clipping must not synchronize through torch.all")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "all", forbidden_reduction)
        model.clip_training_weights()

    hm = transformer.hm_bucket_weight + transformer.hm_virtual_weight
    assert torch.all(hm[:, :1024] >= -32768.0 / FT_ONE)
    assert torch.all(hm[:, :1024] <= 32767.0 / FT_ONE)
    assert torch.all(hm[:, 1024:] >= PSQT_EXPORT_MIN)
    assert torch.all(hm[:, 1024:] <= PSQT_EXPORT_MAX)
    fc0_weight = (
        model.network.fc0.linear.weight
        + model.network.fc0.factorized_linear.weight.repeat(8, 1)
    )
    fc0_limit = 127.0 / FC0_WEIGHT_SCALE
    assert torch.all(fc0_weight >= -fc0_limit)
    assert torch.all(fc0_weight <= fc0_limit)
    fc0_bias = (
        model.network.fc0.linear.bias
        + model.network.fc0.factorized_linear.bias.repeat(8)
    )
    assert torch.all(fc0_bias >= FC0_BIAS_EXPORT_MIN)
    assert torch.all(fc0_bias <= FC0_BIAS_EXPORT_MAX)

    # The strict boundary remains responsible for exact signed-i32 safety.
    model.clip_weights()
    _assert_hm_psqt_serializable_in_both_widths(transformer)
    _assert_fc0_serializable_in_both_widths(model)


def test_training_and_exact_clips_match_for_interior_parameters():
    fast = _tiny_model()
    generator = torch.Generator().manual_seed(0xA70C0004)
    with torch.no_grad():
        for parameter in fast.parameters():
            parameter.uniform_(-0.05, 0.05, generator=generator)
    exact = copy.deepcopy(fast)

    fast.clip_training_weights()
    exact.clip_weights()

    for fast_parameter, exact_parameter in zip(fast.parameters(), exact.parameters()):
        torch.testing.assert_close(fast_parameter, exact_parameter, rtol=0.0, atol=0.0)


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


def test_hm_psqt_opposite_sign_rounding_regressions_are_clipped_before_serialization():
    transformer = _tiny_transformer()
    with torch.no_grad():
        # Both sums are just outside the signed-i32 domain after a float32
        # limit subtraction has rounded the persisted base away from zero.
        transformer.hm_bucket_weight[0, 1024:1026] = torch.tensor(
            [345969.1875, -407320.25]
        )
        transformer.hm_virtual_weight[0, 1024:1026] = torch.tensor(
            [-122272.96875, 183624.03125]
        )

    base = transformer.hm_bucket_weight[:, 1024:]
    factor = transformer.hm_virtual_weight[:, 1024:]
    before32 = torch.round((base + factor) * PSQT_WEIGHT_SCALE)
    before64 = torch.round(
        (base.to(torch.float64) + factor.to(torch.float64)) * PSQT_WEIGHT_SCALE
    )
    assert before32[0, 0].item() == 2147483648
    assert before64[0, 0].item() == 2147483700
    assert before64[0, 1].item() == -2147483700
    with pytest.raises(OverflowError):
        _hm_psqt_serialized_integers(transformer, torch.float32)
    with pytest.raises(OverflowError):
        _hm_psqt_serialized_integers(transformer, torch.float64)

    transformer.clip_weights()
    _assert_hm_psqt_serializable_in_both_widths(transformer)


def test_hm_psqt_coordinated_clamp_randomized_persistent_wire_sweep():
    generator = torch.Generator().manual_seed(0xA70C0003)
    base = torch.nn.Parameter(torch.empty((32768, 8), dtype=torch.float32))
    factor = torch.nn.Parameter(torch.empty((32768, 8), dtype=torch.float32))
    with torch.no_grad():
        base.uniform_(-1.0e9, 1.0e9, generator=generator)
        factor.uniform_(PSQT_EXPORT_MIN, PSQT_EXPORT_MAX, generator=generator)
        base[0, :2] = torch.tensor([345969.1875, -407320.25])
        factor[0, :2] = torch.tensor([-122272.96875, 183624.03125])

    original_device = base.device
    original_dtype = base.dtype
    _clip_factorized_i32_sums_(
        base,
        factor,
        scale=PSQT_WEIGHT_SCALE,
        minimum=PSQT_EXPORT_MIN,
        maximum=PSQT_EXPORT_MAX,
        label="HM PSQT randomized test",
    )

    assert base.device == factor.device == original_device
    assert base.dtype == factor.dtype == original_dtype
    assert base.requires_grad and factor.requires_grad
    assert base.grad_fn is None and factor.grad_fn is None
    for dtype in (torch.float32, torch.float64):
        integers = _factorized_i32_serialized_integers(
            base, factor, dtype, PSQT_WEIGHT_SCALE, "HM PSQT randomized test"
        )
        assert torch.all(integers >= INT32_MIN)
        assert torch.all(integers <= INT32_MAX)


@pytest.mark.parametrize("parameter", ["bucket", "factor"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_hm_psqt_factorized_clamp_rejects_nonfinite_operands(parameter, nonfinite):
    transformer = _tiny_transformer()
    with torch.no_grad():
        target = (
            transformer.hm_bucket_weight
            if parameter == "bucket"
            else transformer.hm_virtual_weight
        )
        target[0, 1024] = nonfinite
    with pytest.raises(FloatingPointError, match="HM PSQT factor is non-finite"):
        transformer.clip_weights()


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_clip_weights_rejects_nonfinite_dense_biases(nonfinite):
    model = _tiny_model()
    with torch.no_grad():
        model.network.fc1.linear.bias[0] = nonfinite
    with pytest.raises(FloatingPointError, match="non-finite"):
        model.clip_weights()


@pytest.mark.parametrize(
    "parameter_path",
    [
        "feature_transformer.capture_pair_weight",
        "network.fc0.linear.weight",
    ],
)
def test_persistence_audit_rejects_nan_relation_and_dense_weights(parameter_path):
    model = _tiny_model()
    optimizer = create_core_optimizer(model)
    target = model
    for component in parameter_path.split("."):
        target = getattr(target, component)
    with torch.no_grad():
        target.reshape(-1)[0] = float("nan")

    with pytest.raises(PersistenceFiniteStateError, match="non-finite"):
        audit_persistence_finite_state(model, optimizer)


@pytest.mark.parametrize("state_name", ["exp_avg", "exp_avg_sq"])
def test_persistence_audit_rejects_infinite_ranger_moments(state_name):
    model = _tiny_model()
    optimizer, _scheduler = create_production_optimizer(model)
    parameter = next(model.parameters())
    optimizer.state[parameter][state_name] = torch.full_like(parameter, float("inf"))

    with pytest.raises(PersistenceFiniteStateError, match="optimizer state"):
        audit_persistence_finite_state(model, optimizer)


def test_persistence_audit_reduces_to_one_host_check_on_single_device(monkeypatch):
    model = _tiny_model()
    optimizer, _scheduler = create_production_optimizer(model)
    parameter = next(model.parameters())
    optimizer.state[parameter]["nested"] = {
        "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.ones_like(parameter),
    }
    calls = []
    original = torch.Tensor.item

    def item(tensor, *args, **kwargs):
        calls.append(tensor.device)
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", item)
    report = audit_persistence_finite_state(model, optimizer)

    assert report["device_synchronizations"] == 1
    assert calls == [torch.device("cpu")]


def test_psqt_fake_quantizer_stays_inside_i32_after_adversarial_float32_input():
    values = torch.tensor([[-1.0e30, 1.0e30]], dtype=torch.float32)
    quantized = fake_quantize_psqt_weight(values)
    integers = torch.round(quantized.to(torch.float64) * PSQT_WEIGHT_SCALE)
    assert torch.all(integers >= -(1 << 31))
    assert torch.all(integers <= (1 << 31) - 1)


def test_model_psqt_qat_matches_serializer_at_half_steps_and_i32_boundaries():
    transformer = _tiny_transformer()
    half_steps = torch.tensor(
        [
            (16_292.0 + 0.5) / PSQT_WEIGHT_SCALE,
            (-499_998.0 + 0.5) / PSQT_WEIGHT_SCALE,
            (0.0 + 0.5) / PSQT_WEIGHT_SCALE,
            (-1.0 + 0.5) / PSQT_WEIGHT_SCALE,
        ],
        dtype=torch.float32,
    )
    lower = torch.tensor(PSQT_EXPORT_MIN, dtype=torch.float32)
    upper = torch.tensor(PSQT_EXPORT_MAX, dtype=torch.float32)
    values = torch.cat(
        (
            half_steps,
            torch.stack(
                (
                    lower,
                    torch.nextafter(lower, torch.tensor(float("inf"))),
                    torch.nextafter(upper, torch.tensor(float("-inf"))),
                    upper,
                )
            ),
        )
    )
    with torch.no_grad():
        transformer.hm_bucket_weight[0, 1024:] = values
        transformer.hm_virtual_weight.zero_()

    prepared, *_ = transformer._prepared_weights(fake_quantize_weights=True)
    actual = prepared[0, 1024:]
    wire = _quantized_numpy(values, PSQT_WEIGHT_SCALE, "<i4", "PSQT parity")
    expected = torch.from_numpy(wire).to(torch.float64).div(PSQT_WEIGHT_SCALE).to(torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


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


def test_optimizer_step_keeps_all_factorized_i32_sums_safe_before_and_after_update(monkeypatch):
    model = _tiny_model()
    with torch.no_grad():
        model.feature_transformer.hm_bucket_weight[0, 1024:1026] = torch.tensor(
            [345969.1875, -407320.25]
        )
        model.feature_transformer.hm_virtual_weight[0, 1024:1026] = torch.tensor(
            [-122272.96875, 183624.03125]
        )
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
        _assert_hm_psqt_serializable_in_both_widths(candidate.feature_transformer)
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


@pytest.mark.cuda_gate
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Atomic V3 CUDA smoke needs a GPU")
def test_atomic_v3_cuda_clamp_forward_backward_and_optimizer_edges(monkeypatch):
    device = torch.device("cuda")
    model = _tiny_model().to(device)
    batch = _tiny_forward_batch(device)
    with torch.no_grad():
        model.feature_transformer.hm_bucket_weight[0, 1024:1026] = torch.tensor(
            [345969.1875, -407320.25], device=device
        )
        model.feature_transformer.hm_virtual_weight[0, 1024:1026] = torch.tensor(
            [-122272.96875, 183624.03125], device=device
        )
        model.network.fc0.linear.bias[0:2] = torch.tensor(
            [131072.0, -131072.015625], device=device
        )
        model.network.fc0.factorized_linear.bias[0:2] = torch.tensor(
            [-1.0 / 256.0, 1.0 / 128.0], device=device
        )

    observed = []

    def cuda_loss(candidate, ignored_batch, lambda_=0.5):
        _assert_hm_psqt_serializable_in_both_widths(candidate.feature_transformer)
        _assert_fc0_serializable_in_both_widths(candidate)
        output = candidate(batch, validate=False)
        assert output.device.type == "cuda"
        assert torch.all(torch.isfinite(output))
        observed.append(True)
        return output.square().mean()

    monkeypatch.setattr(v3_training, "batch_loss", cuda_loss)
    optimizer = create_core_optimizer(model, learning_rate=1.0e-3)
    reported = optimizer_step(model, optimizer, batch)
    torch.cuda.synchronize()

    assert torch.isfinite(torch.tensor(reported))
    assert observed == [True]
    _assert_hm_psqt_serializable_in_both_widths(model.feature_transformer)
    _assert_fc0_serializable_in_both_widths(model)
    assert model.feature_transformer.hm_bucket_weight.grad is not None
    assert model.feature_transformer.hm_bucket_weight.grad.is_sparse is False
    assert torch.all(torch.isfinite(model.feature_transformer.hm_bucket_weight.grad))
    del optimizer, batch, model
    torch.cuda.empty_cache()


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


def test_sfnnv15_dense_tail_rejects_invalid_bucket_for_direct_callers():
    stacks = AtomicV3LayerStacks()
    with pytest.raises(ValueError, match="layer-stack index outside"):
        stacks(
            torch.zeros((1, 1024)),
            torch.tensor([8]),
            fake_quantize_activations=False,
            fake_quantize_weights=False,
        )


def test_model_trusted_hot_path_does_not_execute_bucket_host_predicate(monkeypatch):
    model = _tiny_model()
    batch = _tiny_forward_batch("cpu")

    def forbidden_any(*args, **kwargs):
        raise AssertionError("trusted Atomic V3 hot path called torch.any")

    monkeypatch.setattr(torch, "any", forbidden_any)
    output = model(batch, validate=False)
    assert output.shape == (1, 1)
    assert torch.all(torch.isfinite(output))


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

    rng_before = _rng_snapshot()
    first = deterministic_cpu_one_step(seed=20260716)
    _assert_rng_snapshot_equal(rng_before)
    second = deterministic_cpu_one_step(seed=20260716)
    _assert_rng_snapshot_equal(rng_before)

    assert first == second
    assert first.train_loss_after < first.train_loss_before
    assert first.validation_loss_before >= 0.0
    assert first.validation_loss_after >= 0.0
    assert len(first.parameter_sha256) == 64


def test_deterministic_cpu_step_restores_cpu_rng_when_fixture_raises(monkeypatch):
    rng_before = _rng_snapshot()
    monkeypatch.setattr(
        v3_training,
        "load_canonical_fixture",
        lambda: (_ for _ in ()).throw(RuntimeError("fixture failure")),
    )
    with pytest.raises(RuntimeError, match="fixture failure"):
        deterministic_cpu_one_step(seed=20260716)
    _assert_rng_snapshot_equal(rng_before)


def test_deterministic_cpu_step_restores_warn_only_mode_when_fixture_raises(monkeypatch):
    threads_before = torch.get_num_threads()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setattr(
        v3_training,
        "load_canonical_fixture",
        lambda: (_ for _ in ()).throw(RuntimeError("fixture failure")),
    )
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.raises(RuntimeError, match="fixture failure"):
            deterministic_cpu_one_step(seed=20260716)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.get_num_threads() == threads_before
    finally:
        torch.use_deterministic_algorithms(
            deterministic_before, warn_only=warn_only_before
        )


def test_deterministic_cpu_step_does_not_initialize_cuda_for_healthcheck(monkeypatch):
    cpu_before = torch.random.get_rng_state().clone()
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU healthcheck touched an uninitialized CUDA RNG")

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", forbidden)
    monkeypatch.setattr(
        v3_training,
        "load_canonical_fixture",
        lambda: (_ for _ in ()).throw(RuntimeError("fixture failure")),
    )
    with pytest.raises(RuntimeError, match="fixture failure"):
        deterministic_cpu_one_step(seed=20260716)
    assert torch.equal(torch.random.get_rng_state(), cpu_before)


@pytest.mark.cuda_gate
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA RNG restore needs a GPU")
def test_deterministic_cpu_step_restores_initialized_cuda_rng_on_exception(monkeypatch):
    torch.cuda.manual_seed_all(0xA70C0003)
    rng_before = _rng_snapshot()
    monkeypatch.setattr(
        v3_training,
        "load_canonical_fixture",
        lambda: (_ for _ in ()).throw(RuntimeError("fixture failure")),
    )
    with pytest.raises(RuntimeError, match="fixture failure"):
        deterministic_cpu_one_step(seed=20260716)
    _assert_rng_snapshot_equal(rng_before)


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
