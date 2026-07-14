import io
import gc
import hashlib
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atomic_v2.contract import FILE_VERSION, NETWORK_HASH
from atomic_v2.model import AtomicLayerStacks, AtomicNNUEV2, pairwise_multiply
from atomic_v2.quantization import (
    FC0_BIAS_SCALE,
    FC0_WEIGHT_SCALE,
    FC1_BIAS_SCALE,
    FC1_WEIGHT_SCALE,
    FC2_BIAS_SCALE,
    FC2_WEIGHT_SCALE,
    FT_ONE,
    HIDDEN_ONE,
    PSQT_WEIGHT_SCALE,
    clip_feature_activation,
    clip_hidden_activation,
    fake_quantize_activation,
    fake_quantize_output,
)
from atomic_v2.serialization import (
    LEB128_MAGIC,
    AtomicV2FormatError,
    decode_signed_leb128,
    encode_signed_leb128,
    read_compressed_array,
    read_header,
    read_nnue,
    write_compressed_array,
    write_header,
    write_nnue,
    _encode_numpy_signed_leb128,
)


@pytest.mark.parametrize(
    "dtype,values",
    [
        (np.dtype("<i2"), [-32768, -8193, -65, -64, -1, 0, 63, 64, 8192, 32767]),
        (np.dtype("<i4"), [-(2**31), -(2**20), -1, 0, 1, 2**20, 2**31 - 1]),
    ],
)
def test_signed_leb128_round_trip_is_canonical(dtype, values):
    encoded = encode_signed_leb128(values, dtype)
    decoded = decode_signed_leb128(encoded, len(values), dtype)

    assert decoded.dtype == dtype
    assert decoded.tolist() == values
    assert encode_signed_leb128(decoded, dtype) == encoded


@pytest.mark.parametrize("dtype", [np.dtype("<i2"), np.dtype("<i4")])
def test_vectorized_writer_codec_matches_the_scalar_reference(dtype):
    generator = np.random.default_rng(20260714)
    limits = np.iinfo(dtype)
    values = generator.integers(limits.min, limits.max, size=4096, dtype=dtype)

    assert _encode_numpy_signed_leb128(values) == encode_signed_leb128(values, dtype)

    stream = io.BytesIO()
    write_compressed_array(stream, values)
    stream.seek(0)
    np.testing.assert_array_equal(read_compressed_array(stream, values.size, dtype), values)


def test_streaming_compressed_reader_rejects_noncanonical_values():
    payload = b"\x80\x00"
    stream = io.BytesIO(LEB128_MAGIC + struct.pack("<I", len(payload)) + payload)
    with pytest.raises(AtomicV2FormatError, match="non-canonical"):
        read_compressed_array(stream, 1, np.dtype("<i2"))


@pytest.mark.parametrize(
    "payload,error",
    [
        (b"", "truncated"),
        (b"\x80", "truncated"),
        (b"\x80\x00", "non-canonical"),
        (b"\xff\xff\xff\xff\x7f", "out of range"),
        (b"\x00\x00", "trailing"),
    ],
)
def test_signed_leb128_rejects_malformed_payloads(payload, error):
    with pytest.raises(AtomicV2FormatError, match=error):
        decode_signed_leb128(payload, 1, np.dtype("<i2"))


def test_compressed_array_has_strict_magic_length_and_value_count():
    stream = io.BytesIO()
    values = np.asarray([-2, 0, 127], dtype="<i2")
    write_compressed_array(stream, values)
    payload = stream.getvalue()

    assert payload.startswith(LEB128_MAGIC)
    assert read_compressed_array(io.BytesIO(payload), 3, np.dtype("<i2")).tolist() == [-2, 0, 127]

    with pytest.raises(AtomicV2FormatError, match="magic"):
        read_compressed_array(io.BytesIO(b"X" + payload[1:]), 3, np.dtype("<i2"))
    with pytest.raises(AtomicV2FormatError, match="payload"):
        read_compressed_array(io.BytesIO(payload[:-1]), 3, np.dtype("<i2"))
    # The encoded byte length can still be legal for four values (the final
    # value above needs two bytes), so the strict decoder may only discover
    # the count mismatch when it reaches the end of the payload.
    with pytest.raises(AtomicV2FormatError, match="length|truncated"):
        read_compressed_array(io.BytesIO(payload), 4, np.dtype("<i2"))


def test_v2_header_round_trip_and_v1_rejection_happen_before_parameters():
    stream = io.BytesIO()
    write_header(stream, "atomic-v2-test")
    stream.seek(0)

    assert read_header(stream) == "atomic-v2-test"

    legacy = io.BytesIO(struct.pack("<II", 0x7AF32F20, NETWORK_HASH) + struct.pack("<I", 0))
    with pytest.raises(AtomicV2FormatError, match="version"):
        read_header(legacy)

    wrong_hash = io.BytesIO(struct.pack("<III", FILE_VERSION, 0, 0))
    with pytest.raises(AtomicV2FormatError, match="network hash"):
        read_header(wrong_hash)


def test_v2_header_rejects_oversized_invalid_utf8_and_truncation():
    oversized = io.BytesIO(struct.pack("<III", FILE_VERSION, NETWORK_HASH, (1 << 20) + 1))
    with pytest.raises(AtomicV2FormatError, match="description length"):
        read_header(oversized)

    invalid_utf8 = io.BytesIO(struct.pack("<III", FILE_VERSION, NETWORK_HASH, 1) + b"\xff")
    with pytest.raises(AtomicV2FormatError, match="UTF-8"):
        read_header(invalid_utf8)

    truncated = io.BytesIO(struct.pack("<III", FILE_VERSION, NETWORK_HASH, 3) + b"ab")
    with pytest.raises(AtomicV2FormatError, match="description"):
        read_header(truncated)


class _HashingSink:
    def __init__(self):
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, data):
        self.digest.update(data)
        self.size += len(data)
        return len(data)


FT_PAIR_BIASES = (
    (2, 32, 32),
    (15, 16, 32),
    (16, 32, 48),
    (31, 48, 48),
    (32, 48, 56),
    (63, 56, 56),
    (64, 60, 60),
    (255, 20, 32),
    (256, 32, 40),
    (511, 32, 56),
)
FT_PAIR_INDICES = tuple(index for index, _first, _second in FT_PAIR_BIASES)
FC0_CONNECTIONS = (
    (0, 2, 5),
    (0, 514, 7),
    (1, 2, 11),
    (1, 514, 13),
) + tuple(
    connection
    for ordinal, index in enumerate(FT_PAIR_INDICES[1:], start=1)
    for connection in (
        (0, index, 1 + ordinal % 5),
        (0, 512 + index, 6 + ordinal % 5),
        (1, index, 2 + ordinal % 4),
        (1, 512 + index, 7 + ordinal % 4),
    )
)


def _transformed_inputs():
    values = {}
    for index, first, second in FT_PAIR_BIASES:
        transformed = first * second // 512
        values[index] = transformed
        values[512 + index] = transformed
    return values


def _reference_dense_trace(bucket=7):
    inputs = _transformed_inputs()
    transformed = tuple(inputs[index] for index in FT_PAIR_INDICES)
    fc0_biases = []
    for output, target in enumerate((8216, 4144)):
        contribution = sum(
            value * inputs[input_index]
            for candidate, input_index, value in FC0_CONNECTIONS
            if candidate == output
        )
        fc0_biases.append(target - contribution)
    fc0 = tuple(
        fc0_biases[output]
        + sum(
            value * inputs[input_index]
            for candidate, input_index, value in FC0_CONNECTIONS
            if candidate == output
        )
        for output in range(2)
    )
    sqr0 = tuple(value * value >> 21 for value in fc0)
    crelu0 = tuple(value >> 7 for value in fc0)
    fc1 = (
        4096 + 2 * sqr0[0] + 3 * sqr0[1] + 4 * crelu0[0] + 5 * crelu0[1],
        2048 + 6 * sqr0[0] + 7 * sqr0[1] + 8 * crelu0[0] + 9 * crelu0[1],
    )
    sqr1 = tuple(value * value >> 19 for value in fc1)
    crelu1 = tuple(value >> 6 for value in fc1)
    fc2 = (
        1024
        + 128 * (bucket - 7)
        + sqr0[0]
        + 2 * sqr0[1]
        + 3 * crelu0[0]
        + 4 * crelu0[1]
        + 5 * sqr1[0]
        + 6 * sqr1[1]
        + 7 * crelu1[0]
        + 8 * crelu1[1]
    )
    fwd = fc2 + 16384
    return {
        "transformed": transformed,
        "fc0": fc0,
        "sqr0": sqr0,
        "crelu0": crelu0,
        "fc1": fc1,
        "sqr1": sqr1,
        "crelu1": crelu1,
        "fc2": fc2,
        "fwd": fwd,
        "raw": fwd * int(PSQT_WEIGHT_SCALE) // int(FC2_BIAS_SCALE),
    }


def _controlled_wire_model():
    # This identity is pinned by Atomic-Stockfish's independent
    # tests/create_synthetic_atomic_v2_nnue.py writer.
    model = AtomicNNUEV2.__new__(AtomicNNUEV2)
    torch.nn.Module.__init__(model)
    weight = torch.zeros((45056, 1032), dtype=torch.float32)
    feature = torch.arange(45056, dtype=torch.int64)
    weight[:, 0] = (feature.remainder(63) + 1).to(torch.float32) / 256.0
    weight[:, 1] = -((feature * 17).remainder(64) + 1).to(torch.float32) / 256.0
    buckets = torch.arange(7, dtype=torch.int64)
    psqt = (
        feature.reshape(-1, 1) * (2 * buckets + 1) + 17 * buckets
    ).remainder(127) - 63
    weight[:, 1024:1031] = psqt.to(torch.float32) / PSQT_WEIGHT_SCALE
    bias = torch.zeros(1024, dtype=torch.float32)
    for index, first, second in FT_PAIR_BIASES:
        bias[index] = first / FT_ONE
        bias[512 + index] = second / FT_ONE
    object.__setattr__(
        model,
        "feature_transformer",
        SimpleNamespace(
            weight=weight,
            bias=bias,
        ),
    )
    model.network = AtomicLayerStacks()
    with torch.no_grad():
        for parameter in model.network.parameters():
            parameter.zero_()
        for bucket in range(8):
            fc0_row = bucket * 32
            model.network.fc0.linear.bias[fc0_row] = 7836.0 / FC0_BIAS_SCALE
            model.network.fc0.linear.bias[fc0_row + 1] = 3718.0 / FC0_BIAS_SCALE
            model.network.fc0.linear.bias[fc0_row + 30] = 16384.0 / FC0_BIAS_SCALE
            for output, input_index, value in FC0_CONNECTIONS:
                model.network.fc0.linear.weight[fc0_row + output, input_index] = (
                    value / FC0_WEIGHT_SCALE
                )

            fc1_row = bucket * 32
            model.network.fc1.linear.bias[fc1_row] = 4096.0 / FC1_BIAS_SCALE
            model.network.fc1.linear.bias[fc1_row + 1] = 2048.0 / FC1_BIAS_SCALE
            for output, input_index, value in (
                (0, 0, 2),
                (0, 1, 3),
                (0, 32, 4),
                (0, 33, 5),
                (1, 0, 6),
                (1, 1, 7),
                (1, 32, 8),
                (1, 33, 9),
            ):
                model.network.fc1.linear.weight[fc1_row + output, input_index] = (
                    value / FC1_WEIGHT_SCALE
                )

            model.network.fc2.linear.bias[bucket] = (
                1024.0 + 128.0 * (bucket - 7)
            ) / FC2_BIAS_SCALE
            for input_index, value in (
                (0, 1),
                (1, 2),
                (32, 3),
                (33, 4),
                (64, 5),
                (65, 6),
                (96, 7),
                (97, 8),
            ):
                model.network.fc2.linear.weight[bucket, input_index] = (
                    value / FC2_WEIGHT_SCALE
                )
    return model


@torch.no_grad()
def _model_dense_trace(model, indices, values, bucket_index=7):
    white, black = model.feature_transformer.forward_pair(
        indices,
        values,
        indices,
        values,
        True,
    )
    white_main = white[:, :1024]
    black_main = black[:, :1024]
    side_ordered = torch.cat((white_main, black_main), dim=1)
    transformed = fake_quantize_activation(
        pairwise_multiply(clip_feature_activation(side_ordered))
    )
    bucket = torch.tensor([bucket_index], dtype=torch.long)

    fc0 = model.network.fc0(transformed, bucket, True)
    activated0 = clip_hidden_activation(
        torch.cat(
            (
                fake_quantize_activation(fc0.square()),
                fake_quantize_activation(fc0),
            ),
            dim=1,
        )
    )
    fc1 = model.network.fc1(activated0, bucket, True)
    activated1 = clip_hidden_activation(
        torch.cat(
            (
                fake_quantize_activation(fc1.square()),
                fake_quantize_activation(fc1),
            ),
            dim=1,
        )
    )
    fc2 = model.network.fc2(torch.cat((activated0, activated1), dim=1), bucket, True)
    fwd = fc2 + fc0[:, -2:-1] - fc0[:, -1:]
    output = fake_quantize_output(fwd)

    def raw(tensor, scale):
        return tuple(int(round(float(value) * scale)) for value in tensor.reshape(-1))

    return {
        "transformed": tuple(
            raw(transformed[:, index], HIDDEN_ONE)[0]
            for index in FT_PAIR_INDICES
        ),
        "fc0": raw(fc0[:, :2], FC0_BIAS_SCALE),
        "sqr0": raw(activated0[:, :2], HIDDEN_ONE),
        "crelu0": raw(activated0[:, 32:34], HIDDEN_ONE),
        "fc1": raw(fc1[:, :2], FC1_BIAS_SCALE),
        "sqr1": raw(activated1[:, :2], HIDDEN_ONE),
        "crelu1": raw(activated1[:, 32:34], HIDDEN_ONE),
        "fc2": raw(fc2, FC2_BIAS_SCALE)[0],
        "fwd": raw(fwd, FC2_BIAS_SCALE)[0],
        "raw": raw(output, PSQT_WEIGHT_SCALE)[0],
    }


def _zero_stride_wire_model():
    model = AtomicNNUEV2.__new__(AtomicNNUEV2)
    torch.nn.Module.__init__(model)
    object.__setattr__(
        model,
        "feature_transformer",
        SimpleNamespace(
            weight=torch.zeros(1, dtype=torch.float32).expand(45056, 1032),
            bias=torch.zeros(1024, dtype=torch.float32),
        ),
    )
    model.network = AtomicLayerStacks()
    return model


def test_writer_validates_every_contract_shape_before_header_bytes():
    model = _zero_stride_wire_model()
    model.network.fc1.linear.weight = torch.nn.Parameter(torch.zeros((255, 64)))
    output = io.BytesIO()

    with pytest.raises(AtomicV2FormatError, match="fc1 weight shape"):
        write_nnue(output, model)

    assert output.getvalue() == b""


@pytest.mark.v2_full_io
def test_reader_imports_and_evaluates_the_engine_controlled_v2_fixture(tmp_path):
    path = tmp_path / "engine-controlled-v2.nnue"
    source_model = _controlled_wire_model()
    with path.open("xb") as output:
        write_nnue(
            output,
            source_model,
            "Atomic-Stockfish AtomicNNUEV2 controlled synthetic CI source",
        )
    with path.open("rb") as source:
        assert hashlib.file_digest(source, "sha256").hexdigest().upper() == (
            "4DEB05CFF79B5D5EBA51C560F64ED24224671C188B6C5DB27521033E587C87C6"
        )
    assert path.stat().st_size == 46_780_619
    del source_model
    gc.collect()

    with path.open("rb") as source:
        model, description = read_nnue(source)
    assert description == "Atomic-Stockfish AtomicNNUEV2 controlled synthetic CI source"
    assert torch.count_nonzero(model.feature_transformer.bias) == 20
    for index, first, second in FT_PAIR_BIASES:
        assert model.feature_transformer.bias[index] * FT_ONE == first
        assert model.feature_transformer.bias[512 + index] * FT_ONE == second
    expected_psqt = torch.tensor(
        [
            [-63, -46, -29, -12, 5, 22, 39, 0],
            [-62, -43, -24, -5, 14, 33, 52, 0],
        ],
        dtype=torch.float32,
    ) / PSQT_WEIGHT_SCALE
    torch.testing.assert_close(
        model.feature_transformer.weight[:2, 1024:],
        expected_psqt,
    )
    assert torch.count_nonzero(model.feature_transformer.weight[:, 1024:1031]) > 0
    assert torch.count_nonzero(model.feature_transformer.weight[:, 1031]) == 0

    us = torch.tensor([[1.0]], dtype=torch.float32)
    indices = torch.tensor([[0, -1]], dtype=torch.int32)
    values = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    expected_trace = {
        "transformed": (2, 1, 3, 4, 5, 6, 7, 1, 2, 3),
        "fc0": (8216, 4144),
        "sqr0": (32, 8),
        "crelu0": (64, 32),
        "fc1": (4600, 3096),
        "sqr1": (40, 18),
        "crelu1": (71, 48),
        "fc2": 2581,
        "fwd": 18965,
        "raw": 11112,
    }
    traces = tuple(_reference_dense_trace(bucket) for bucket in range(8))
    assert traces[-1] == expected_trace
    assert _model_dense_trace(model, indices, values, 7) == expected_trace
    assert tuple(trace["raw"] for trace in traces) == (
        10587,
        10662,
        10737,
        10812,
        10887,
        10962,
        11037,
        11112,
    )
    with torch.no_grad():
        output = model(
            us.repeat(8, 1),
            (1.0 - us).repeat(8, 1),
            indices.repeat(8, 1),
            values.repeat(8, 1),
            indices.repeat(8, 1),
            values.repeat(8, 1),
            torch.full((8,), 7, dtype=torch.long),
            torch.arange(8, dtype=torch.long),
        )
    raw_by_bucket = torch.tensor(
        [10587, 10662, 10737, 10812, 10887, 10962, 11037, 11112],
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        output,
        (raw_by_bucket / PSQT_WEIGHT_SCALE).reshape(-1, 1),
    )
    assert tuple((raw_by_bucket.to(torch.int64) // 16).tolist()) == (
        661,
        666,
        671,
        675,
        680,
        685,
        689,
        694,
    )

    with torch.no_grad():
        diagnostic_output = model(
            us,
            1.0 - us,
            torch.tensor([[0, -1]], dtype=torch.int32),
            values,
            torch.tensor([[1, -1]], dtype=torch.int32),
            values,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([7], dtype=torch.long),
        )
    torch.testing.assert_close(
        diagnostic_output,
        torch.tensor([[11112.0 / PSQT_WEIGHT_SCALE]]),
    )

    reexport = _HashingSink()
    write_nnue(reexport, model, description)
    assert reexport.digest.hexdigest().upper() == (
        "4DEB05CFF79B5D5EBA51C560F64ED24224671C188B6C5DB27521033E587C87C6"
    )
    del model
    gc.collect()
