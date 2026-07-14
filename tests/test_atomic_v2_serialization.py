import io
import gc
import hashlib
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atomic_v2.contract import FILE_VERSION, NETWORK_HASH
from atomic_v2.model import AtomicLayerStacks, AtomicNNUEV2
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


def _controlled_wire_model():
    # This identity is pinned by Atomic-Stockfish's independent
    # tests/create_synthetic_atomic_v2_nnue.py writer.
    model = AtomicNNUEV2.__new__(AtomicNNUEV2)
    torch.nn.Module.__init__(model)
    weight = torch.zeros((45056, 1032), dtype=torch.float32)
    feature = torch.arange(45056, dtype=torch.int64)
    weight[:, 0] = (feature.remainder(63) + 1).to(torch.float32) / 256.0
    weight[:, 1] = -((feature * 17).remainder(64) + 1).to(torch.float32) / 256.0
    object.__setattr__(
        model,
        "feature_transformer",
        SimpleNamespace(
            weight=weight,
            bias=torch.zeros(1024, dtype=torch.float32),
        ),
    )
    model.network = AtomicLayerStacks()
    with torch.no_grad():
        for parameter in model.network.parameters():
            parameter.zero_()
        for bucket in range(8):
            model.network.fc0.linear.bias[bucket * 32 + 30] = 1.0
    return model


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
            "2AF647392F7A30072C2DB29837C9175D22BA2DF450931762D0E2DE8CFE337B50"
        )
    del source_model
    gc.collect()

    with path.open("rb") as source:
        model, description = read_nnue(source)
    assert description == "Atomic-Stockfish AtomicNNUEV2 controlled synthetic CI source"

    us = torch.tensor([[1.0]], dtype=torch.float32)
    indices = torch.tensor([[0, -1]], dtype=torch.int32)
    values = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    with torch.no_grad():
        output = model(
            us,
            1.0 - us,
            indices,
            values,
            indices,
            values,
            torch.tensor([7], dtype=torch.long),
            torch.tensor([7], dtype=torch.long),
        )
    torch.testing.assert_close(output, torch.ones((1, 1)))

    reexport = _HashingSink()
    write_nnue(reexport, model, description)
    assert reexport.digest.hexdigest().upper() == (
        "2AF647392F7A30072C2DB29837C9175D22BA2DF450931762D0E2DE8CFE337B50"
    )
    del model
    gc.collect()
