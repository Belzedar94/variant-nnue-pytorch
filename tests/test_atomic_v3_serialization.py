import gc
import hashlib
import io
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import atomic_v3.serialization as serialization

from atomic_v3.contract import (
    ACCUMULATOR_DIMENSIONS,
    ARCHITECTURE_HASH,
    BLAST_RING_DIMENSIONS,
    CAPTURE_PAIR_DIMENSIONS,
    FEATURE_TRANSFORMER_HASH,
    FILE_VERSION,
    HM_OUTPUT_DIMENSIONS,
    HM_PHYSICAL_DIMENSIONS,
    HM_TRAINING_DIMENSIONS,
    HM_VIRTUAL_DIMENSIONS,
    KING_BLAST_EP_DIMENSIONS,
    LAYER_STACKS,
    NETWORK_HASH,
    PSQT_BUCKETS,
)
from atomic_v3.dense import AtomicV3LayerStacks
from atomic_v3.model import AtomicNNUEV3
from atomic_v3.quantization import FT_ONE
from atomic_v3.serialization import (
    LEB128_MAGIC,
    AtomicV3FormatError,
    _DenseSummary,
    _HM_PHYSICAL_TO_TRAINING,
    _HM_PHYSICAL_TO_VIRTUAL,
    _dense_bounds,
    _dense_parameters,
    _iter_hm_chunks,
    _observe_psqt,
    _quantized_numpy,
    _quantized_numpy_preserving_i32,
    _validate_dense_summaries,
    _validate_psqt,
    _write_all,
    _write_compressed_arrays,
    check_nnue,
    decode_signed_leb128,
    dumps_header,
    encode_signed_leb128,
    read_compressed_array,
    read_header,
    read_nnue,
    save_nnue,
    write_compressed_array,
    write_nnue,
)


def _empty_v3_model():
    model = AtomicNNUEV3.__new__(AtomicNNUEV3)
    torch.nn.Module.__init__(model)
    zero = torch.zeros(1, dtype=torch.float32)
    object.__setattr__(
        model,
        "feature_transformer",
        SimpleNamespace(
            bias=zero.expand(ACCUMULATOR_DIMENSIONS),
            hm_bucket_weight=zero.expand(
                HM_TRAINING_DIMENSIONS, HM_OUTPUT_DIMENSIONS
            ),
            hm_virtual_weight=zero.expand(
                HM_VIRTUAL_DIMENSIONS, HM_OUTPUT_DIMENSIONS
            ),
            capture_pair_weight=zero.expand(
                CAPTURE_PAIR_DIMENSIONS, ACCUMULATOR_DIMENSIONS
            ),
            king_blast_ep_weight=zero.expand(
                KING_BLAST_EP_DIMENSIONS, ACCUMULATOR_DIMENSIONS
            ),
            blast_ring_weight=zero.expand(
                BLAST_RING_DIMENSIONS, ACCUMULATOR_DIMENSIONS
            ),
        ),
    )
    model.network = AtomicV3LayerStacks()
    with torch.no_grad():
        for parameter in model.network.parameters():
            parameter.zero_()
    return model


@pytest.mark.parametrize(
    "dtype,values",
    [
        (np.dtype("<i2"), [-32768, -8193, -65, -64, -1, 0, 63, 64, 8192, 32767]),
        (
            np.dtype("<i4"),
            [-(2**31), -(2**20), -1, 0, 1, 2**20, 2**31 - 1],
        ),
    ],
)
def test_v3_signed_leb128_round_trip_is_canonical(dtype, values):
    encoded = encode_signed_leb128(values, dtype)
    decoded = decode_signed_leb128(encoded, len(values), dtype)

    assert decoded.tolist() == values
    assert encode_signed_leb128(decoded, dtype) == encoded


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
def test_v3_signed_leb128_rejects_malformed_payloads(payload, error):
    with pytest.raises(AtomicV3FormatError, match=error):
        decode_signed_leb128(payload, 1, np.dtype("<i2"))


def test_v3_streaming_compressed_array_rejects_noncanonical_values():
    payload = b"\x80\x00"
    stream = io.BytesIO(LEB128_MAGIC + struct.pack("<I", len(payload)) + payload)
    with pytest.raises(AtomicV3FormatError, match="non-canonical"):
        read_compressed_array(stream, 1, np.dtype("<i2"))

    values = np.asarray([-32768, -64, 0, 63, 32767], dtype="<i2")
    output = io.BytesIO()
    write_compressed_array(output, values)
    output.seek(0)
    np.testing.assert_array_equal(
        read_compressed_array(output, values.size, values.dtype), values
    )


@pytest.mark.parametrize("mode", ["none", "zero", "short"])
def test_write_all_rejects_incomplete_sink_writes(mode):
    class AdversarialSink:
        def write(self, data):
            if mode == "none":
                return None
            if mode == "zero":
                return 0
            return len(data) - 1

    with pytest.raises(AtomicV3FormatError, match="short write"):
        _write_all(AdversarialSink(), b"Atomic V3", "adversarial payload")


def test_streamed_compressed_payload_uses_checked_sink_writes():
    class PayloadShortSink(io.BytesIO):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def write(self, data):
            self.calls += 1
            if self.calls == 3:
                return super().write(data[:-1])
            return super().write(data)

    sink = PayloadShortSink()
    with pytest.raises(AtomicV3FormatError, match="short write"):
        _write_compressed_arrays(
            sink, [np.zeros(10, dtype="<i2")], 10, "adversarial array"
        )
    assert sink.calls == 3


def test_v3_header_is_exact_and_preserves_opaque_description_bytes():
    header = dumps_header(b"\xff\x00Atomic")
    assert header[:12] == struct.pack("<III", FILE_VERSION, NETWORK_HASH, 8)
    assert read_header(io.BytesIO(header)) == b"\xff\x00Atomic"

    wrong_version = io.BytesIO(struct.pack("<III", 0x7AF32F20, NETWORK_HASH, 0))
    with pytest.raises(AtomicV3FormatError, match="version"):
        read_header(wrong_version)


def test_quantization_uses_float64_scale_ties_to_even_and_never_saturates():
    values = torch.tensor(
        [-2.5 / FT_ONE, -1.5 / FT_ONE, -0.5 / FT_ONE, 0.5 / FT_ONE,
         1.5 / FT_ONE, 2.5 / FT_ONE],
        dtype=torch.float32,
    )
    before = values.clone()
    quantized = _quantized_numpy(values, FT_ONE, "<i2", "ties")
    assert quantized.tolist() == [-2, -2, 0, 0, 2, 2]
    torch.testing.assert_close(values, before)

    with pytest.raises(AtomicV3FormatError, match="exceeds int8"):
        _quantized_numpy(torch.tensor([128.0 / FT_ONE]), FT_ONE, "i1", "i8")
    with pytest.raises(AtomicV3FormatError, match="non-finite"):
        _quantized_numpy(torch.tensor([float("nan")]), FT_ONE, "<i2", "nan")


def test_imported_i32_bits_survive_float32_aliasing_until_parameter_changes():
    raw = np.asarray([(1 << 31) - 1], dtype="<i4")
    aliased = torch.from_numpy(raw.astype(np.float32)).div_(16384.0)
    assert aliased.item() == 131072.0
    np.testing.assert_array_equal(
        _quantized_numpy_preserving_i32(aliased, 16384.0, "bias", raw), raw
    )

    small_raw = np.asarray([100], dtype="<i4")
    changed = torch.from_numpy(small_raw.astype(np.float32)).div_(9600.0)
    changed.add_(1.0 / 9600.0)
    assert _quantized_numpy_preserving_i32(
        changed, 9600.0, "PSQT", small_raw
    ).tolist() == [101]


def test_hm_physical_map_merges_only_the_own_king_square():
    # Bucket 31 has oriented own king e1 == square 4.
    merged_begin = 31 * 704 + 10 * 64
    assert _HM_PHYSICAL_TO_TRAINING[merged_begin + 4] == 31 * 768 + 10 * 64 + 4
    assert _HM_PHYSICAL_TO_VIRTUAL[merged_begin + 4] == 10 * 64 + 4
    assert _HM_PHYSICAL_TO_TRAINING[merged_begin + 60] == 31 * 768 + 11 * 64 + 60
    assert _HM_PHYSICAL_TO_VIRTUAL[merged_begin + 60] == 11 * 64 + 60

    base_column = torch.arange(HM_TRAINING_DIMENSIONS, dtype=torch.float32).reshape(-1, 1)
    factor_column = (
        torch.arange(HM_VIRTUAL_DIMENSIONS, dtype=torch.float32).reshape(-1, 1)
        / 16.0
    )
    model = SimpleNamespace(
        feature_transformer=SimpleNamespace(
            hm_bucket_weight=base_column.expand(-1, HM_OUTPUT_DIMENSIONS),
            hm_virtual_weight=factor_column.expand(-1, HM_OUTPUT_DIMENSIONS),
        )
    )
    exported = torch.cat(list(_iter_hm_chunks(model, 0, 1)))
    expected = (
        base_column[_HM_PHYSICAL_TO_TRAINING, 0]
        + factor_column[_HM_PHYSICAL_TO_VIRTUAL, 0]
    )
    torch.testing.assert_close(exported, expected)


def test_fc0_factor_is_merged_in_float32_per_bucket_without_mutation():
    model = _empty_v3_model()
    base = model.network.fc0.linear.weight
    factor = model.network.fc0.factorized_linear.weight
    with torch.no_grad():
        base[0, 0] = 1.0
        factor[0, 0] = 0.5
        model.network.fc0.linear.bias[0] = -0.25
        model.network.fc0.factorized_linear.bias[0] = 0.125
    base_before = base.clone()
    factor_before = factor.clone()

    bucket0 = _dense_parameters(model, 0)[0]
    bucket1 = _dense_parameters(model, 1)[0]
    assert bucket0[0][0, 0].item() == 1.5
    assert bucket1[0][0, 0].item() == 0.5
    assert bucket0[1][0].item() == -0.125
    torch.testing.assert_close(base, base_before)
    torch.testing.assert_close(factor, factor_before)


def test_fc0_factor_merge_happens_before_float64_quantization():
    model = _empty_v3_model()
    with torch.no_grad():
        # In float32 this coalesces to 131072 exactly and is therefore one
        # integer beyond signed-i32 after the 16384 bias scale.  A forbidden
        # float64 factor merge would instead produce the in-range 2147483584.
        model.network.fc0.linear.bias[0] = 131072.0
        model.network.fc0.factorized_linear.bias[0] = -1.0 / 256.0
    merged = _dense_parameters(model, 0)[0][1]
    assert merged.dtype == torch.float32
    assert merged[0].item() == 131072.0
    with pytest.raises(AtomicV3FormatError, match="exceeds int32"):
        _quantized_numpy(merged, 16384.0, "<i4", "fc0 bias")


def test_writer_rejects_shape_and_nonfinite_before_header_bytes():
    model = _empty_v3_model()
    model.network.fc1.linear.weight = torch.nn.Parameter(torch.zeros((255, 64)))
    output = io.BytesIO()
    with pytest.raises(AtomicV3FormatError, match="fc1 weight shape"):
        write_nnue(output, model)
    assert output.getvalue() == b""

    model = _empty_v3_model()
    model.feature_transformer.bias = torch.full(
        (ACCUMULATOR_DIMENSIONS,), float("nan"), dtype=torch.float32
    )
    with pytest.raises(AtomicV3FormatError, match="non-finite"):
        write_nnue(output, model)
    assert output.getvalue() == b""


def test_save_nnue_uses_configurable_atomic_no_overwrite_path(tmp_path, monkeypatch):
    target = tmp_path / "directory with spaces" / "epoch-37.nnue"
    payload = b"strict-v3-test-payload"

    def fake_write(stream, model, description):
        assert model is sentinel
        assert description == b"opaque"
        stream.write(payload)

    def fake_check(stream):
        assert stream.read() == payload
        return serialization.WireMetadata(
            b"opaque", len(payload), hashlib.sha256(payload).hexdigest().upper()
        )

    sentinel = object()
    monkeypatch.setattr(serialization, "write_nnue", fake_write)
    monkeypatch.setattr(serialization, "check_nnue", fake_check)
    metadata = save_nnue(target, sentinel, b"opaque")

    assert target.read_bytes() == payload
    assert metadata.size == len(payload)
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))
    with pytest.raises(FileExistsError):
        save_nnue(target, sentinel, b"opaque")
    assert target.read_bytes() == payload


def test_psqt_and_dense_numeric_envelopes_match_the_engine():
    heaps = [[] for _ in range(PSQT_BUCKETS)]
    values = np.zeros(32 * PSQT_BUCKETS, dtype="<i4")
    values[0::PSQT_BUCKETS] = 67_108_864
    overflow = _observe_psqt(heaps, 0, values)
    assert not overflow
    with pytest.raises(AtomicV3FormatError, match="top-32"):
        _validate_psqt(heaps, overflow)

    zero_fc0 = _dense_bounds(
        np.zeros(32, dtype="<i4"), np.zeros((32, 1024), dtype="i1")
    )
    zero_fc1 = _dense_bounds(
        np.zeros(32, dtype="<i4"), np.zeros((32, 64), dtype="i1")
    )
    zero_fc2 = _dense_bounds(
        np.zeros(1, dtype="<i4"), np.zeros((1, 128), dtype="i1")
    )
    _validate_dense_summaries([_DenseSummary(0, zero_fc0, zero_fc1, zero_fc2)])

    bias = np.zeros(32, dtype="<i4")
    bias[0] = (1 << 31) - 1
    weight = np.zeros((32, 1024), dtype="i1")
    weight[0, 0] = 1
    unsafe_fc0 = _dense_bounds(bias, weight)
    with pytest.raises(AtomicV3FormatError, match="affine"):
        _validate_dense_summaries(
            [_DenseSummary(0, unsafe_fc0, zero_fc1, zero_fc2)]
        )


def _write_zeros(stream, count):
    block = bytes(1 << 20)
    while count:
        size = min(count, len(block))
        stream.write(block[:size])
        count -= size


def _write_zero_frame(stream, count):
    stream.write(LEB128_MAGIC)
    stream.write(struct.pack("<I", count))
    _write_zeros(stream, count)


def _write_first_i32_frame(stream, count, first):
    encoded = encode_signed_leb128([first], np.dtype("<i4"))
    stream.write(LEB128_MAGIC)
    stream.write(struct.pack("<I", len(encoded) + count - 1))
    stream.write(encoded)
    _write_zeros(stream, count - 1)


def _write_zero_network(
    path: Path,
    description=b"Atomic V3 Python zero wire",
    *,
    i32_alias_boundaries=False,
):
    with path.open("xb") as stream:
        stream.write(struct.pack("<III", FILE_VERSION, NETWORK_HASH, len(description)))
        stream.write(description)
        stream.write(struct.pack("<I", FEATURE_TRANSFORMER_HASH))
        _write_zero_frame(stream, ACCUMULATOR_DIMENSIONS)
        _write_zero_frame(
            stream, HM_PHYSICAL_DIMENSIONS * ACCUMULATOR_DIMENSIONS
        )
        _write_zeros(
            stream, CAPTURE_PAIR_DIMENSIONS * ACCUMULATOR_DIMENSIONS
        )
        _write_zero_frame(
            stream, KING_BLAST_EP_DIMENSIONS * ACCUMULATOR_DIMENSIONS
        )
        _write_zeros(stream, BLAST_RING_DIMENSIONS * ACCUMULATOR_DIMENSIONS)
        if i32_alias_boundaries:
            _write_first_i32_frame(
                stream, HM_PHYSICAL_DIMENSIONS * PSQT_BUCKETS, (1 << 31) - 1
            )
        else:
            _write_zero_frame(stream, HM_PHYSICAL_DIMENSIONS * PSQT_BUCKETS)
        for _ in range(LAYER_STACKS):
            stream.write(struct.pack("<I", ARCHITECTURE_HASH))
            if i32_alias_boundaries:
                stream.write(struct.pack("<i", (1 << 31) - 1))
                _write_zeros(stream, 31 * 4)
            else:
                _write_zeros(stream, 32 * 4)
            _write_zeros(stream, 32 * 1024)
            if i32_alias_boundaries:
                stream.write(struct.pack("<i", -((1 << 31) - 1)))
                _write_zeros(stream, 31 * 4)
            else:
                _write_zeros(stream, 32 * 4)
            _write_zeros(stream, 32 * 64)
            _write_zeros(stream, 4 + 128)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


@pytest.mark.v3_full_io
def test_checker_and_import_export_round_trip_the_complete_zero_wire(tmp_path):
    source = tmp_path / "zero-v3.nnue"
    target = tmp_path / "zero-v3-reexport.nnue"
    checkpoint = tmp_path / "zero-v3-state.pt"
    _write_zero_network(source, i32_alias_boundaries=True)
    expected_sha = _sha256(source)

    with source.open("rb") as stream:
        metadata = check_nnue(stream)
    assert metadata.description == b"Atomic V3 Python zero wire"
    assert metadata.size == source.stat().st_size
    assert metadata.sha256 == expected_sha

    with source.open("rb") as stream:
        model, description = read_nnue(stream)
    assert description == metadata.description
    assert torch.count_nonzero(model.feature_transformer.hm_virtual_weight) == 0
    assert model.feature_transformer.hm_bucket_weight[0, 0].item() == 0.0
    assert model._atomic_v3_imported_i32_valid.item() is True
    assert model._atomic_v3_imported_hm_psqt_i32[0, 0].item() == (1 << 31) - 1
    assert model._atomic_v3_imported_fc0_bias_i32[0, 0].item() == (1 << 31) - 1
    assert model._atomic_v3_imported_fc1_bias_i32[0, 0].item() == -((1 << 31) - 1)
    imported_buffer_names = {
        name
        for name, _ in model.named_buffers()
        if name.startswith("_atomic_v3_imported_")
    }
    assert imported_buffer_names == {
        "_atomic_v3_imported_i32_valid",
        "_atomic_v3_imported_hm_psqt_i32",
        "_atomic_v3_imported_fc0_bias_i32",
        "_atomic_v3_imported_fc1_bias_i32",
        "_atomic_v3_imported_fc2_bias_i32",
    }
    assert not any(
        name.startswith("_atomic_v3_imported_")
        for name, _ in model.named_parameters()
    )
    torch.save(model.state_dict(), checkpoint)
    del model
    gc.collect()

    restored = AtomicNNUEV3(initialize=False)
    restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    assert restored._atomic_v3_imported_i32_valid.item() is True
    published = save_nnue(target, restored, description)
    assert target.stat().st_size == source.stat().st_size
    assert published.size == target.stat().st_size
    assert _sha256(target) == expected_sha
    assert published.sha256 == expected_sha
    del restored
    gc.collect()

    with source.open("ab") as stream:
        stream.write(b"X")
    with source.open("rb") as stream:
        with pytest.raises(AtomicV3FormatError, match="trailing bytes"):
            check_nnue(stream)
