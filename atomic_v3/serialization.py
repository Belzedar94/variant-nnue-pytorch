"""Strict mixed-wire AtomicNNUEV3 reader and writer.

The file layout in this module is the private wire-v1 frozen by
Atomic-Stockfish.  It is deliberately independent from the legacy and V2
serializers: V3 has four feature slices, two integer widths, raw signed-i8
sections, a factorized 12-to-11 HalfKAv2 export, and eight SFNNv15 stacks.

Export is streaming and preflights every quantized parameter before emitting
the first header byte.  Import validates the complete canonical stream and
the engine's numeric envelopes before returning a model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import io
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import BinaryIO, Callable, Iterable, Iterator, Optional

import numpy as np
import torch

from .contract import (
    ACCUMULATOR_DIMENSIONS,
    ARCHITECTURE_HASH,
    BLAST_RING_DIMENSIONS,
    CAPTURE_PAIR_DIMENSIONS,
    FC0_INPUTS,
    FC0_OUTPUTS,
    FC1_INPUTS,
    FC1_OUTPUTS,
    FC2_INPUTS,
    FC2_OUTPUTS,
    FEATURE_TRANSFORMER_HASH,
    FILE_VERSION,
    HM_KING_BUCKETS,
    HM_OUTPUT_DIMENSIONS,
    HM_PHYSICAL_DIMENSIONS,
    HM_PHYSICAL_ROWS_PER_BUCKET,
    HM_ROWS_PER_BUCKET,
    HM_TRAINING_DIMENSIONS,
    HM_VIRTUAL_DIMENSIONS,
    KING_BLAST_EP_DIMENSIONS,
    LAYER_STACKS,
    NETWORK_HASH,
    PSQT_BUCKETS,
)
from .model import AtomicNNUEV3
from .quantization import (
    FC0_BIAS_SCALE,
    FC0_WEIGHT_SCALE,
    FC1_BIAS_SCALE,
    FC1_WEIGHT_SCALE,
    FC2_BIAS_SCALE,
    FC2_WEIGHT_SCALE,
    FT_ONE,
    INT32_MAX,
    INT32_MIN,
    PSQT_WEIGHT_SCALE,
)


LEB128_MAGIC = b"COMPRESSED_LEB128"
MAX_DESCRIPTION_BYTES = 1 << 20
STREAM_CHUNK_BYTES = 1 << 20
DEFAULT_DESCRIPTION = (
    "AtomicNNUEV3 trained with Belzedar94/variant-nnue-pytorch atomic branch."
)

# Frozen by Atomic-Stockfish's scalar output mapping.  The interval is
# asymmetric because the positive endpoint maps exactly to INT32_MAX.
RAW_OUTPUT_MIN = -3_665_038_760
RAW_OUTPUT_MAX = 3_665_038_759


class AtomicV3FormatError(ValueError):
    """A V3 network is malformed, non-canonical, unsafe, or incompatible."""


@dataclass(frozen=True)
class WireMetadata:
    """Authenticated metadata returned by :func:`check_nnue`."""

    description: bytes
    size: int
    sha256: str

    def description_text(self) -> str:
        """Decode a textual description without narrowing the wire contract."""

        return self.description.decode("utf-8")


@dataclass(frozen=True)
class _DenseBounds:
    absolute: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class _DenseSummary:
    bucket: int
    fc0: _DenseBounds
    fc1: _DenseBounds
    fc2: _DenseBounds


@dataclass(frozen=True)
class _ImportedI32State:
    """Raw i32 values whose float32 trainer image can be non-injective."""

    hm_psqt: np.ndarray
    dense_biases: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]


class _HashingReader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(size)
        if data is None:
            data = b""
        self.digest.update(data)
        self.size += len(data)
        return data


def _readonly_i32(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype="<i4").copy()
    result.flags.writeable = False
    return result


def _description_bytes(description: str | bytes | bytearray | memoryview) -> bytes:
    if isinstance(description, str):
        encoded = description.encode("utf-8")
    elif isinstance(description, (bytes, bytearray, memoryview)):
        encoded = bytes(description)
    else:
        raise TypeError("description must be text or bytes")
    if len(encoded) > MAX_DESCRIPTION_BYTES:
        raise AtomicV3FormatError("description length exceeds 1 MiB")
    return encoded


def _write_all(
    stream: BinaryIO, data: bytes | bytearray | memoryview, label: str
) -> None:
    view = memoryview(data).cast("B")
    offset = 0
    while offset < len(view):
        written = stream.write(view[offset:])
        if written is None:
            return
        if written <= 0:
            raise AtomicV3FormatError(f"short write while writing {label}")
        offset += written


def _read_exact(stream: BinaryIO, size: int, label: str) -> bytes:
    if size < 0:
        raise ValueError("read size must be non-negative")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise AtomicV3FormatError(f"truncated {label}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_u32(stream: BinaryIO, label: str) -> int:
    return struct.unpack("<I", _read_exact(stream, 4, label))[0]


def _write_u32(stream: BinaryIO, value: int) -> None:
    _write_all(stream, struct.pack("<I", value), "unsigned integer")


def _signed_dtype(dtype: np.dtype | str) -> np.dtype:
    result = np.dtype(dtype)
    if result.kind != "i" or result.itemsize not in (1, 2, 4):
        raise TypeError("Atomic V3 integer arrays require signed i8, i16, or i32")
    return result.newbyteorder("<")


def _encode_one(value: int) -> bytes:
    output = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        done = (value == 0 and (byte & 0x40) == 0) or (
            value == -1 and (byte & 0x40) != 0
        )
        output.append(byte if done else byte | 0x80)
        if done:
            return bytes(output)


def encode_signed_leb128(
    values: Iterable[int] | np.ndarray, dtype: np.dtype | str
) -> bytes:
    dtype = _signed_dtype(dtype)
    limits = np.iinfo(dtype)
    output = bytearray()
    for raw_value in values:
        value = int(raw_value)
        if value < limits.min or value > limits.max:
            raise AtomicV3FormatError(
                f"SLEB value {value} is out of range for {dtype.name}"
            )
        output.extend(_encode_one(value))
    return bytes(output)


def _encode_numpy_signed_leb128(values: np.ndarray) -> bytes:
    values = np.asarray(values)
    dtype = _signed_dtype(values.dtype)
    flat = values.reshape(-1).astype(np.int64, copy=False)
    if flat.size == 0:
        return b""
    max_bytes = (dtype.itemsize * 8 + 6) // 7
    encoded = np.zeros((flat.size, max_bytes), dtype=np.uint8)
    lengths = np.zeros(flat.size, dtype=np.uint8)
    active = np.ones(flat.size, dtype=bool)
    current = flat.copy()
    for column in range(max_bytes):
        byte = (current & 0x7F).astype(np.uint8)
        shifted = current >> 7
        done = active & (
            ((shifted == 0) & ((byte & 0x40) == 0))
            | ((shifted == -1) & ((byte & 0x40) != 0))
        )
        encoded[active, column] = byte[active] | np.where(
            done[active], 0, 0x80
        ).astype(np.uint8)
        lengths[done] = column + 1
        active &= ~done
        current = shifted
    if np.any(active):
        raise AtomicV3FormatError(
            f"signed LEB128 value is out of range for {dtype.name}"
        )
    mask = np.arange(max_bytes, dtype=np.uint8)[None, :] < lengths[:, None]
    return encoded[mask].tobytes()


def decode_signed_leb128(
    payload: bytes | bytearray | memoryview,
    count: int,
    dtype: np.dtype | str,
) -> np.ndarray:
    dtype = _signed_dtype(dtype)
    if count < 0:
        raise ValueError("count must be non-negative")
    data = memoryview(payload).cast("B")
    limits = np.iinfo(dtype)
    max_bytes = (dtype.itemsize * 8 + 6) // 7
    output = np.empty(count, dtype=dtype)
    position = 0
    for index in range(count):
        start = position
        decoded = 0
        shift = 0
        terminal = None
        for used in range(1, max_bytes + 1):
            if position >= len(data):
                raise AtomicV3FormatError("truncated signed LEB128 value")
            byte = int(data[position])
            position += 1
            decoded |= (byte & 0x7F) << shift
            shift += 7
            if (byte & 0x80) == 0:
                terminal = byte
                break
            if used == max_bytes:
                raise AtomicV3FormatError(
                    f"signed LEB128 value is out of range for {dtype.name}"
                )
        if terminal is None:
            raise AtomicV3FormatError("truncated signed LEB128 value")
        if terminal & 0x40:
            decoded |= -(1 << shift)
        if decoded < limits.min or decoded > limits.max:
            raise AtomicV3FormatError(
                f"signed LEB128 value is out of range for {dtype.name}"
            )
        if bytes(data[start:position]) != _encode_one(decoded):
            raise AtomicV3FormatError("non-canonical signed LEB128 value")
        output[index] = decoded
    if position != len(data):
        raise AtomicV3FormatError("trailing bytes in signed LEB128 payload")
    return output


def _decode_complete_numpy_sleb(payload: bytes, dtype: np.dtype) -> np.ndarray:
    dtype = _signed_dtype(dtype)
    if not payload:
        return np.empty(0, dtype=dtype)
    data = np.frombuffer(payload, dtype=np.uint8)
    ends = np.flatnonzero((data & 0x80) == 0)
    if ends.size == 0 or int(ends[-1]) != data.size - 1:
        raise AtomicV3FormatError("truncated signed LEB128 value")
    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    lengths = ends - starts + 1
    max_bytes = (dtype.itemsize * 8 + 6) // 7
    if np.any(lengths > max_bytes):
        raise AtomicV3FormatError(
            f"signed LEB128 value is out of range for {dtype.name}"
        )
    decoded = np.zeros(ends.size, dtype=np.int64)
    for column in range(max_bytes):
        selected = lengths > column
        if not np.any(selected):
            break
        bytes_at_column = data[starts[selected] + column]
        decoded[selected] |= (bytes_at_column & 0x7F).astype(np.int64) << (
            7 * column
        )
    terminal = data[ends]
    signed = (terminal & 0x40) != 0
    decoded[signed] |= -(1 << (7 * lengths[signed]))
    limits = np.iinfo(dtype)
    if np.any(decoded < limits.min) or np.any(decoded > limits.max):
        raise AtomicV3FormatError(
            f"signed LEB128 value is out of range for {dtype.name}"
        )
    expected_lengths = np.zeros(decoded.size, dtype=lengths.dtype)
    active = np.ones(decoded.size, dtype=bool)
    current = decoded.copy()
    for column in range(max_bytes):
        byte = current & 0x7F
        shifted = current >> 7
        done = active & (
            ((shifted == 0) & ((byte & 0x40) == 0))
            | ((shifted == -1) & ((byte & 0x40) != 0))
        )
        expected_lengths[done] = column + 1
        active &= ~done
        current = shifted
    if np.any(active):
        raise AtomicV3FormatError(
            f"signed LEB128 value is out of range for {dtype.name}"
        )
    if not np.array_equal(lengths, expected_lengths):
        raise AtomicV3FormatError("non-canonical signed LEB128 value")
    return decoded.astype(dtype, copy=False)


def write_compressed_array(stream: BinaryIO, values: np.ndarray) -> None:
    values = np.asarray(values)
    dtype = _signed_dtype(values.dtype)
    payload = encode_signed_leb128(values.reshape(-1), dtype)
    if len(payload) > 0xFFFFFFFF:
        raise AtomicV3FormatError("compressed payload exceeds uint32 length")
    _write_all(stream, LEB128_MAGIC, "compressed magic")
    _write_u32(stream, len(payload))
    _write_all(stream, payload, "compressed payload")


def _read_compressed_chunks(
    stream: BinaryIO,
    count: int,
    dtype: np.dtype | str,
    label: str,
    consumer: Optional[Callable[[int, np.ndarray], None]] = None,
) -> None:
    dtype = _signed_dtype(dtype)
    if count < 0:
        raise ValueError("count must be non-negative")
    magic = _read_exact(stream, len(LEB128_MAGIC), f"{label} compressed magic")
    if magic != LEB128_MAGIC:
        raise AtomicV3FormatError(f"invalid compressed magic for {label}")
    byte_count = _read_u32(stream, f"{label} compressed payload length")
    max_bytes = (dtype.itemsize * 8 + 6) // 7
    if byte_count < count or byte_count > count * max_bytes:
        raise AtomicV3FormatError(
            f"invalid compressed length {byte_count} for {count} {dtype.name} values"
        )
    output_offset = 0
    remaining = byte_count
    pending = b""
    while remaining:
        block = _read_exact(
            stream, min(remaining, STREAM_CHUNK_BYTES), f"{label} compressed payload"
        )
        remaining -= len(block)
        data = pending + block
        byte_view = np.frombuffer(data, dtype=np.uint8)
        terminal = np.flatnonzero((byte_view & 0x80) == 0)
        if terminal.size == 0:
            if len(data) > max_bytes:
                raise AtomicV3FormatError(
                    f"signed LEB128 value is out of range for {dtype.name}"
                )
            pending = data
            continue
        boundary = int(terminal[-1]) + 1
        complete = data[:boundary]
        pending = data[boundary:]
        if len(pending) >= max_bytes:
            raise AtomicV3FormatError(
                f"signed LEB128 value is out of range for {dtype.name}"
            )
        if complete.count(0) == len(complete):
            values = np.zeros(len(complete), dtype=dtype)
        else:
            values = _decode_complete_numpy_sleb(complete, dtype)
        end = output_offset + values.size
        if end > count:
            raise AtomicV3FormatError(
                f"trailing values in {label} signed LEB128 payload"
            )
        if consumer is not None:
            consumer(output_offset, values)
        output_offset = end
    if pending:
        raise AtomicV3FormatError(f"truncated {label} signed LEB128 value")
    if output_offset != count:
        raise AtomicV3FormatError(
            f"truncated {label} payload: decoded {output_offset} of {count} values"
        )


def read_compressed_array(
    stream: BinaryIO, count: int, dtype: np.dtype | str
) -> np.ndarray:
    dtype = _signed_dtype(dtype)
    output = np.empty(count, dtype=dtype)

    def consume(offset: int, values: np.ndarray) -> None:
        output[offset : offset + values.size] = values

    _read_compressed_chunks(stream, count, dtype, "array", consume)
    return output


def write_header(
    stream: BinaryIO,
    description: str | bytes | bytearray | memoryview = DEFAULT_DESCRIPTION,
) -> None:
    encoded = _description_bytes(description)
    _write_u32(stream, FILE_VERSION)
    _write_u32(stream, NETWORK_HASH)
    _write_u32(stream, len(encoded))
    _write_all(stream, encoded, "description")


def read_header(stream: BinaryIO) -> bytes:
    version = _read_u32(stream, "file version")
    if version != FILE_VERSION:
        raise AtomicV3FormatError(
            f"incompatible file version 0x{version:08X}; expected Atomic V3 "
            f"0x{FILE_VERSION:08X}"
        )
    network_hash = _read_u32(stream, "network hash")
    if network_hash != NETWORK_HASH:
        raise AtomicV3FormatError(
            f"incompatible network hash 0x{network_hash:08X}; expected "
            f"0x{NETWORK_HASH:08X}"
        )
    description_length = _read_u32(stream, "description length")
    if description_length > MAX_DESCRIPTION_BYTES:
        raise AtomicV3FormatError("invalid description length")
    return _read_exact(stream, description_length, "description")


def dumps_header(
    description: str | bytes | bytearray | memoryview = DEFAULT_DESCRIPTION,
) -> bytes:
    stream = io.BytesIO()
    write_header(stream, description)
    return stream.getvalue()


def _physical_hm_maps() -> tuple[np.ndarray, np.ndarray]:
    training = np.empty(HM_PHYSICAL_DIMENSIONS, dtype=np.int64)
    virtual = np.empty(HM_PHYSICAL_DIMENSIONS, dtype=np.int64)
    physical_row = 0
    for bucket in range(HM_KING_BUCKETS):
        rank = 7 - bucket // 4
        file = 7 - bucket % 4
        oriented_own_king = rank * 8 + file
        for physical_plane in range(11):
            for square in range(64):
                training_plane = physical_plane
                if physical_plane == 10:
                    training_plane = 10 if square == oriented_own_king else 11
                training[physical_row] = (
                    bucket * HM_ROWS_PER_BUCKET + training_plane * 64 + square
                )
                virtual[physical_row] = training_plane * 64 + square
                physical_row += 1
    if physical_row != HM_PHYSICAL_DIMENSIONS:
        raise AssertionError("Atomic V3 physical HM map has the wrong size")
    return training, virtual


_HM_PHYSICAL_TO_TRAINING, _HM_PHYSICAL_TO_VIRTUAL = _physical_hm_maps()


def _iter_tensor_chunks(
    tensor: torch.Tensor, *, chunk_elements: int = 1 << 18
) -> Iterator[torch.Tensor]:
    source = tensor.detach()
    if source.is_contiguous():
        flat = source.reshape(-1)
        for offset in range(0, flat.numel(), chunk_elements):
            yield flat[offset : offset + chunk_elements]
        return
    if source.ndim == 2:
        columns = source.shape[1]
        rows_per_chunk = max(1, chunk_elements // columns)
        for row in range(0, source.shape[0], rows_per_chunk):
            yield source[row : row + rows_per_chunk].contiguous().reshape(-1)
        return
    flat = source.contiguous().reshape(-1)
    for offset in range(0, flat.numel(), chunk_elements):
        yield flat[offset : offset + chunk_elements]


def _iter_hm_chunks(
    model: AtomicNNUEV3,
    column_begin: int,
    column_end: int,
    *,
    chunk_elements: int = 1 << 18,
) -> Iterator[torch.Tensor]:
    base = model.feature_transformer.hm_bucket_weight
    factor = model.feature_transformer.hm_virtual_weight
    columns = column_end - column_begin
    rows_per_chunk = max(1, chunk_elements // columns)
    for physical_begin in range(0, HM_PHYSICAL_DIMENSIONS, rows_per_chunk):
        physical_end = min(
            physical_begin + rows_per_chunk, HM_PHYSICAL_DIMENSIONS
        )
        training_rows = torch.as_tensor(
            _HM_PHYSICAL_TO_TRAINING[physical_begin:physical_end],
            dtype=torch.long,
            device=base.device,
        )
        virtual_rows = torch.as_tensor(
            _HM_PHYSICAL_TO_VIRTUAL[physical_begin:physical_end],
            dtype=torch.long,
            device=factor.device,
        )
        base_rows = base.index_select(0, training_rows)[
            :, column_begin:column_end
        ]
        factor_rows = factor.index_select(0, virtual_rows)[
            :, column_begin:column_end
        ]
        # The training graph and wire contract both coalesce factors in
        # float32.  Widening happens only after this persisted sum.
        yield (base_rows + factor_rows).reshape(-1)


def _quantized_numpy(
    tensor: torch.Tensor,
    scale: float,
    dtype: np.dtype | str,
    label: str,
) -> np.ndarray:
    dtype = _signed_dtype(dtype)
    # Quantization is deliberately performed on CPU in float64 after any
    # float32 factor merge.  torch.round implements ties-to-even.
    source = tensor.detach().to(device="cpu", dtype=torch.float32)
    rounded = torch.round(source.to(torch.float64) * float(scale))
    if not torch.isfinite(rounded).all():
        raise AtomicV3FormatError(f"{label} contains a non-finite parameter")
    limits = np.iinfo(dtype)
    if rounded.numel():
        minimum = int(rounded.min().item())
        maximum = int(rounded.max().item())
        if minimum < limits.min or maximum > limits.max:
            raise AtomicV3FormatError(
                f"{label} range [{minimum}, {maximum}] exceeds {dtype.name}"
            )
    return rounded.numpy().astype(dtype, copy=False)


def _quantized_numpy_preserving_i32(
    tensor: torch.Tensor,
    scale: float,
    label: str,
    preserved: Optional[np.ndarray],
) -> np.ndarray:
    """Reuse imported i32 bits while their float32 parameter is unchanged.

    Division by 9600 and large dense biases are not injective in float32.
    Retaining the original integer is therefore necessary for byte-exact
    read/write round trips.  Once a trainer parameter changes, normal strict
    ties-to-even quantization is used for that element.
    """

    if preserved is None:
        return _quantized_numpy(tensor, scale, "<i4", label)
    raw = np.asarray(preserved, dtype="<i4").reshape(-1)
    current = (
        tensor.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .reshape(-1)
    )
    if current.numel() != raw.size:
        raise AtomicV3FormatError(
            f"{label} preserved i32 shape does not match the parameter"
        )
    if not torch.isfinite(current).all():
        raise AtomicV3FormatError(f"{label} contains a non-finite parameter")
    baseline = torch.from_numpy(raw.astype(np.float32)).div_(float(scale))
    unchanged = current.view(torch.int32) == baseline.view(torch.int32)
    result = raw.copy()
    changed = torch.nonzero(~unchanged, as_tuple=False).reshape(-1)
    if changed.numel():
        rounded = torch.round(current[changed].to(torch.float64) * float(scale))
        if not torch.isfinite(rounded).all():
            raise AtomicV3FormatError(f"{label} contains a non-finite parameter")
        minimum = int(rounded.min().item())
        maximum = int(rounded.max().item())
        if minimum < INT32_MIN or maximum > INT32_MAX:
            raise AtomicV3FormatError(
                f"{label} range [{minimum}, {maximum}] exceeds int32"
            )
        changed_indices = changed.numpy()
        result[changed_indices] = rounded.numpy().astype("<i4", copy=False)
    return result.reshape(np.asarray(preserved).shape)


def _iter_quantized(
    chunks: Iterable[torch.Tensor],
    scale: float,
    dtype: np.dtype | str,
    label: str,
) -> Iterator[np.ndarray]:
    for chunk in chunks:
        yield _quantized_numpy(chunk, scale, dtype, label).reshape(-1)


def _imported_i32_state(model: AtomicNNUEV3) -> Optional[_ImportedI32State]:
    state = getattr(model, "_atomic_v3_imported_i32", None)
    return state if isinstance(state, _ImportedI32State) else None


def _iter_quantized_hm_psqt(model: AtomicNNUEV3) -> Iterator[np.ndarray]:
    state = _imported_i32_state(model)
    preserved = None if state is None else state.hm_psqt.reshape(-1)
    offset = 0
    for chunk in _iter_hm_chunks(
        model, ACCUMULATOR_DIMENSIONS, HM_OUTPUT_DIMENSIONS
    ):
        count = chunk.numel()
        raw = None if preserved is None else preserved[offset : offset + count]
        yield _quantized_numpy_preserving_i32(
            chunk, PSQT_WEIGHT_SCALE, "HM PSQT", raw
        ).reshape(-1)
        offset += count
    expected = HM_PHYSICAL_DIMENSIONS * PSQT_BUCKETS
    if offset != expected:
        raise AtomicV3FormatError(
            f"HM PSQT yielded {offset} values, expected {expected}"
        )


def _write_compressed_quantized(
    stream: BinaryIO,
    chunks: Iterable[torch.Tensor],
    count: int,
    scale: float,
    dtype: np.dtype | str,
    label: str,
) -> None:
    _write_compressed_arrays(
        stream, _iter_quantized(chunks, scale, dtype, label), count, label
    )


def _write_compressed_arrays(
    stream: BinaryIO,
    arrays: Iterable[np.ndarray],
    count: int,
    label: str,
) -> None:
    with tempfile.SpooledTemporaryFile(max_size=8 << 20) as encoded:
        byte_count = 0
        value_count = 0
        for array in arrays:
            array = np.asarray(array)
            _signed_dtype(array.dtype)
            payload = _encode_numpy_signed_leb128(array)
            encoded.write(payload)
            byte_count += len(payload)
            value_count += array.size
            if byte_count > 0xFFFFFFFF:
                raise AtomicV3FormatError(
                    f"{label} compressed payload exceeds uint32 length"
                )
        if value_count != count:
            raise AtomicV3FormatError(
                f"{label} yielded {value_count} values, expected {count}"
            )
        _write_all(stream, LEB128_MAGIC, f"{label} compressed magic")
        _write_u32(stream, byte_count)
        encoded.seek(0)
        shutil.copyfileobj(encoded, stream, length=STREAM_CHUNK_BYTES)


def _write_raw_i8_quantized(
    stream: BinaryIO,
    chunks: Iterable[torch.Tensor],
    count: int,
    scale: float,
    label: str,
) -> None:
    value_count = 0
    for array in _iter_quantized(chunks, scale, "i1", label):
        _write_all(stream, array.tobytes(order="C"), label)
        value_count += array.size
    if value_count != count:
        raise AtomicV3FormatError(
            f"{label} yielded {value_count} values, expected {count}"
        )


def _validate_tensor(
    label: str, tensor: object, expected_shape: tuple[int, ...]
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise AtomicV3FormatError(f"{label} is not a tensor")
    if tensor.layout != torch.strided:
        raise AtomicV3FormatError(f"{label} must be a strided tensor")
    if tensor.dtype != torch.float32:
        raise AtomicV3FormatError(f"{label} must use torch.float32")
    if tuple(tensor.shape) != expected_shape:
        raise AtomicV3FormatError(
            f"{label} shape {tuple(tensor.shape)} does not match {expected_shape}"
        )
    return tensor


def _validate_model_shapes(model: AtomicNNUEV3) -> None:
    try:
        tensors = (
            (
                "feature-transformer bias",
                model.feature_transformer.bias,
                (ACCUMULATOR_DIMENSIONS,),
            ),
            (
                "HM bucket weight",
                model.feature_transformer.hm_bucket_weight,
                (HM_TRAINING_DIMENSIONS, HM_OUTPUT_DIMENSIONS),
            ),
            (
                "HM virtual weight",
                model.feature_transformer.hm_virtual_weight,
                (HM_VIRTUAL_DIMENSIONS, HM_OUTPUT_DIMENSIONS),
            ),
            (
                "CapturePair weight",
                model.feature_transformer.capture_pair_weight,
                (CAPTURE_PAIR_DIMENSIONS, ACCUMULATOR_DIMENSIONS),
            ),
            (
                "KingBlastEP weight",
                model.feature_transformer.king_blast_ep_weight,
                (KING_BLAST_EP_DIMENSIONS, ACCUMULATOR_DIMENSIONS),
            ),
            (
                "BlastRing weight",
                model.feature_transformer.blast_ring_weight,
                (BLAST_RING_DIMENSIONS, ACCUMULATOR_DIMENSIONS),
            ),
            (
                "fc0 weight",
                model.network.fc0.linear.weight,
                (LAYER_STACKS * FC0_OUTPUTS, FC0_INPUTS),
            ),
            (
                "fc0 bias",
                model.network.fc0.linear.bias,
                (LAYER_STACKS * FC0_OUTPUTS,),
            ),
            (
                "fc0 factorized weight",
                model.network.fc0.factorized_linear.weight,
                (FC0_OUTPUTS, FC0_INPUTS),
            ),
            (
                "fc0 factorized bias",
                model.network.fc0.factorized_linear.bias,
                (FC0_OUTPUTS,),
            ),
            (
                "fc1 weight",
                model.network.fc1.linear.weight,
                (LAYER_STACKS * FC1_OUTPUTS, FC1_INPUTS),
            ),
            (
                "fc1 bias",
                model.network.fc1.linear.bias,
                (LAYER_STACKS * FC1_OUTPUTS,),
            ),
            (
                "fc2 weight",
                model.network.fc2.linear.weight,
                (LAYER_STACKS * FC2_OUTPUTS, FC2_INPUTS),
            ),
            (
                "fc2 bias",
                model.network.fc2.linear.bias,
                (LAYER_STACKS * FC2_OUTPUTS,),
            ),
        )
    except AttributeError as error:
        raise AtomicV3FormatError("model is missing an Atomic V3 contract tensor") from error
    validated = {
        label: _validate_tensor(label, tensor, shape)
        for label, tensor, shape in tensors
    }
    if validated["HM bucket weight"].device != validated["HM virtual weight"].device:
        raise AtomicV3FormatError("HM factor tensors must use the same device")
    if validated["fc0 weight"].device != validated["fc0 factorized weight"].device:
        raise AtomicV3FormatError("fc0 weight factors must use the same device")
    if validated["fc0 bias"].device != validated["fc0 factorized bias"].device:
        raise AtomicV3FormatError("fc0 bias factors must use the same device")


def _dense_bounds(bias: np.ndarray, weight: np.ndarray) -> _DenseBounds:
    bias64 = np.asarray(bias, dtype=np.int64).reshape(-1)
    weight64 = np.asarray(weight, dtype=np.int64).reshape(bias64.size, -1)
    absolute = np.abs(bias64) + 127 * np.abs(weight64).sum(axis=1)
    lower = bias64 + 127 * np.minimum(weight64, 0).sum(axis=1)
    upper = bias64 + 127 * np.maximum(weight64, 0).sum(axis=1)
    return _DenseBounds(absolute, lower, upper)


def _validate_dense_summaries(summaries: Iterable[_DenseSummary]) -> None:
    for summary in summaries:
        for label, bounds in (("fc0", summary.fc0), ("fc1", summary.fc1)):
            failed = np.flatnonzero(bounds.absolute > INT32_MAX)
            if failed.size:
                raise AtomicV3FormatError(
                    f"dense bucket {summary.bucket} {label} output "
                    f"{int(failed[0])} exceeds the signed-i32 affine envelope"
                )
        if summary.fc2.absolute[0] > INT32_MAX:
            raise AtomicV3FormatError(
                f"dense bucket {summary.bucket} fc2 exceeds the signed-i32 affine envelope"
            )
        lower = (
            int(summary.fc2.lower[0])
            + int(summary.fc0.lower[30])
            - int(summary.fc0.upper[31])
        )
        upper = (
            int(summary.fc2.upper[0])
            + int(summary.fc0.upper[30])
            - int(summary.fc0.lower[31])
        )
        if lower < RAW_OUTPUT_MIN or upper > RAW_OUTPUT_MAX:
            raise AtomicV3FormatError(
                f"dense bucket {summary.bucket} skip/output envelope exceeded"
            )


def _observe_psqt(
    heaps: list[list[int]], offset: int, values: np.ndarray
) -> bool:
    wide = np.asarray(values, dtype=np.int64).reshape(-1)
    magnitudes = np.abs(wide)
    overflow = bool(np.any(magnitudes > INT32_MAX))
    for bucket in range(PSQT_BUCKETS):
        first = (bucket - offset) % PSQT_BUCKETS
        candidates = magnitudes[first::PSQT_BUCKETS]
        heap = heaps[bucket]
        for magnitude in candidates.tolist():
            magnitude = int(magnitude)
            if len(heap) < 32:
                heapq.heappush(heap, magnitude)
            elif magnitude > heap[0]:
                heapq.heapreplace(heap, magnitude)
    return overflow


def _validate_psqt(heaps: list[list[int]], magnitude_overflow: bool) -> None:
    if magnitude_overflow:
        raise AtomicV3FormatError("HM PSQT contains INT32_MIN")
    for bucket, heap in enumerate(heaps):
        if sum(heap) > INT32_MAX:
            raise AtomicV3FormatError(
                f"HM PSQT bucket {bucket} exceeds the top-32 signed-i32 envelope"
            )


def _dense_parameters(
    model: AtomicNNUEV3, bucket: int
) -> tuple[tuple[torch.Tensor, torch.Tensor, float, float, str], ...]:
    begin = bucket * FC0_OUTPUTS
    end = begin + FC0_OUTPUTS
    fc0_weight = (
        model.network.fc0.linear.weight[begin:end]
        + model.network.fc0.factorized_linear.weight
    )
    fc0_bias = (
        model.network.fc0.linear.bias[begin:end]
        + model.network.fc0.factorized_linear.bias
    )
    begin1 = bucket * FC1_OUTPUTS
    end1 = begin1 + FC1_OUTPUTS
    begin2 = bucket * FC2_OUTPUTS
    end2 = begin2 + FC2_OUTPUTS
    return (
        (fc0_weight, fc0_bias, FC0_WEIGHT_SCALE, FC0_BIAS_SCALE, "fc0"),
        (
            model.network.fc1.linear.weight[begin1:end1],
            model.network.fc1.linear.bias[begin1:end1],
            FC1_WEIGHT_SCALE,
            FC1_BIAS_SCALE,
            "fc1",
        ),
        (
            model.network.fc2.linear.weight[begin2:end2],
            model.network.fc2.linear.bias[begin2:end2],
            FC2_WEIGHT_SCALE,
            FC2_BIAS_SCALE,
            "fc2",
        ),
    )


def _preserved_dense_bias(
    model: AtomicNNUEV3, bucket: int, layer: int
) -> Optional[np.ndarray]:
    state = _imported_i32_state(model)
    if state is None:
        return None
    return state.dense_biases[bucket][layer]


def _preflight_model(model: AtomicNNUEV3) -> None:
    _validate_model_shapes(model)
    sections = (
        (
            _iter_tensor_chunks(model.feature_transformer.bias),
            FT_ONE,
            "<i2",
            "feature-transformer bias",
        ),
        (_iter_hm_chunks(model, 0, ACCUMULATOR_DIMENSIONS), FT_ONE, "<i2", "HM"),
        (
            _iter_tensor_chunks(model.feature_transformer.capture_pair_weight),
            FT_ONE,
            "i1",
            "CapturePair",
        ),
        (
            _iter_tensor_chunks(model.feature_transformer.king_blast_ep_weight),
            FT_ONE,
            "<i2",
            "KingBlastEP",
        ),
        (
            _iter_tensor_chunks(model.feature_transformer.blast_ring_weight),
            FT_ONE,
            "i1",
            "BlastRing",
        ),
    )
    for chunks, scale, dtype, label in sections:
        for _ in _iter_quantized(chunks, scale, dtype, label):
            pass

    psqt_heaps: list[list[int]] = [[] for _ in range(PSQT_BUCKETS)]
    psqt_overflow = False
    psqt_offset = 0
    for array in _iter_quantized_hm_psqt(model):
        psqt_overflow |= _observe_psqt(psqt_heaps, psqt_offset, array)
        psqt_offset += array.size
    _validate_psqt(psqt_heaps, psqt_overflow)

    summaries: list[_DenseSummary] = []
    for bucket in range(LAYER_STACKS):
        quantized = []
        for layer_index, (
            weight,
            bias,
            weight_scale,
            bias_scale,
            label,
        ) in enumerate(_dense_parameters(model, bucket)):
            bias_array = _quantized_numpy_preserving_i32(
                bias,
                bias_scale,
                f"bucket {bucket} {label} bias",
                _preserved_dense_bias(model, bucket, layer_index),
            )
            weight_array = _quantized_numpy(
                weight, weight_scale, "i1", f"bucket {bucket} {label} weight"
            )
            quantized.append(_dense_bounds(bias_array, weight_array))
        summaries.append(_DenseSummary(bucket, *quantized))
    _validate_dense_summaries(summaries)


def _write_fc_layer(
    stream: BinaryIO,
    weight: torch.Tensor,
    bias: torch.Tensor,
    weight_scale: float,
    bias_scale: float,
    label: str,
    preserved_bias: Optional[np.ndarray] = None,
) -> None:
    bias_array = _quantized_numpy_preserving_i32(
        bias, bias_scale, f"{label} bias", preserved_bias
    )
    weight_array = _quantized_numpy(weight, weight_scale, "i1", f"{label} weight")
    _write_all(stream, bias_array.tobytes(order="C"), f"{label} bias")
    _write_all(stream, weight_array.tobytes(order="C"), f"{label} weight")


def write_nnue(
    stream: BinaryIO,
    model: AtomicNNUEV3,
    description: str | bytes | bytearray | memoryview = DEFAULT_DESCRIPTION,
) -> None:
    if not isinstance(model, AtomicNNUEV3):
        raise TypeError("Atomic V3 writer accepts only AtomicNNUEV3 models")
    encoded_description = _description_bytes(description)
    # This full pass is intentional: a range or numeric-envelope failure must
    # not leave a plausible-looking header in the destination stream.
    _preflight_model(model)

    write_header(stream, encoded_description)
    _write_u32(stream, FEATURE_TRANSFORMER_HASH)
    transformer = model.feature_transformer
    _write_compressed_quantized(
        stream,
        _iter_tensor_chunks(transformer.bias),
        ACCUMULATOR_DIMENSIONS,
        FT_ONE,
        "<i2",
        "feature-transformer bias",
    )
    _write_compressed_quantized(
        stream,
        _iter_hm_chunks(model, 0, ACCUMULATOR_DIMENSIONS),
        HM_PHYSICAL_DIMENSIONS * ACCUMULATOR_DIMENSIONS,
        FT_ONE,
        "<i2",
        "HM",
    )
    _write_raw_i8_quantized(
        stream,
        _iter_tensor_chunks(transformer.capture_pair_weight),
        CAPTURE_PAIR_DIMENSIONS * ACCUMULATOR_DIMENSIONS,
        FT_ONE,
        "CapturePair",
    )
    _write_compressed_quantized(
        stream,
        _iter_tensor_chunks(transformer.king_blast_ep_weight),
        KING_BLAST_EP_DIMENSIONS * ACCUMULATOR_DIMENSIONS,
        FT_ONE,
        "<i2",
        "KingBlastEP",
    )
    _write_raw_i8_quantized(
        stream,
        _iter_tensor_chunks(transformer.blast_ring_weight),
        BLAST_RING_DIMENSIONS * ACCUMULATOR_DIMENSIONS,
        FT_ONE,
        "BlastRing",
    )
    _write_compressed_arrays(
        stream,
        _iter_quantized_hm_psqt(model),
        HM_PHYSICAL_DIMENSIONS * PSQT_BUCKETS,
        "HM PSQT",
    )
    for bucket in range(LAYER_STACKS):
        _write_u32(stream, ARCHITECTURE_HASH)
        for layer_index, (
            weight,
            bias,
            weight_scale,
            bias_scale,
            label,
        ) in enumerate(_dense_parameters(model, bucket)):
            _write_fc_layer(
                stream,
                weight,
                bias,
                weight_scale,
                bias_scale,
                f"bucket {bucket} {label}",
                _preserved_dense_bias(model, bucket, layer_index),
            )


def _copy_flat_values(
    target: torch.Tensor, offset: int, values: np.ndarray, scale: float
) -> None:
    if values.size == 0:
        return
    converted = torch.from_numpy(values.astype(np.float32)).div_(float(scale))
    if target.ndim == 1:
        target[offset : offset + converted.numel()].copy_(converted)
        return
    if target.ndim != 2:
        raise AtomicV3FormatError("streaming target must be one- or two-dimensional")
    columns = target.shape[1]
    cursor = 0
    row, column = divmod(offset, columns)
    if column:
        width = min(columns - column, converted.numel())
        target[row, column : column + width].copy_(converted[:width])
        cursor += width
        row += 1
    complete_rows = (converted.numel() - cursor) // columns
    if complete_rows:
        end = cursor + complete_rows * columns
        target[row : row + complete_rows].copy_(
            converted[cursor:end].reshape(complete_rows, columns)
        )
        cursor = end
        row += complete_rows
    if cursor < converted.numel():
        target[row, : converted.numel() - cursor].copy_(converted[cursor:])


def _copy_physical_hm_values(
    target: torch.Tensor,
    column_begin: int,
    columns: int,
    offset: int,
    values: np.ndarray,
    scale: float,
) -> None:
    converted = torch.from_numpy(values.astype(np.float32)).div_(float(scale))
    cursor = 0
    physical_row, column = divmod(offset, columns)
    if column:
        width = min(columns - column, converted.numel())
        training_row = int(_HM_PHYSICAL_TO_TRAINING[physical_row])
        target[
            training_row,
            column_begin + column : column_begin + column + width,
        ].copy_(converted[:width])
        cursor += width
        physical_row += 1
        column = 0
    complete_rows = (converted.numel() - cursor) // columns
    if complete_rows:
        end = cursor + complete_rows * columns
        indices = torch.from_numpy(
            _HM_PHYSICAL_TO_TRAINING[
                physical_row : physical_row + complete_rows
            ].copy()
        ).to(torch.long)
        destination = target[:, column_begin : column_begin + columns]
        destination.index_copy_(
            0, indices, converted[cursor:end].reshape(complete_rows, columns)
        )
        cursor = end
        physical_row += complete_rows
    if cursor < converted.numel():
        width = converted.numel() - cursor
        training_row = int(_HM_PHYSICAL_TO_TRAINING[physical_row])
        target[training_row, column_begin : column_begin + width].copy_(
            converted[cursor:]
        )


def _read_raw_i8(
    stream: BinaryIO,
    count: int,
    label: str,
    consumer: Optional[Callable[[int, np.ndarray], None]] = None,
) -> None:
    offset = 0
    while offset < count:
        block = _read_exact(
            stream, min(STREAM_CHUNK_BYTES, count - offset), label
        )
        values = np.frombuffer(block, dtype="i1")
        if consumer is not None:
            consumer(offset, values)
        offset += values.size


def _read_fc_layer(
    stream: BinaryIO,
    outputs: int,
    inputs: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray, _DenseBounds]:
    bias = np.frombuffer(
        _read_exact(stream, outputs * 4, f"{label} bias"), dtype="<i4"
    ).copy()
    weight = np.frombuffer(
        _read_exact(stream, outputs * inputs, f"{label} weight"), dtype="i1"
    ).copy().reshape(outputs, inputs)
    return weight, bias, _dense_bounds(bias, weight)


def _copy_dense_layer(
    layer: object,
    bucket: int,
    outputs: int,
    weight: np.ndarray,
    bias: np.ndarray,
    weight_scale: float,
    bias_scale: float,
) -> None:
    begin = bucket * outputs
    end = begin + outputs
    layer.linear.weight[begin:end].copy_(
        torch.from_numpy(weight.astype(np.float32)).div_(float(weight_scale))
    )
    layer.linear.bias[begin:end].copy_(
        torch.from_numpy(bias.astype(np.float32)).div_(float(bias_scale))
    )


def _parse_body(stream: BinaryIO, model: Optional[AtomicNNUEV3]) -> None:
    transformer = model.feature_transformer if model is not None else None
    _read_compressed_chunks(
        stream,
        ACCUMULATOR_DIMENSIONS,
        "<i2",
        "feature-transformer bias",
        None
        if transformer is None
        else lambda offset, values: _copy_flat_values(
            transformer.bias, offset, values, FT_ONE
        ),
    )
    _read_compressed_chunks(
        stream,
        HM_PHYSICAL_DIMENSIONS * ACCUMULATOR_DIMENSIONS,
        "<i2",
        "HM",
        None
        if transformer is None
        else lambda offset, values: _copy_physical_hm_values(
            transformer.hm_bucket_weight,
            0,
            ACCUMULATOR_DIMENSIONS,
            offset,
            values,
            FT_ONE,
        ),
    )
    _read_raw_i8(
        stream,
        CAPTURE_PAIR_DIMENSIONS * ACCUMULATOR_DIMENSIONS,
        "CapturePair",
        None
        if transformer is None
        else lambda offset, values: _copy_flat_values(
            transformer.capture_pair_weight, offset, values, FT_ONE
        ),
    )
    _read_compressed_chunks(
        stream,
        KING_BLAST_EP_DIMENSIONS * ACCUMULATOR_DIMENSIONS,
        "<i2",
        "KingBlastEP",
        None
        if transformer is None
        else lambda offset, values: _copy_flat_values(
            transformer.king_blast_ep_weight, offset, values, FT_ONE
        ),
    )
    _read_raw_i8(
        stream,
        BLAST_RING_DIMENSIONS * ACCUMULATOR_DIMENSIONS,
        "BlastRing",
        None
        if transformer is None
        else lambda offset, values: _copy_flat_values(
            transformer.blast_ring_weight, offset, values, FT_ONE
        ),
    )

    psqt_heaps: list[list[int]] = [[] for _ in range(PSQT_BUCKETS)]
    psqt_overflow = False
    imported_psqt = (
        None
        if model is None
        else np.empty(HM_PHYSICAL_DIMENSIONS * PSQT_BUCKETS, dtype="<i4")
    )

    def consume_psqt(offset: int, values: np.ndarray) -> None:
        nonlocal psqt_overflow
        psqt_overflow |= _observe_psqt(psqt_heaps, offset, values)
        if transformer is not None:
            _copy_physical_hm_values(
                transformer.hm_bucket_weight,
                ACCUMULATOR_DIMENSIONS,
                PSQT_BUCKETS,
                offset,
                values,
                PSQT_WEIGHT_SCALE,
            )
        if imported_psqt is not None:
            imported_psqt[offset : offset + values.size] = values

    _read_compressed_chunks(
        stream,
        HM_PHYSICAL_DIMENSIONS * PSQT_BUCKETS,
        "<i4",
        "HM PSQT",
        consume_psqt,
    )

    dense_summaries: list[_DenseSummary] = []
    imported_dense_biases: list[
        tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = []
    if model is not None:
        model.network.fc0.factorized_linear.weight.zero_()
        model.network.fc0.factorized_linear.bias.zero_()
    for bucket in range(LAYER_STACKS):
        architecture_hash = _read_u32(
            stream, f"architecture hash for bucket {bucket}"
        )
        if architecture_hash != ARCHITECTURE_HASH:
            raise AtomicV3FormatError(
                f"incompatible architecture hash in bucket {bucket}"
            )
        fc0_weight, fc0_bias, fc0_bounds = _read_fc_layer(
            stream, FC0_OUTPUTS, FC0_INPUTS, f"bucket {bucket} fc0"
        )
        fc1_weight, fc1_bias, fc1_bounds = _read_fc_layer(
            stream, FC1_OUTPUTS, FC1_INPUTS, f"bucket {bucket} fc1"
        )
        fc2_weight, fc2_bias, fc2_bounds = _read_fc_layer(
            stream, FC2_OUTPUTS, FC2_INPUTS, f"bucket {bucket} fc2"
        )
        dense_summaries.append(
            _DenseSummary(bucket, fc0_bounds, fc1_bounds, fc2_bounds)
        )
        if model is not None:
            imported_dense_biases.append(
                (
                    _readonly_i32(fc0_bias),
                    _readonly_i32(fc1_bias),
                    _readonly_i32(fc2_bias),
                )
            )
        if model is not None:
            _copy_dense_layer(
                model.network.fc0,
                bucket,
                FC0_OUTPUTS,
                fc0_weight,
                fc0_bias,
                FC0_WEIGHT_SCALE,
                FC0_BIAS_SCALE,
            )
            _copy_dense_layer(
                model.network.fc1,
                bucket,
                FC1_OUTPUTS,
                fc1_weight,
                fc1_bias,
                FC1_WEIGHT_SCALE,
                FC1_BIAS_SCALE,
            )
            _copy_dense_layer(
                model.network.fc2,
                bucket,
                FC2_OUTPUTS,
                fc2_weight,
                fc2_bias,
                FC2_WEIGHT_SCALE,
                FC2_BIAS_SCALE,
            )

    trailing = stream.read(1)
    if trailing not in (b"", None):
        raise AtomicV3FormatError("trailing bytes after Atomic V3 network")
    # Match the engine transaction: numeric rejection occurs only after the
    # complete canonical stream and strict EOF have been established.
    _validate_psqt(psqt_heaps, psqt_overflow)
    _validate_dense_summaries(dense_summaries)
    if model is not None:
        if imported_psqt is None or len(imported_dense_biases) != LAYER_STACKS:
            raise AssertionError("Atomic V3 imported i32 state is incomplete")
        object.__setattr__(
            model,
            "_atomic_v3_imported_i32",
            _ImportedI32State(
                _readonly_i32(imported_psqt).reshape(
                    HM_PHYSICAL_DIMENSIONS, PSQT_BUCKETS
                ),
                tuple(imported_dense_biases),
            ),
        )


def _read_prefix(stream: BinaryIO) -> bytes:
    description = read_header(stream)
    transformer_hash = _read_u32(stream, "feature-transformer hash")
    if transformer_hash != FEATURE_TRANSFORMER_HASH:
        raise AtomicV3FormatError("incompatible feature-transformer hash")
    return description


def check_nnue(stream: BinaryIO) -> WireMetadata:
    """Validate one V3 stream with bounded memory, without allocating a model."""

    reader = _HashingReader(stream)
    description = _read_prefix(reader)
    _parse_body(reader, None)
    return WireMetadata(
        description=description,
        size=reader.size,
        sha256=reader.digest.hexdigest().upper(),
    )


def read_nnue(stream: BinaryIO) -> tuple[AtomicNNUEV3, bytes]:
    """Strictly import a V3 network into the canonical factorized trainer form."""

    # Reject incompatible headers before allocating the roughly 306 MiB
    # production graph.
    description = _read_prefix(stream)
    model = AtomicNNUEV3(initialize=False)
    with torch.no_grad():
        model.feature_transformer.hm_virtual_weight.zero_()
        _parse_body(stream, model)
    return model, description


def save_nnue(
    path: str | os.PathLike[str],
    model: AtomicNNUEV3,
    description: str | bytes | bytearray | memoryview = DEFAULT_DESCRIPTION,
) -> WireMetadata:
    """Validate and publish a network atomically without overwriting a path.

    The temporary file is created in the destination directory.  Windows
    rename is atomic and refuses an existing destination; POSIX publication
    uses a same-filesystem hard link to retain the same no-overwrite property.
    This API intentionally fails rather than falling back to ``os.replace``.
    """

    target = Path(path).expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(target):
        raise FileExistsError(f"Atomic V3 network already exists: {target}")

    temporary: Optional[Path] = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            write_nnue(stream, model, description)
            stream.flush()
            os.fsync(stream.fileno())
        with temporary.open("rb") as stream:
            metadata = check_nnue(stream)

        try:
            if os.name == "nt":
                os.rename(temporary, target)
                temporary = None
                return metadata
            os.link(temporary, target)
        except FileExistsError:
            raise FileExistsError(
                f"Atomic V3 network already exists: {target}"
            ) from None
        except OSError as error:
            raise AtomicV3FormatError(
                "atomic no-overwrite publication failed on the destination filesystem"
            ) from error
        temporary.unlink()
        temporary = None
        return metadata
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


assert HM_PHYSICAL_ROWS_PER_BUCKET == 11 * 64
assert HM_PHYSICAL_DIMENSIONS == HM_KING_BUCKETS * HM_PHYSICAL_ROWS_PER_BUCKET
assert HM_TRAINING_DIMENSIONS == HM_KING_BUCKETS * HM_ROWS_PER_BUCKET
