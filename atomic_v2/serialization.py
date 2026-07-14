"""Strict AtomicNNUEV2 reader/writer.

Only V2 is accepted here. Legacy V1 remains available through the historical
root serializer and is never silently converted by this package.
"""

from __future__ import annotations

import io
import shutil
import struct
import tempfile
from typing import BinaryIO, Iterable

import numpy as np
import torch

from .contract import (
    ACCUMULATOR_DIMENSIONS,
    ARCHITECTURE_HASH,
    FEATURE_DIMENSIONS,
    FEATURE_TRANSFORMER_HASH,
    FILE_VERSION,
    LAYER_STACKS,
    MAX_DESCRIPTION_BYTES,
    NETWORK_HASH,
    PSQT_BUCKETS,
)
from .model import AtomicNNUEV2
from .quantization import (
    FC0_BIAS_SCALE,
    FC0_WEIGHT_SCALE,
    FC1_BIAS_SCALE,
    FC1_WEIGHT_SCALE,
    FC2_BIAS_SCALE,
    FC2_WEIGHT_SCALE,
    FT_ONE,
    PSQT_WEIGHT_SCALE,
)


LEB128_MAGIC = b"COMPRESSED_LEB128"
DEFAULT_DESCRIPTION = (
    "AtomicNNUEV2 trained with Belzedar94/variant-nnue-pytorch atomic branch."
)


class AtomicV2FormatError(ValueError):
    """A network is truncated, malformed, non-canonical, or incompatible."""


def _signed_dtype(dtype: np.dtype | str) -> np.dtype:
    dtype = np.dtype(dtype)
    if dtype.kind != "i" or dtype.itemsize not in (1, 2, 4):
        raise TypeError("Atomic V2 SLEB arrays require signed i8, i16, or i32 values")
    return dtype.newbyteorder("<")


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


def encode_signed_leb128(values: Iterable[int] | np.ndarray, dtype: np.dtype | str) -> bytes:
    dtype = _signed_dtype(dtype)
    limits = np.iinfo(dtype)
    output = bytearray()
    for raw_value in values:
        value = int(raw_value)
        if value < limits.min or value > limits.max:
            raise AtomicV2FormatError(
                f"SLEB value {value} is out of range for {dtype.name}"
            )
        output.extend(_encode_one(value))
    return bytes(output)


def _encode_numpy_signed_leb128(values: np.ndarray) -> bytes:
    """Vectorized canonical SLEB encoder used for production-sized FT chunks."""
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
        encoded[active, column] = byte[active] | np.where(done[active], 0, 0x80).astype(
            np.uint8
        )
        lengths[done] = column + 1
        active &= ~done
        current = shifted
    if np.any(active):
        raise AtomicV2FormatError(f"signed LEB128 value is out of range for {dtype.name}")
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
        terminal_byte = None
        for used in range(1, max_bytes + 1):
            if position >= len(data):
                raise AtomicV2FormatError("truncated signed LEB128 value")
            byte = int(data[position])
            position += 1
            decoded |= (byte & 0x7F) << shift
            shift += 7
            if (byte & 0x80) == 0:
                terminal_byte = byte
                break
            if used == max_bytes:
                raise AtomicV2FormatError(
                    f"signed LEB128 value is out of range for {dtype.name}"
                )
        if terminal_byte is None:
            raise AtomicV2FormatError("truncated signed LEB128 value")
        if terminal_byte & 0x40:
            decoded |= -(1 << shift)
        if decoded < limits.min or decoded > limits.max:
            raise AtomicV2FormatError(
                f"signed LEB128 value is out of range for {dtype.name}"
            )
        encoded = bytes(data[start:position])
        if encoded != _encode_one(decoded):
            raise AtomicV2FormatError("non-canonical signed LEB128 value")
        output[index] = decoded

    if position != len(data):
        raise AtomicV2FormatError("trailing bytes in signed LEB128 payload")
    return output


def _decode_complete_numpy_sleb(payload: bytes, dtype: np.dtype) -> np.ndarray:
    """Decode a payload ending on a value boundary with bounded Numpy memory."""
    dtype = _signed_dtype(dtype)
    if not payload:
        return np.empty(0, dtype=dtype)
    data = np.frombuffer(payload, dtype=np.uint8)
    ends = np.flatnonzero((data & 0x80) == 0)
    if ends.size == 0 or int(ends[-1]) != data.size - 1:
        raise AtomicV2FormatError("truncated signed LEB128 value")
    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    lengths = ends - starts + 1
    max_bytes = (dtype.itemsize * 8 + 6) // 7
    if np.any(lengths > max_bytes):
        raise AtomicV2FormatError(
            f"signed LEB128 value is out of range for {dtype.name}"
        )

    decoded = np.zeros(ends.size, dtype=np.int64)
    for column in range(max_bytes):
        selected = lengths > column
        if not np.any(selected):
            break
        bytes_at_column = data[starts[selected] + column]
        decoded[selected] |= (bytes_at_column & 0x7F).astype(np.int64) << (7 * column)
    terminal = data[ends]
    signed = (terminal & 0x40) != 0
    decoded[signed] |= -(1 << (7 * lengths[signed]))

    limits = np.iinfo(dtype)
    if np.any(decoded < limits.min) or np.any(decoded > limits.max):
        raise AtomicV2FormatError(
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
        raise AtomicV2FormatError(
            f"signed LEB128 value is out of range for {dtype.name}"
        )
    if not np.array_equal(lengths, expected_lengths):
        raise AtomicV2FormatError("non-canonical signed LEB128 value")
    return decoded.astype(dtype, copy=False)


def _write_all(stream: BinaryIO, data: bytes | bytearray | memoryview, label: str) -> None:
    written = stream.write(data)
    if written is not None and written != len(data):
        raise AtomicV2FormatError(f"short write while writing {label}")


def _read_exact(stream: BinaryIO, size: int, label: str) -> bytes:
    data = stream.read(size)
    if data is None or len(data) != size:
        raise AtomicV2FormatError(f"truncated {label}")
    return data


def _read_u32(stream: BinaryIO, label: str) -> int:
    return struct.unpack("<I", _read_exact(stream, 4, label))[0]


def _write_u32(stream: BinaryIO, value: int) -> None:
    _write_all(stream, struct.pack("<I", value), "integer")


def write_compressed_array(stream: BinaryIO, values: np.ndarray) -> None:
    values = np.asarray(values)
    dtype = _signed_dtype(values.dtype)
    payload = encode_signed_leb128(values.reshape(-1), dtype)
    if len(payload) > 0xFFFFFFFF:
        raise AtomicV2FormatError("compressed payload exceeds the V2 uint32 length")
    _write_all(stream, LEB128_MAGIC, "compressed magic")
    _write_u32(stream, len(payload))
    _write_all(stream, payload, "compressed payload")


def _write_compressed_tensor(
    stream: BinaryIO,
    tensor: torch.Tensor,
    scale: float,
    dtype: np.dtype | str,
    *,
    chunk_elements: int = 1 << 18,
) -> None:
    dtype = _signed_dtype(dtype)
    limits = np.iinfo(dtype)
    source = tensor.detach()

    if source.numel() <= 0xFFFFFFFF and int(torch.count_nonzero(source).item()) == 0:
        # Canonical SLEB(0) is one zero byte. Controlled fixtures and freshly
        # zero-initialized PSQT tensors can therefore be streamed without a
        # 46-million-value Python/Numpy encoding pass or a temporary spool.
        _write_all(stream, LEB128_MAGIC, "compressed magic")
        _write_u32(stream, source.numel())
        zero_block = bytes(1 << 20)
        remaining = source.numel()
        while remaining:
            size = min(remaining, len(zero_block))
            _write_all(stream, zero_block[:size], "compressed zero payload")
            remaining -= size
        return

    def chunks():
        if source.is_contiguous():
            flat = source.reshape(-1)
            for offset in range(0, flat.numel(), chunk_elements):
                yield flat[offset : offset + chunk_elements].cpu()
            return
        if source.ndim == 2:
            # The main FT slice is [45056, :1024] of a [45056, 1032]
            # parameter. Flattening it directly would materialize an extra
            # ~184 MiB copy. Keep the wire feature-major while copying only a
            # bounded number of complete rows at a time.
            columns = source.shape[1]
            rows_per_chunk = max(1, chunk_elements // columns)
            for row in range(0, source.shape[0], rows_per_chunk):
                yield source[row : row + rows_per_chunk].contiguous().reshape(-1).cpu()
            return
        flat = source.contiguous().reshape(-1)
        for offset in range(0, flat.numel(), chunk_elements):
            yield flat[offset : offset + chunk_elements].cpu()

    with tempfile.SpooledTemporaryFile(max_size=8 << 20) as encoded:
        byte_count = 0
        for chunk in chunks():
            rounded = torch.round(chunk * scale)
            if not torch.isfinite(rounded).all():
                raise AtomicV2FormatError("network contains a non-finite parameter")
            minimum = int(rounded.min().item()) if rounded.numel() else 0
            maximum = int(rounded.max().item()) if rounded.numel() else 0
            if minimum < -limits.max or maximum > limits.max:
                raise AtomicV2FormatError(
                    f"quantized parameter range [{minimum}, {maximum}] exceeds {dtype.name}"
                )
            array = rounded.numpy().astype(dtype, copy=False)
            payload = _encode_numpy_signed_leb128(array)
            encoded.write(payload)
            byte_count += len(payload)
            if byte_count > 0xFFFFFFFF:
                raise AtomicV2FormatError("compressed payload exceeds the V2 uint32 length")
        _write_all(stream, LEB128_MAGIC, "compressed magic")
        _write_u32(stream, byte_count)
        encoded.seek(0)
        shutil.copyfileobj(encoded, stream, length=1 << 20)


def read_compressed_array(
    stream: BinaryIO,
    count: int,
    dtype: np.dtype | str,
) -> np.ndarray:
    dtype = _signed_dtype(dtype)
    magic = _read_exact(stream, len(LEB128_MAGIC), "compressed magic")
    if magic != LEB128_MAGIC:
        raise AtomicV2FormatError("invalid compressed magic")
    byte_count = _read_u32(stream, "compressed payload length")
    max_bytes = (dtype.itemsize * 8 + 6) // 7
    if byte_count < count or byte_count > count * max_bytes:
        raise AtomicV2FormatError(
            f"invalid compressed length {byte_count} for {count} {dtype.name} values"
        )
    output = np.empty(count, dtype=dtype)
    output_offset = 0
    remaining = byte_count
    pending = b""
    while remaining:
        block = _read_exact(stream, min(remaining, 1 << 20), "compressed payload")
        remaining -= len(block)
        data = pending + block
        byte_view = np.frombuffer(data, dtype=np.uint8)
        terminal = np.flatnonzero((byte_view & 0x80) == 0)
        if terminal.size == 0:
            if len(data) > max_bytes:
                raise AtomicV2FormatError(
                    f"signed LEB128 value is out of range for {dtype.name}"
                )
            pending = data
            continue
        boundary = int(terminal[-1]) + 1
        complete = data[:boundary]
        pending = data[boundary:]
        if len(pending) >= max_bytes:
            raise AtomicV2FormatError(
                f"signed LEB128 value is out of range for {dtype.name}"
            )

        if complete.count(0) == len(complete):
            values = np.zeros(len(complete), dtype=dtype)
        else:
            values = _decode_complete_numpy_sleb(complete, dtype)
        end = output_offset + values.size
        if end > count:
            raise AtomicV2FormatError("trailing values in signed LEB128 payload")
        output[output_offset:end] = values
        output_offset = end

    if pending:
        raise AtomicV2FormatError("truncated signed LEB128 value")
    if output_offset != count:
        raise AtomicV2FormatError(
            f"truncated signed LEB128 payload: decoded {output_offset} of {count} values"
        )
    return output


def _copy_flat_values(
    target: torch.Tensor,
    offset: int,
    values: np.ndarray,
    scale: float,
) -> None:
    if values.size == 0:
        return
    converted = torch.from_numpy(values.astype(np.float32)).div_(scale)
    if target.ndim == 1:
        target[offset : offset + converted.numel()].copy_(converted)
        return
    if target.ndim != 2:
        raise AtomicV2FormatError("streaming tensor target must be one- or two-dimensional")
    columns = target.shape[1]
    cursor = 0
    row, column = divmod(offset, columns)
    if column:
        width = min(columns - column, converted.numel())
        target[row, column : column + width].copy_(converted[:width])
        cursor += width
        row += 1
    remaining = converted.numel() - cursor
    complete_rows = remaining // columns
    if complete_rows:
        end = cursor + complete_rows * columns
        target[row : row + complete_rows].copy_(
            converted[cursor:end].reshape(complete_rows, columns)
        )
        cursor = end
        row += complete_rows
    if cursor < converted.numel():
        target[row, : converted.numel() - cursor].copy_(converted[cursor:])


def _read_compressed_tensor_into(
    stream: BinaryIO,
    target: torch.Tensor,
    scale: float,
    dtype: np.dtype | str,
) -> None:
    dtype = _signed_dtype(dtype)
    count = target.numel()
    magic = _read_exact(stream, len(LEB128_MAGIC), "compressed magic")
    if magic != LEB128_MAGIC:
        raise AtomicV2FormatError("invalid compressed magic")
    byte_count = _read_u32(stream, "compressed payload length")
    max_bytes = (dtype.itemsize * 8 + 6) // 7
    if byte_count < count or byte_count > count * max_bytes:
        raise AtomicV2FormatError(
            f"invalid compressed length {byte_count} for {count} {dtype.name} values"
        )

    output_offset = 0
    remaining = byte_count
    pending = b""
    while remaining:
        block = _read_exact(stream, min(remaining, 1 << 20), "compressed payload")
        remaining -= len(block)
        data = pending + block
        byte_view = np.frombuffer(data, dtype=np.uint8)
        terminal = np.flatnonzero((byte_view & 0x80) == 0)
        if terminal.size == 0:
            if len(data) > max_bytes:
                raise AtomicV2FormatError(
                    f"signed LEB128 value is out of range for {dtype.name}"
                )
            pending = data
            continue
        boundary = int(terminal[-1]) + 1
        complete = data[:boundary]
        pending = data[boundary:]
        if len(pending) >= max_bytes:
            raise AtomicV2FormatError(
                f"signed LEB128 value is out of range for {dtype.name}"
            )

        if complete.count(0) == len(complete):
            value_count = len(complete)
            end = output_offset + value_count
            if end > count:
                raise AtomicV2FormatError("trailing values in signed LEB128 payload")
            target_flat_zero = target if target.ndim == 1 else None
            if target_flat_zero is not None:
                target_flat_zero[output_offset:end].zero_()
            else:
                _copy_flat_values(
                    target, output_offset, np.zeros(value_count, dtype=dtype), scale
                )
            output_offset = end
            continue

        values = _decode_complete_numpy_sleb(complete, dtype)
        end = output_offset + values.size
        if end > count:
            raise AtomicV2FormatError("trailing values in signed LEB128 payload")
        _copy_flat_values(target, output_offset, values, scale)
        output_offset = end

    if pending:
        raise AtomicV2FormatError("truncated signed LEB128 value")
    if output_offset != count:
        raise AtomicV2FormatError(
            f"truncated signed LEB128 payload: decoded {output_offset} of {count} values"
        )


def write_header(stream: BinaryIO, description: str = DEFAULT_DESCRIPTION) -> None:
    if not isinstance(description, str):
        raise TypeError("description must be text")
    encoded = description.encode("utf-8")
    if len(encoded) > MAX_DESCRIPTION_BYTES:
        raise AtomicV2FormatError("description length exceeds 1 MiB")
    _write_u32(stream, FILE_VERSION)
    _write_u32(stream, NETWORK_HASH)
    _write_u32(stream, len(encoded))
    _write_all(stream, encoded, "description")


def read_header(stream: BinaryIO) -> str:
    version = _read_u32(stream, "file version")
    if version != FILE_VERSION:
        raise AtomicV2FormatError(
            f"incompatible file version 0x{version:08X}; expected Atomic V2 0x{FILE_VERSION:08X}"
        )
    network_hash = _read_u32(stream, "network hash")
    if network_hash != NETWORK_HASH:
        raise AtomicV2FormatError(
            f"incompatible network hash 0x{network_hash:08X}; expected 0x{NETWORK_HASH:08X}"
        )
    description_length = _read_u32(stream, "description length")
    if description_length > MAX_DESCRIPTION_BYTES:
        raise AtomicV2FormatError("invalid description length")
    encoded = _read_exact(stream, description_length, "description")
    try:
        return encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AtomicV2FormatError("description is not valid UTF-8") from error


def _quantized_numpy(
    tensor: torch.Tensor,
    scale: float,
    dtype: np.dtype | str,
    label: str,
) -> np.ndarray:
    dtype = _signed_dtype(dtype)
    rounded = torch.round(tensor.detach().cpu() * scale)
    if not torch.isfinite(rounded).all():
        raise AtomicV2FormatError(f"{label} contains a non-finite parameter")
    limits = np.iinfo(dtype)
    if rounded.numel():
        minimum = int(rounded.min().item())
        maximum = int(rounded.max().item())
        if minimum < -limits.max or maximum > limits.max:
            raise AtomicV2FormatError(
                f"{label} range [{minimum}, {maximum}] exceeds {dtype.name}"
            )
    return rounded.numpy().astype(dtype, copy=False)


def _write_fc_layer(
    stream: BinaryIO,
    weight: torch.Tensor,
    bias: torch.Tensor,
    weight_scale: float,
    bias_scale: float,
    label: str,
) -> None:
    bias_array = _quantized_numpy(bias, bias_scale, "<i4", f"{label} bias")
    weight_array = _quantized_numpy(weight, weight_scale, "i1", f"{label} weight")
    _write_all(stream, bias_array.tobytes(order="C"), f"{label} bias")
    _write_all(stream, weight_array.tobytes(order="C"), f"{label} weight")


def _validate_model_shapes(model: AtomicNNUEV2) -> None:
    try:
        tensors = (
            (
                "feature-transformer weight",
                model.feature_transformer.weight,
                (FEATURE_DIMENSIONS, ACCUMULATOR_DIMENSIONS + PSQT_BUCKETS),
            ),
            (
                "feature-transformer bias",
                model.feature_transformer.bias,
                (ACCUMULATOR_DIMENSIONS,),
            ),
            ("fc0 weight", model.network.fc0.linear.weight, (LAYER_STACKS * 32, 1024)),
            ("fc0 bias", model.network.fc0.linear.bias, (LAYER_STACKS * 32,)),
            (
                "fc0 factorized weight",
                model.network.fc0.factorized_linear.weight,
                (32, 1024),
            ),
            ("fc0 factorized bias", model.network.fc0.factorized_linear.bias, (32,)),
            ("fc1 weight", model.network.fc1.linear.weight, (LAYER_STACKS * 32, 64)),
            ("fc1 bias", model.network.fc1.linear.bias, (LAYER_STACKS * 32,)),
            ("fc2 weight", model.network.fc2.linear.weight, (LAYER_STACKS, 128)),
            ("fc2 bias", model.network.fc2.linear.bias, (LAYER_STACKS,)),
        )
    except AttributeError as error:
        raise AtomicV2FormatError("model is missing a V2 contract tensor") from error
    for label, tensor, expected_shape in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise AtomicV2FormatError(f"{label} is not a tensor")
        if tuple(tensor.shape) != expected_shape:
            raise AtomicV2FormatError(
                f"{label} shape {tuple(tensor.shape)} does not match {expected_shape}"
            )


def write_nnue(
    stream: BinaryIO,
    model: AtomicNNUEV2,
    description: str = DEFAULT_DESCRIPTION,
) -> None:
    if not isinstance(model, AtomicNNUEV2):
        raise TypeError("Atomic V2 writer accepts only AtomicNNUEV2 models")
    _validate_model_shapes(model)

    write_header(stream, description)
    _write_u32(stream, FEATURE_TRANSFORMER_HASH)
    transformer = model.feature_transformer
    _write_compressed_tensor(stream, transformer.bias, FT_ONE, "<i2")
    _write_compressed_tensor(
        stream, transformer.weight[:, :1024], FT_ONE, "<i2"
    )
    _write_compressed_tensor(
        stream, transformer.weight[:, 1024:], PSQT_WEIGHT_SCALE, "<i4"
    )

    for bucket in range(LAYER_STACKS):
        _write_u32(stream, ARCHITECTURE_HASH)
        fc0_weight, fc0_bias = model.network.fc0.bucket_parameters(bucket)
        fc1_weight, fc1_bias = model.network.fc1.bucket_parameters(bucket)
        fc2_weight, fc2_bias = model.network.fc2.bucket_parameters(bucket)
        _write_fc_layer(
            stream, fc0_weight, fc0_bias, FC0_WEIGHT_SCALE, FC0_BIAS_SCALE, "fc0"
        )
        _write_fc_layer(
            stream, fc1_weight, fc1_bias, FC1_WEIGHT_SCALE, FC1_BIAS_SCALE, "fc1"
        )
        _write_fc_layer(
            stream, fc2_weight, fc2_bias, FC2_WEIGHT_SCALE, FC2_BIAS_SCALE, "fc2"
        )


def _read_fc_layer(
    stream: BinaryIO,
    outputs: int,
    inputs: int,
    weight_scale: float,
    bias_scale: float,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    bias = np.frombuffer(_read_exact(stream, outputs * 4, f"{label} bias"), dtype="<i4").copy()
    weight = np.frombuffer(
        _read_exact(stream, outputs * inputs, f"{label} weight"), dtype="i1"
    ).copy().reshape(outputs, inputs)
    return (
        torch.from_numpy(weight.astype(np.float32)).div(weight_scale),
        torch.from_numpy(bias.astype(np.float32)).div(bias_scale),
    )


def read_nnue(stream: BinaryIO) -> tuple[AtomicNNUEV2, str]:
    # Header validation deliberately precedes the 180+ MiB production model
    # allocation, so V1 and incompatible files fail cheaply and explicitly.
    description = read_header(stream)
    transformer_hash = _read_u32(stream, "feature-transformer hash")
    if transformer_hash != FEATURE_TRANSFORMER_HASH:
        raise AtomicV2FormatError("incompatible feature-transformer hash")

    model = AtomicNNUEV2(initialize=False)
    with torch.no_grad():
        _read_compressed_tensor_into(
            stream, model.feature_transformer.bias, FT_ONE, "<i2"
        )
        _read_compressed_tensor_into(
            stream, model.feature_transformer.weight[:, :1024], FT_ONE, "<i2"
        )
        _read_compressed_tensor_into(
            stream,
            model.feature_transformer.weight[:, 1024:],
            PSQT_WEIGHT_SCALE,
            "<i4",
        )

        model.network.fc0.factorized_linear.weight.zero_()
        model.network.fc0.factorized_linear.bias.zero_()
        for bucket in range(LAYER_STACKS):
            architecture_hash = _read_u32(stream, f"architecture hash for bucket {bucket}")
            if architecture_hash != ARCHITECTURE_HASH:
                raise AtomicV2FormatError(
                    f"incompatible architecture hash in bucket {bucket}"
                )
            layers = (
                (model.network.fc0, 32, 1024, FC0_WEIGHT_SCALE, FC0_BIAS_SCALE, "fc0"),
                (model.network.fc1, 32, 64, FC1_WEIGHT_SCALE, FC1_BIAS_SCALE, "fc1"),
                (model.network.fc2, 1, 128, FC2_WEIGHT_SCALE, FC2_BIAS_SCALE, "fc2"),
            )
            for layer, outputs, inputs, weight_scale, bias_scale, label in layers:
                layer_weight, layer_bias = _read_fc_layer(
                    stream, outputs, inputs, weight_scale, bias_scale, label
                )
                begin = bucket * outputs
                end = begin + outputs
                layer.linear.weight[begin:end].copy_(layer_weight)
                layer.linear.bias[begin:end].copy_(layer_bias)

    trailing = stream.read(1)
    if trailing not in (b"", None):
        raise AtomicV2FormatError("trailing bytes after Atomic V2 network")
    return model, description


def dumps_header(description: str = DEFAULT_DESCRIPTION) -> bytes:
    stream = io.BytesIO()
    write_header(stream, description)
    return stream.getvalue()
