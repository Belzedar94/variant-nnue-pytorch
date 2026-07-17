"""Dense-gradient sparse feature-transformer primitives for AtomicNNUEV3.

The CUDA implementation is derived from the proven Stockfish NNUE CuPy
feature-transformer kernel.  It accumulates active rows directly into one
output row per position and accumulates a *dense* weight gradient with atomic
adds.  In particular it never materializes a
``[batch, active_features, outputs]`` tensor and never creates sparse PyTorch
gradients.

The Atomic V3 contract guarantees binary active features, so the production
CUDA path has an index-only specialization which does not load or multiply a
feature-value matrix.  The value-aware CUDA specialization and the portable
:func:`torch.nn.functional.embedding_bag` fallback remain available as
independent parity oracles.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import autograd


_CUPY_IMPORT_ERROR: Exception | None = None
try:
    import cupy as cp
except (ImportError, OSError, RuntimeError) as error:
    cp = None
    _CUPY_IMPORT_ERROR = error


_CUDA_PROBE_ERRORS: dict[int, Exception] = {}


@lru_cache(maxsize=None)
def _probe_cuda_kernels(device_index: int) -> bool:
    """Compile *and launch* a minimal CuPy kernel on one Torch device."""

    if cp is None or not torch.cuda.is_available():
        return False
    try:
        source = r'''
extern "C" __global__ void atomic_v3_cupy_probe(float* output) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        output[0] = 1.0f;
}
'''
        with torch.cuda.device(device_index), cp.cuda.Device(device_index):
            output = torch.zeros(1, dtype=torch.float32, device=f"cuda:{device_index}")
            kernel = cp.RawKernel(source, "atomic_v3_cupy_probe")
            kernel.compile()
            stream = cp.cuda.ExternalStream(
                torch.cuda.current_stream(device_index).cuda_stream
            )
            kernel(
                grid=(1,),
                block=(1,),
                args=(output.data_ptr(),),
                stream=stream,
            )
            torch.cuda.synchronize(device_index)
            if output.item() != 1.0:
                raise RuntimeError("CuPy probe kernel returned an invalid result")
        return True
    except Exception as error:  # Preserve the real CUDA/NVRTC diagnostic.
        _CUDA_PROBE_ERRORS[device_index] = error
        return False


def cuda_kernels_available(device: torch.device | int | None = None) -> bool:
    """Return whether a CuPy raw kernel was launched on ``device``."""

    if cp is None or not torch.cuda.is_available():
        return False
    if device is None:
        device_index = torch.cuda.current_device()
    elif isinstance(device, int):
        device_index = device
    else:
        resolved = torch.device(device)
        if resolved.type != "cuda":
            return False
        device_index = resolved.index
        if device_index is None:
            device_index = torch.cuda.current_device()
    return _probe_cuda_kernels(device_index)


def _require_cuda_kernels(device: torch.device) -> None:
    if cuda_kernels_available(device):
        return
    if cp is None:
        detail = (
            repr(_CUPY_IMPORT_ERROR)
            if _CUPY_IMPORT_ERROR is not None
            else "CuPy is absent"
        )
    else:
        device_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        detail = repr(_CUDA_PROBE_ERRORS.get(device_index, "unknown probe failure"))
    raise RuntimeError(
        "Atomic V3 CUDA sparse transformer requires a working CuPy RawKernel; "
        f"compile/launch probe failed: {detail}"
    )


def _nearest_divisor(value: int, target: int = 512) -> int:
    divisors = [candidate for candidate in range(1, value + 1) if value % candidate == 0]
    return min(divisors, key=lambda candidate: abs(candidate - target))


def _kernel_launcher(kernel, threads: int):
    def launch(grid, args, *, device: torch.device) -> None:
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError("CuPy sparse transformer kernels require a CUDA device")
        device_index = (
            resolved.index
            if resolved.index is not None
            else torch.cuda.current_device()
        )
        with torch.cuda.device(device_index), cp.cuda.Device(device_index):
            stream = cp.cuda.ExternalStream(
                torch.cuda.current_stream(device_index).cuda_stream
            )
            kernel(
                grid=grid,
                block=(threads,),
                args=args,
                stream=stream,
            )

    return launch


@lru_cache(maxsize=None)
def _forward_kernel(max_active: int, outputs: int, unit_values: bool):
    if cp is None:
        raise RuntimeError("CuPy sparse feature-transformer kernels are unavailable")
    threads = _nearest_divisor(outputs)
    output_slice = outputs // threads
    kernel_name = (
        "atomic_v3_sparse_forward_binary"
        if unit_values
        else "atomic_v3_sparse_forward_weighted"
    )
    values_parameter = "" if unit_values else "const float* const feature_values,"
    values_row = "" if unit_values else (
        "const float* const value_row = feature_values + row * MAX_ACTIVE;"
    )
    accumulate = (
        "shared_slice[column] += weight_slice[column];"
        if unit_values
        else "shared_slice[column] += weight_slice[column] * value_row[active];"
    )
    source = r"""
typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__
void KERNEL_NAME(
    const int32_t* const feature_indices,
    VALUES_PARAMETER
    const float*   const weight,
    const float*   const bias,
          float*   const output
) {
    __shared__ float shared_output[OUTPUTS];

    const uint32_t row = blockIdx.x;
    const uint32_t slice_offset = threadIdx.x * OUTPUT_SLICE;
    const int32_t* const index_row = feature_indices + row * MAX_ACTIVE;
    VALUES_ROW
    float* const shared_slice = shared_output + slice_offset;

    #pragma unroll
    for (uint32_t column = 0; column < OUTPUT_SLICE; ++column)
        shared_slice[column] = bias[slice_offset + column];

    for (uint32_t active = 0; active < MAX_ACTIVE; ++active) {
        const int32_t index = index_row[active];
        if (index == -1)
            break;
        const float* const weight_slice = weight + index * OUTPUTS + slice_offset;
        #pragma unroll
        for (uint32_t column = 0; column < OUTPUT_SLICE; ++column)
            ACCUMULATE
    }

    float* const output_slice = output + row * OUTPUTS + slice_offset;
    #pragma unroll
    for (uint32_t column = 0; column < OUTPUT_SLICE; ++column)
        output_slice[column] = shared_slice[column];
}
"""
    source = (
        source.replace("KERNEL_NAME", kernel_name)
        .replace("VALUES_PARAMETER", values_parameter)
        .replace("VALUES_ROW", values_row)
        .replace("ACCUMULATE", accumulate)
        .replace("MAX_ACTIVE", str(max_active))
        .replace("OUTPUT_SLICE", str(output_slice))
        .replace("OUTPUTS", str(outputs))
    )
    kernel = cp.RawKernel(source, kernel_name)
    kernel.compile()
    return _kernel_launcher(kernel, threads)


@lru_cache(maxsize=None)
def _backward_kernel(
    max_active: int, outputs: int, unit_values: bool, with_bias_grad: bool
):
    if cp is None:
        raise RuntimeError("CuPy sparse feature-transformer kernels are unavailable")
    threads = _nearest_divisor(outputs)
    output_slice = outputs // threads
    suffix = "binary" if unit_values else "weighted"
    suffix += "_bias" if with_bias_grad else "_no_bias"
    kernel_name = f"atomic_v3_sparse_backward_{suffix}"
    values_parameter = "" if unit_values else "const float* const feature_values,"
    values_row = "" if unit_values else (
        "const float* const value_row = feature_values + row * MAX_ACTIVE;"
    )
    bias_parameter = "float* const bias_grad," if with_bias_grad else ""
    bias_accumulation = (
        r"""
    #pragma unroll
    for (uint32_t column = 0; column < OUTPUT_SLICE; ++column) {
        const float gradient = shared_slice[column];
        if (gradient != 0.0f)
            atomicAdd(&bias_grad[slice_offset + column], gradient);
    }
"""
        if with_bias_grad
        else ""
    )
    weight_accumulation = (
        "atomicAdd(&weight_slice[column], gradient);"
        if unit_values
        else "atomicAdd(&weight_slice[column], gradient * value_row[active]);"
    )
    source = r"""
typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__
void KERNEL_NAME(
    const int32_t* const feature_indices,
    VALUES_PARAMETER
          float*   const weight_grad,
    BIAS_PARAMETER
    const float*   const output_grad
) {
    __shared__ float shared_grad[OUTPUTS];

    const uint32_t row = blockIdx.x;
    const uint32_t slice_offset = threadIdx.x * OUTPUT_SLICE;
    const int32_t* const index_row = feature_indices + row * MAX_ACTIVE;
    VALUES_ROW
    const float* const output_slice = output_grad + row * OUTPUTS + slice_offset;
    float* const shared_slice = shared_grad + slice_offset;

    #pragma unroll
    for (uint32_t column = 0; column < OUTPUT_SLICE; ++column)
        shared_slice[column] = output_slice[column];

    BIAS_ACCUMULATION

    for (uint32_t active = 0; active < MAX_ACTIVE; ++active) {
        const int32_t index = index_row[active];
        if (index == -1)
            break;
        float* const weight_slice = weight_grad + index * OUTPUTS + slice_offset;
        #pragma unroll
        for (uint32_t column = 0; column < OUTPUT_SLICE; ++column) {
            const float gradient = shared_slice[column];
            if (gradient != 0.0f)
                WEIGHT_ACCUMULATION
        }
    }
}
"""
    source = (
        source.replace("KERNEL_NAME", kernel_name)
        .replace("VALUES_PARAMETER", values_parameter)
        .replace("VALUES_ROW", values_row)
        .replace("BIAS_PARAMETER", bias_parameter)
        .replace("BIAS_ACCUMULATION", bias_accumulation)
        .replace("WEIGHT_ACCUMULATION", weight_accumulation)
        .replace("MAX_ACTIVE", str(max_active))
        .replace("OUTPUT_SLICE", str(output_slice))
        .replace("OUTPUTS", str(outputs))
    )
    kernel = cp.RawKernel(source, kernel_name)
    kernel.compile()
    return _kernel_launcher(kernel, threads)


def _validate_arguments(
    indices: torch.Tensor,
    values: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    if indices.ndim != 2 or values.shape != indices.shape:
        raise ValueError("sparse indices/values must have equal [batch, width] shape")
    if indices.dtype != torch.int32 or values.dtype != torch.float32:
        raise TypeError("sparse indices/values must use int32/float32")
    if weight.ndim != 2 or weight.dtype != torch.float32:
        raise TypeError("sparse feature-transformer weight must be float32 [rows, outputs]")
    if bias.shape != (weight.shape[1],) or bias.dtype != torch.float32:
        raise TypeError("sparse feature-transformer bias must be float32 [outputs]")
    devices = {indices.device, values.device, weight.device, bias.device}
    if len(devices) != 1:
        raise ValueError("sparse feature-transformer tensors must share one device")


def _fallback_sparse_linear(
    indices: torch.Tensor,
    values: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    _validate_arguments(indices, values, weight, bias)
    batch_size, max_active = indices.shape
    valid = indices >= 0
    safe_indices = indices.clamp_min(0).to(torch.long).reshape(-1)
    active_values = torch.where(valid, values, torch.zeros_like(values)).reshape(-1)
    offsets = torch.arange(
        0,
        batch_size * max_active,
        max_active,
        dtype=torch.long,
        device=indices.device,
    )
    output = F.embedding_bag(
        safe_indices,
        weight,
        offsets,
        mode="sum",
        per_sample_weights=active_values,
        include_last_offset=False,
        sparse=False,
    )
    return output + bias


class _CudaSparseLinear(autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, weight, bias, unit_values):
        _validate_arguments(indices, values, weight, bias)
        required = (
            (indices, weight, bias)
            if unit_values
            else (indices, values, weight, bias)
        )
        if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in required):
            raise ValueError("CUDA sparse feature-transformer tensors must be contiguous")
        if unit_values:
            ctx.save_for_backward(indices)
        else:
            ctx.save_for_backward(indices, values)
        ctx.unit_values = bool(unit_values)
        ctx.weight_shape = tuple(weight.shape)
        ctx.bias_shape = tuple(bias.shape)
        output = torch.empty(
            (indices.shape[0], weight.shape[1]),
            dtype=torch.float32,
            device=indices.device,
        )
        args = [indices.data_ptr()]
        if not unit_values:
            args.append(values.data_ptr())
        args.extend((weight.data_ptr(), bias.data_ptr(), output.data_ptr()))
        _forward_kernel(indices.shape[1], weight.shape[1], bool(unit_values))(
            grid=(indices.shape[0],),
            args=tuple(args),
            device=indices.device,
        )
        return output

    @staticmethod
    def backward(ctx, output_grad):
        indices = ctx.saved_tensors[0]
        values = None if ctx.unit_values else ctx.saved_tensors[1]
        output_grad = output_grad.contiguous()
        weight_grad = torch.zeros(
            ctx.weight_shape, dtype=torch.float32, device=output_grad.device
        )
        with_bias_grad = bool(ctx.needs_input_grad[3])
        bias_grad = (
            torch.zeros(ctx.bias_shape, dtype=torch.float32, device=output_grad.device)
            if with_bias_grad
            else None
        )
        args = [indices.data_ptr()]
        if not ctx.unit_values:
            assert values is not None
            args.append(values.data_ptr())
        args.append(weight_grad.data_ptr())
        if with_bias_grad:
            assert bias_grad is not None
            args.append(bias_grad.data_ptr())
        args.append(output_grad.data_ptr())
        _backward_kernel(
            indices.shape[1], output_grad.shape[1], ctx.unit_values, with_bias_grad
        )(
            grid=(indices.shape[0],),
            args=tuple(args),
            device=indices.device,
        )
        return None, None, weight_grad, bias_grad, None


class _CudaDoubleSparseLinear(autograd.Function):
    @staticmethod
    def forward(ctx, indices0, values0, indices1, values1, weight, bias, unit_values):
        _validate_arguments(indices0, values0, weight, bias)
        _validate_arguments(indices1, values1, weight, bias)
        if indices0.shape != indices1.shape:
            raise ValueError("double sparse feature-transformer POV shapes differ")
        tensors = (
            (indices0, indices1, weight, bias)
            if unit_values
            else (indices0, values0, indices1, values1, weight, bias)
        )
        if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors):
            raise ValueError("CUDA sparse feature-transformer tensors must be contiguous")
        if unit_values:
            ctx.save_for_backward(indices0, indices1)
        else:
            ctx.save_for_backward(indices0, values0, indices1, values1)
        ctx.unit_values = bool(unit_values)
        ctx.weight_shape = tuple(weight.shape)
        ctx.bias_shape = tuple(bias.shape)
        shape = (indices0.shape[0], weight.shape[1])
        output0 = torch.empty(shape, dtype=torch.float32, device=indices0.device)
        output1 = torch.empty_like(output0)
        kernel = _forward_kernel(indices0.shape[1], weight.shape[1], bool(unit_values))
        for indices, values, output in (
            (indices0, values0, output0),
            (indices1, values1, output1),
        ):
            args = [indices.data_ptr()]
            if not unit_values:
                args.append(values.data_ptr())
            args.extend((weight.data_ptr(), bias.data_ptr(), output.data_ptr()))
            kernel(
                grid=(indices.shape[0],),
                args=tuple(args),
                device=indices.device,
            )
        return output0, output1

    @staticmethod
    def backward(ctx, output_grad0, output_grad1):
        if ctx.unit_values:
            indices0, indices1 = ctx.saved_tensors
            values0 = values1 = None
        else:
            indices0, values0, indices1, values1 = ctx.saved_tensors
        output_grad0 = output_grad0.contiguous()
        output_grad1 = output_grad1.contiguous()
        weight_grad = torch.zeros(
            ctx.weight_shape, dtype=torch.float32, device=output_grad0.device
        )
        with_bias_grad = bool(ctx.needs_input_grad[5])
        bias_grad = (
            torch.zeros(ctx.bias_shape, dtype=torch.float32, device=output_grad0.device)
            if with_bias_grad
            else None
        )
        kernel = _backward_kernel(
            indices0.shape[1], output_grad0.shape[1], ctx.unit_values, with_bias_grad
        )
        for indices, values, output_grad in (
            (indices0, values0, output_grad0),
            (indices1, values1, output_grad1),
        ):
            args = [indices.data_ptr()]
            if not ctx.unit_values:
                assert values is not None
                args.append(values.data_ptr())
            args.append(weight_grad.data_ptr())
            if with_bias_grad:
                assert bias_grad is not None
                args.append(bias_grad.data_ptr())
            args.append(output_grad.data_ptr())
            kernel(
                grid=(indices.shape[0],),
                args=tuple(args),
                device=indices.device,
            )
        return None, None, None, None, weight_grad, bias_grad, None


def sparse_linear(
    indices: torch.Tensor,
    values: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    unit_values: bool = False,
) -> torch.Tensor:
    """Apply one sparse feature table with a dense parameter gradient."""

    if indices.is_cuda:
        _require_cuda_kernels(indices.device)
        return _CudaSparseLinear.apply(
            indices.contiguous(),
            values.contiguous(),
            weight.contiguous(),
            bias.contiguous(),
            unit_values,
        )
    return _fallback_sparse_linear(indices, values, weight, bias)


def double_sparse_linear(
    indices0: torch.Tensor,
    values0: torch.Tensor,
    indices1: torch.Tensor,
    values1: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    unit_values: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one table to both perspectives, sharing one dense gradient."""

    if indices0.is_cuda:
        _require_cuda_kernels(indices0.device)
        return _CudaDoubleSparseLinear.apply(
            indices0.contiguous(),
            values0.contiguous(),
            indices1.contiguous(),
            values1.contiguous(),
            weight.contiguous(),
            bias.contiguous(),
            unit_values,
        )
    return (
        _fallback_sparse_linear(indices0, values0, weight, bias),
        _fallback_sparse_linear(indices1, values1, weight, bias),
    )


__all__ = ["cuda_kernels_available", "double_sparse_linear", "sparse_linear"]
