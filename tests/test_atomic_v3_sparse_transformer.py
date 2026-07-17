import pytest
import torch

from atomic_v3.sparse_transformer import (
    cuda_kernels_available,
    double_sparse_linear,
    sparse_linear,
)


def _inputs(device):
    indices0 = torch.tensor(
        [[0, 2, -1], [1, -1, -1]], dtype=torch.int32, device=device
    )
    values0 = torch.tensor(
        [[1.0, 0.5, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32, device=device
    )
    indices1 = torch.tensor(
        [[2, -1, -1], [0, 1, -1]], dtype=torch.int32, device=device
    )
    values1 = torch.tensor(
        [[1.5, 0.0, 0.0], [0.25, 0.75, 0.0]], dtype=torch.float32, device=device
    )
    return indices0, values0, indices1, values1


def test_cpu_sparse_linear_matches_explicit_sum_and_uses_dense_gradient():
    indices, values, _, _ = _inputs("cpu")
    weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 8.0)
    bias = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3, -0.4]))

    output = sparse_linear(indices, values, weight, bias)
    expected = torch.stack(
        (weight[0] + 0.5 * weight[2] + bias, 2.0 * weight[1] + bias)
    )
    torch.testing.assert_close(output, expected)

    output.sum().backward()
    assert weight.grad is not None and weight.grad.is_sparse is False
    torch.testing.assert_close(
        weight.grad,
        torch.tensor([[1.0] * 4, [2.0] * 4, [0.5] * 4]),
    )
    torch.testing.assert_close(bias.grad, torch.full((4,), 2.0))


def test_cpu_double_sparse_linear_shares_one_logical_weight_gradient():
    indices0, values0, indices1, values1 = _inputs("cpu")
    weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 7.0)
    bias = torch.nn.Parameter(torch.tensor([0.25, -0.5, 0.75, -1.0]))

    output0, output1 = double_sparse_linear(
        indices0, values0, indices1, values1, weight, bias
    )
    expected0 = sparse_linear(indices0, values0, weight, bias)
    expected1 = sparse_linear(indices1, values1, weight, bias)
    torch.testing.assert_close(output0, expected0)
    torch.testing.assert_close(output1, expected1)

    (output0.sum() + output1.sum()).backward()
    assert weight.grad is not None and weight.grad.is_sparse is False
    torch.testing.assert_close(
        weight.grad,
        torch.tensor([[1.25] * 4, [2.75] * 4, [2.0] * 4]),
    )
    torch.testing.assert_close(bias.grad, torch.full((4,), 4.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_double_kernel_matches_cpu_forward_and_backward():
    if not cuda_kernels_available():
        pytest.skip("CuPy sparse feature-transformer kernels are unavailable")

    cpu_inputs = _inputs("cpu")
    cuda_inputs = tuple(value.cuda() for value in cpu_inputs)
    initial_weight = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 13.0
    initial_bias = torch.linspace(-0.2, 0.2, 8)
    cpu_weight = torch.nn.Parameter(initial_weight.clone())
    cpu_bias = torch.nn.Parameter(initial_bias.clone())
    cuda_weight = torch.nn.Parameter(initial_weight.cuda())
    cuda_bias = torch.nn.Parameter(initial_bias.cuda())

    cpu_outputs = double_sparse_linear(*cpu_inputs, cpu_weight, cpu_bias)
    cuda_outputs = double_sparse_linear(*cuda_inputs, cuda_weight, cuda_bias)
    for actual, expected in zip(cuda_outputs, cpu_outputs):
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)

    sum(output.square().mean() for output in cpu_outputs).backward()
    sum(output.square().mean() for output in cuda_outputs).backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(cuda_weight.grad.cpu(), cpu_weight.grad, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(cuda_bias.grad.cpu(), cpu_bias.grad, rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_binary_no_bias_kernel_matches_value_aware_cpu_oracle():
    if not cuda_kernels_available():
        pytest.skip("CuPy sparse feature-transformer kernels are unavailable")

    indices0, _, indices1, _ = _inputs("cpu")
    values0 = (indices0 >= 0).to(torch.float32)
    values1 = (indices1 >= 0).to(torch.float32)
    initial_weight = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 11.0
    cpu_weight = torch.nn.Parameter(initial_weight.clone())
    cuda_weight = torch.nn.Parameter(initial_weight.cuda())
    cpu_bias = torch.zeros(8, dtype=torch.float32)
    cuda_bias = cpu_bias.cuda()

    cpu_outputs = double_sparse_linear(
        indices0,
        values0,
        indices1,
        values1,
        cpu_weight,
        cpu_bias,
        unit_values=True,
    )
    cuda_outputs = double_sparse_linear(
        indices0.cuda(),
        values0.cuda(),
        indices1.cuda(),
        values1.cuda(),
        cuda_weight,
        cuda_bias,
        unit_values=True,
    )
    for actual, expected in zip(cuda_outputs, cpu_outputs):
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)

    sum(output.square().mean() for output in cpu_outputs).backward()
    sum(output.square().mean() for output in cuda_outputs).backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(cuda_weight.grad.cpu(), cpu_weight.grad, rtol=2e-5, atol=2e-6)
    assert cpu_bias.grad is None
    assert cuda_bias.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_kernel_obeys_non_default_torch_stream_forward_and_backward():
    if not cuda_kernels_available():
        pytest.skip("CuPy sparse feature-transformer kernels are unavailable")

    indices, _, _, _ = _inputs("cuda")
    values = (indices >= 0).to(torch.float32)

    # Compile both kernels before the deliberately delayed launch below.  This
    # keeps the regression focused on stream ordering rather than NVRTC time.
    warm_weight = torch.nn.Parameter(torch.zeros((3, 8), device="cuda"))
    warm_bias = torch.nn.Parameter(torch.zeros(8, device="cuda"))
    sparse_linear(
        indices, values, warm_weight, warm_bias, unit_values=True
    ).sum().backward()
    torch.cuda.synchronize()

    weight = torch.nn.Parameter(torch.zeros((3, 8), device="cuda"))
    bias = torch.nn.Parameter(torch.zeros(8, device="cuda"))
    output_gradient = torch.zeros((2, 8), device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # CuPy's default stream used to overtake these delayed mutations.  A
        # RawKernel launched on PyTorch's current stream must observe both.
        torch.cuda._sleep(100_000_000)
        with torch.no_grad():
            weight.fill_(1.0)
            bias.fill_(2.0)
        output = sparse_linear(
            indices, values, weight, bias, unit_values=True
        )
        torch.cuda._sleep(100_000_000)
        output_gradient.fill_(3.0)
        output.backward(output_gradient)

    stream.synchronize()
    expected_output = torch.stack(
        (torch.full((8,), 4.0, device="cuda"), torch.full((8,), 3.0, device="cuda"))
    )
    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(weight.grad, torch.full_like(weight, 3.0))
    torch.testing.assert_close(bias.grad, torch.full_like(bias, 6.0))
