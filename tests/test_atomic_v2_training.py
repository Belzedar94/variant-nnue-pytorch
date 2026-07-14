import gc

import pytest
import torch

from atomic_v2.training import deterministic_cpu_one_step
from train_atomic_v2 import _validation_loss, default_batch_size


def test_production_shape_cpu_step_is_deterministic():
    first = deterministic_cpu_one_step(seed=20260714)
    gc.collect()
    second = deterministic_cpu_one_step(seed=20260714)

    assert first == second
    assert first.loss_before > 0
    assert first.loss_after <= first.loss_before
    assert len(first.parameter_sha256) == 64


class ConstantValidationModel(torch.nn.Module):
    def forward(self, *inputs):
        return torch.zeros((inputs[0].shape[0], 1), dtype=torch.float32)


def validation_batch(outcomes):
    size = len(outcomes)
    indices = torch.full((size, 1), -1, dtype=torch.int32)
    values = torch.zeros((size, 1), dtype=torch.float32)
    us = torch.ones((size, 1), dtype=torch.float32)
    return (
        us,
        1.0 - us,
        indices,
        values,
        indices.clone(),
        values.clone(),
        torch.tensor(outcomes, dtype=torch.float32).reshape(-1, 1),
        torch.zeros((size, 1), dtype=torch.float32),
        torch.zeros(size, dtype=torch.long),
        torch.zeros(size, dtype=torch.long),
    )


def test_validation_is_sample_weighted_and_honors_persisted_budget():
    model = ConstantValidationModel()
    batches = [validation_batch([0.0]), validation_batch([0.5, 0.5, 1.0])]

    loss, samples = _validation_loss(model, batches, 0.0, max_samples=3)

    # sigmoid(0) is 0.5: first sample has squared loss .25 and the next
    # two have zero loss. The fourth sample is outside the explicit budget.
    assert samples == 3
    assert loss == pytest.approx(0.25 / 3.0)
    assert model.training


def test_batch_size_default_is_cpu_appropriate():
    assert default_batch_size("cpu") == 128
    assert default_batch_size("cuda") == 16_384
