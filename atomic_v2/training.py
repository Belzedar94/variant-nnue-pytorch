"""Minimal deterministic CPU training primitive for AtomicNNUEV2."""

from __future__ import annotations

import gc
import hashlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import validate_batch
from .model import AtomicNNUEV2


@dataclass(frozen=True)
class OneStepResult:
    loss_before: float
    loss_after: float
    parameter_sha256: str


def atomic_loss(
    prediction: torch.Tensor,
    outcome: torch.Tensor,
    score: torch.Tensor,
    *,
    lambda_: float = 1.0,
    score_input_scale: float = 410.0,
    score_output_scale: float = 361.0,
) -> torch.Tensor:
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")
    predicted_probability = (prediction * 600.0 / score_output_scale).sigmoid()
    evaluation_probability = (score / score_input_scale).sigmoid()
    evaluation_loss = (evaluation_probability - predicted_probability).square().mean()
    result_loss = (predicted_probability - outcome).square().mean()
    return lambda_ * evaluation_loss + (1.0 - lambda_) * result_loss


def create_optimizer(model: AtomicNNUEV2, learning_rate: float = 1.5e-3) -> torch.optim.Optimizer:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    output_parameters = list(model.network.fc2.parameters())
    output_ids = {id(parameter) for parameter in output_parameters}
    main_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in output_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": main_parameters, "lr": learning_rate},
            {"params": output_parameters, "lr": learning_rate / 10.0},
        ],
        betas=(0.9, 0.999),
        eps=1e-7,
        weight_decay=0.0,
    )


def train_batch(
    model: AtomicNNUEV2,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, ...],
    *,
    lambda_: float = 1.0,
) -> float:
    (
        us,
        them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        outcome,
        score,
        psqt_indices,
        layer_stack_indices,
    ) = validate_batch(batch)
    model.clip_weights()
    prediction = model(
        us,
        them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        psqt_indices,
        layer_stack_indices,
    )
    loss = atomic_loss(prediction, outcome, score, lambda_=lambda_)
    if not torch.isfinite(loss):
        raise FloatingPointError("Atomic V2 training loss is not finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    model.clip_weights()
    return float(loss.detach())


def _synthetic_batch() -> tuple[torch.Tensor, ...]:
    us = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    them = 1.0 - us
    white_indices = torch.tensor(
        [[0, 704, 1408, -1], [63, 767, 1471, -1]], dtype=torch.int32
    )
    black_indices = torch.tensor(
        [[45055, 44351, 43647, -1], [44992, 44288, 43584, -1]], dtype=torch.int32
    )
    white_values = torch.tensor([[1.0, 1.0, 1.0, 0.0]] * 2, dtype=torch.float32)
    black_values = white_values.clone()
    psqt_indices = torch.tensor([7, 7], dtype=torch.long)
    layer_stack_indices = torch.tensor([7, 7], dtype=torch.long)
    return (
        us,
        them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        psqt_indices,
        layer_stack_indices,
    )


def _updated_parameter_digest(model: AtomicNNUEV2, active_rows: list[int]) -> str:
    digest = hashlib.sha256()
    tensors = [
        model.feature_transformer.weight.detach()[active_rows],
        model.feature_transformer.bias.detach(),
        *(parameter.detach() for parameter in model.network.parameters()),
    ]
    for tensor in tensors:
        array = tensor.cpu().contiguous().numpy().astype(np.dtype("<f4"), copy=False)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def deterministic_cpu_one_step(seed: int = 0, learning_rate: float = 1.5e-3) -> OneStepResult:
    """Run one production-shape CPU optimizer step and return stable evidence.

    The full 45,056 x 1,032 feature matrix, fake-quantized graph and production
    AdamW parameter groups are exercised rather than replaced with a toy net or
    a smoke-only optimizer.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    previous_threads = torch.get_num_threads()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(seed)
        model = AtomicNNUEV2(initialize=False).cpu()
        batch = _synthetic_batch()
        target = torch.tensor([[0.20], [-0.15]], dtype=torch.float32)
        optimizer = create_optimizer(model, learning_rate)

        model.clip_weights()
        prediction = model(*batch)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.clip_weights()

        with torch.no_grad():
            loss_after = F.mse_loss(model(*batch), target)
        active_rows = [0, 63, 704, 767, 1408, 1471, 43584, 43647, 44288, 44351, 44992, 45055]
        result = OneStepResult(
            loss_before=float(loss.detach()),
            loss_after=float(loss_after.detach()),
            parameter_sha256=_updated_parameter_digest(model, active_rows),
        )
        del optimizer, model, prediction, loss, loss_after
        gc.collect()
        return result
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
        torch.set_num_threads(previous_threads)
