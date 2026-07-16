"""Deterministic core loss and CPU fixture step for AtomicNNUEV3.

The first real optimizer/schedule is intentionally not frozen here.  H9.3l-j
uses zero-state SGD because it supports the transformer's sparse row gradients
without allocating production-sized Adam moments; the controlled training
configuration belongs to the later parameterization milestone.
"""

from __future__ import annotations

import gc
import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch

from .dataset import AtomicV3Batch, load_canonical_fixture, validate_batch
from .model import AtomicNNUEV3


@dataclass(frozen=True)
class OneStepResult:
    train_loss_before: float
    train_loss_after: float
    validation_loss_before: float
    validation_loss_after: float
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
    if prediction.shape != outcome.shape or score.shape != prediction.shape:
        raise ValueError("prediction, outcome and score must have equal shape")
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")
    if score_input_scale <= 0 or score_output_scale <= 0:
        raise ValueError("score scales must be positive")
    predicted_probability = (prediction * 600.0 / score_output_scale).sigmoid()
    evaluation_probability = (score / score_input_scale).sigmoid()
    evaluation_loss = (evaluation_probability - predicted_probability).square().mean()
    result_loss = (predicted_probability - outcome).square().mean()
    return lambda_ * evaluation_loss + (1.0 - lambda_) * result_loss


def create_core_optimizer(
    model: AtomicNNUEV3, learning_rate: float = 1.0e-3
) -> torch.optim.Optimizer:
    if not isinstance(model, AtomicNNUEV3):
        raise TypeError("Atomic V3 optimizer accepts only AtomicNNUEV3")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise TypeError("learning_rate must be a real number")
    learning_rate = float(learning_rate)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    return torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.0, weight_decay=0.0, foreach=False
    )


def batch_loss(
    model: AtomicNNUEV3,
    batch: AtomicV3Batch,
    *,
    lambda_: float = 0.5,
    validate: bool = True,
) -> torch.Tensor:
    if validate:
        validate_batch(batch)
    prediction = model(batch, validate=False)
    loss = atomic_loss(prediction, batch.outcome, batch.score, lambda_=lambda_)
    if not torch.isfinite(loss):
        raise FloatingPointError("Atomic V3 loss is not finite")
    return loss


def optimizer_step(
    model: AtomicNNUEV3,
    optimizer: torch.optim.Optimizer,
    batch: AtomicV3Batch,
    *,
    lambda_: float = 0.5,
) -> float:
    # Both the forward and the post-step state must be serializable by the
    # frozen mixed-width wire.  In particular, clipping the HM factors
    # independently is insufficient: ``clip_weights`` constrains their
    # exported bucket+virtual sums.
    model.clip_weights()
    loss = batch_loss(model, batch, lambda_=lambda_)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        values = gradient.coalesce().values() if gradient.is_sparse else gradient
        if not torch.all(torch.isfinite(values)):
            raise FloatingPointError("Atomic V3 gradient is not finite")
    optimizer.step()
    model.clip_weights()
    return float(loss.detach())


def _active_rows(batch: AtomicV3Batch, attribute: str) -> list[int]:
    rows: set[int] = set()
    for perspective in (batch.white, batch.black):
        indices = getattr(perspective, attribute).indices
        rows.update(int(value) for value in indices.reshape(-1).tolist() if value != -1)
    return sorted(rows)


def _parameter_digest(model: AtomicNNUEV3, train: AtomicV3Batch) -> str:
    transformer = model.feature_transformer
    tensors = [
        transformer.bias.detach(),
        transformer.hm_bucket_weight.detach()[_active_rows(train, "hm")],
        transformer.hm_virtual_weight.detach()[
            sorted({row % 768 for row in _active_rows(train, "hm")})
        ],
        transformer.capture_pair_weight.detach()[_active_rows(train, "capture_pair")],
        transformer.king_blast_ep_weight.detach()[_active_rows(train, "king_blast_ep")],
        transformer.blast_ring_weight.detach()[_active_rows(train, "blast_ring")],
        *(parameter.detach() for parameter in model.network.parameters()),
    ]
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.cpu().contiguous().numpy().astype(np.dtype("<f4"), copy=False)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def deterministic_cpu_one_step(
    seed: int = 20260716, learning_rate: float = 1.0e-3
) -> OneStepResult:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0 or seed > (1 << 64) - 1:
        raise ValueError("seed must be in the non-negative uint64 domain")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise TypeError("learning_rate must be a real number")
    learning_rate = float(learning_rate)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    previous_threads = torch.get_num_threads()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(seed)
        fixture = load_canonical_fixture()
        # Role order is preserved exactly as authenticated by the fixture.
        train = fixture.batch("train")
        validation = fixture.batch("validation")
        model = AtomicNNUEV3(initialize=False).cpu()
        model.clip_weights()
        optimizer = create_core_optimizer(model, learning_rate)

        train_before_tensor = batch_loss(model, train)
        with torch.no_grad():
            validation_before_tensor = batch_loss(model, validation)
        train_before = float(train_before_tensor.detach())
        validation_before = float(validation_before_tensor.detach())
        optimizer.zero_grad(set_to_none=True)
        train_before_tensor.backward()
        for parameter in model.parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            values = gradient.coalesce().values() if gradient.is_sparse else gradient
            if not torch.all(torch.isfinite(values)):
                raise FloatingPointError("Atomic V3 fixture gradient is not finite")
        optimizer.step()
        model.clip_weights()

        with torch.no_grad():
            train_after = float(batch_loss(model, train).detach())
            validation_after = float(batch_loss(model, validation).detach())
        result = OneStepResult(
            train_loss_before=train_before,
            train_loss_after=train_after,
            validation_loss_before=validation_before,
            validation_loss_after=validation_after,
            parameter_sha256=_parameter_digest(model, train),
        )
        if not all(
            np.isfinite(value)
            for value in (
                result.train_loss_before,
                result.train_loss_after,
                result.validation_loss_before,
                result.validation_loss_after,
            )
        ):
            raise FloatingPointError("Atomic V3 CPU one-step produced a non-finite metric")
        del optimizer, model, train_before_tensor, validation_before_tensor
        gc.collect()
        return result
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
        torch.set_num_threads(previous_threads)
