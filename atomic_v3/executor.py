"""Production execution contract for the four AtomicNNUEV3 bootstrap runs.

The native Atomic BIN V2 provider is deliberately injected through a tiny
protocol.  This keeps the optimizer/checkpoint contract executable and
testable without making a second, implicit dataset implementation part of the
trainer.  A provider owns its cyclic logical cursor; the executor never resets
the training provider at an epoch boundary.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import random
from typing import Callable, Mapping, Protocol, runtime_checkable

import numpy as np
import torch

from ranger import Ranger

from .checkpoint import (
    CheckpointBinding,
    TrainingCounters,
    checkpoint_document,
    load_last_checkpoint,
    restore_checkpoint,
    save_last_checkpoint,
)
from .model import AtomicNNUEV3
from .training import batch_loss


CONFIG_FORMAT = "atomic-v3-bootstrap-run-config-v1"
EPOCHS = 37
EPOCH_SIZE = 20_000_000
VALIDATION_SIZE = 1_000_000
EFFECTIVE_BATCH_SIZE = 16_384
MICROBATCH_SIZE = 128
ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // MICROBATCH_SIZE
TRAINING_STEPS_PER_EPOCH = math.ceil(EPOCH_SIZE / EFFECTIVE_BATCH_SIZE)
TRAINING_SAMPLES_PER_EPOCH = TRAINING_STEPS_PER_EPOCH * EFFECTIVE_BATCH_SIZE
VALIDATION_BATCHES_PER_EPOCH = math.ceil(VALIDATION_SIZE / EFFECTIVE_BATCH_SIZE)
VALIDATION_SAMPLES_PER_EPOCH = VALIDATION_BATCHES_PER_EPOCH * EFFECTIVE_BATCH_SIZE
SEED = 42
RANDOM_SKIP = 3
THREADS = 1
WORKERS = 1
GPUS = 1
PRECISION = "fp32"
MAIN_LEARNING_RATE = 1.5e-3
FINAL_LEARNING_RATE = 1.5e-4
RANGER_BETAS = (0.9, 0.999)
RANGER_EPS = 1.0e-7
RANGER_ALPHA = 0.5
RANGER_K = 6
SCHEDULER_GAMMA = 0.987


class ExecutorError(ValueError):
    """A production execution invariant was violated."""


def _positive_int(label: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExecutorError(f"{label} must be a positive integer")
    return value


def _finite_unit(label: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExecutorError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ExecutorError(f"{label} must be finite and in [0, 1]")
    return result


@dataclass(frozen=True)
class LambdaSchedule:
    """A fixed value or an inclusive epoch-zero-to-epoch-36 ramp."""

    kind: str
    start: float
    end: float

    @classmethod
    def fixed(cls, value: float) -> "LambdaSchedule":
        value = _finite_unit("lambda", value)
        return cls("fixed", value, value)

    @classmethod
    def linear(cls, start: float, end: float) -> "LambdaSchedule":
        start = _finite_unit("lambda start", start)
        end = _finite_unit("lambda end", end)
        if end < start:
            raise ExecutorError("linear lambda end must not precede start")
        return cls("inclusive-linear", start, end)

    def value(self, epoch_index: int, *, epochs: int = EPOCHS) -> float:
        epochs = _positive_int("epochs", epochs)
        if isinstance(epoch_index, bool) or not isinstance(epoch_index, int):
            raise ExecutorError("epoch index must be an integer")
        if epoch_index < 0 or epoch_index >= epochs:
            raise ExecutorError("epoch index is outside the configured run")
        if self.kind == "fixed":
            if self.start != self.end:
                raise ExecutorError("fixed lambda endpoints differ")
            return _finite_unit("lambda", self.start)
        if self.kind != "inclusive-linear":
            raise ExecutorError("unsupported lambda schedule")
        start = _finite_unit("lambda start", self.start)
        end = _finite_unit("lambda end", self.end)
        if epochs == 1:
            if start != end:
                raise ExecutorError("one-epoch linear schedule has unequal endpoints")
            return start
        # Spell the endpoint cases explicitly so the frozen advertised values
        # are bit-identical, rather than merely close after interpolation.
        if epoch_index == 0:
            return start
        if epoch_index == epochs - 1:
            return end
        return start + (end - start) * (epoch_index / (epochs - 1))

    def to_document(self) -> dict[str, object]:
        # Validation is intentionally performed on every serialization.
        self.value(0)
        self.value(EPOCHS - 1)
        return {"kind": self.kind, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class ProductionRunConfig:
    run_id: str
    lambda_schedule: LambdaSchedule
    epochs: int = EPOCHS
    epoch_size: int = EPOCH_SIZE
    validation_size: int = VALIDATION_SIZE
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE
    seed: int = SEED
    random_skip: int = RANDOM_SKIP
    precision: str = PRECISION
    gpus: int = GPUS
    threads: int = THREADS
    workers: int = WORKERS

    def validate(self) -> "ProductionRunConfig":
        expected = {
            "epochs": EPOCHS,
            "epoch_size": EPOCH_SIZE,
            "validation_size": VALIDATION_SIZE,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "seed": SEED,
            "random_skip": RANDOM_SKIP,
            "precision": PRECISION,
            "gpus": GPUS,
            "threads": THREADS,
            "workers": WORKERS,
        }
        for field, expected_value in expected.items():
            if getattr(self, field) != expected_value:
                raise ExecutorError(
                    f"{field} differs from the frozen bootstrap value {expected_value!r}"
                )
        if self.run_id not in RUN_CONFIGS:
            raise ExecutorError("run_id is not one of the four bootstrap runs")
        expected_schedule = RUN_CONFIGS[self.run_id].lambda_schedule
        if self.lambda_schedule != expected_schedule:
            raise ExecutorError("lambda schedule differs from the selected bootstrap run")
        self.lambda_schedule.to_document()
        return self

    def to_document(
        self, *, microbatch_size: int = MICROBATCH_SIZE
    ) -> dict[str, object]:
        self.validate()
        microbatch_size = _positive_int("microbatch size", microbatch_size)
        if microbatch_size > self.effective_batch_size:
            raise ExecutorError("microbatch size exceeds effective batch size")
        if microbatch_size != MICROBATCH_SIZE:
            raise ExecutorError(
                f"bootstrap microbatch size must be exactly {MICROBATCH_SIZE}"
            )
        return {
            "config_format": CONFIG_FORMAT,
            "run_id": self.run_id,
            "epochs": self.epochs,
            "epoch_size": self.epoch_size,
            "validation_size": self.validation_size,
            "effective_batch_size": self.effective_batch_size,
            "microbatch_size": microbatch_size,
            "accumulation_steps": ACCUMULATION_STEPS,
            "training_steps_per_epoch": TRAINING_STEPS_PER_EPOCH,
            "training_samples_accepted_per_epoch": TRAINING_SAMPLES_PER_EPOCH,
            "validation_batches_per_epoch": VALIDATION_BATCHES_PER_EPOCH,
            "validation_samples_accepted_per_epoch": VALIDATION_SAMPLES_PER_EPOCH,
            "seed": self.seed,
            "random_skip": self.random_skip,
            "precision": self.precision,
            "gpus": self.gpus,
            "threads": self.threads,
            "workers": self.workers,
            "lambda_schedule": self.lambda_schedule.to_document(),
            "optimizer": {
                "type": "Ranger",
                "main_learning_rate": MAIN_LEARNING_RATE,
                "final_learning_rate": FINAL_LEARNING_RATE,
                "betas": list(RANGER_BETAS),
                "eps": RANGER_EPS,
                "alpha": RANGER_ALPHA,
                "k": RANGER_K,
                "weight_decay": 0.0,
                "gradient_centralization": False,
            },
            "scheduler": {
                "type": "StepLR",
                "step_size_epochs": 1,
                "gamma": SCHEDULER_GAMMA,
            },
            "dataset": {
                "source": "authenticated-bootstrap-receipt-only",
                "provenance_class": "non-publication-bootstrap",
                "training_cursor": "continuous-cyclic-across-epochs",
                "validation_cursor": "fixed-reset-every-epoch",
            },
        }


# Construct only after the class exists; validate() compares against this map.
RUN_CONFIGS: dict[str, ProductionRunConfig] = {
    "lambda-0": ProductionRunConfig("lambda-0", LambdaSchedule.fixed(0.0)),
    "lambda-025": ProductionRunConfig("lambda-025", LambdaSchedule.fixed(0.25)),
    "lambda-050": ProductionRunConfig("lambda-050", LambdaSchedule.fixed(0.5)),
    "lambda-linear-015-050": ProductionRunConfig(
        "lambda-linear-015-050", LambdaSchedule.linear(0.15, 0.5)
    ),
}


def production_config(run_id: str) -> ProductionRunConfig:
    if not isinstance(run_id, str) or run_id not in RUN_CONFIGS:
        raise ExecutorError("run_id is not one of the four bootstrap runs")
    return RUN_CONFIGS[run_id].validate()


@dataclass(frozen=True)
class ProviderBatch:
    payload: object
    samples: int

    def validate(self, maximum: int) -> "ProviderBatch":
        maximum = _positive_int("maximum provider batch", maximum)
        samples = _positive_int("provider batch samples", self.samples)
        if samples > maximum:
            raise ExecutorError("provider returned more samples than requested")
        return self


@runtime_checkable
class ResumableBatchProvider(Protocol):
    """Minimal seam to be implemented by the audited native provider later."""

    def next_batch(self, maximum_samples: int) -> ProviderBatch:
        ...

    def logical_cursor_state(self) -> Mapping[str, object]:
        ...

    def restore_logical_cursor(self, state: Mapping[str, object]) -> None:
        ...


LossFunction = Callable[[torch.nn.Module, object, float], torch.Tensor]
ClipFunction = Callable[[], None]


@dataclass(frozen=True)
class EpochMetrics:
    samples: int
    steps: int
    mean_loss: float


def seed_training(seed: int = SEED) -> None:
    if seed != SEED:
        raise ExecutorError(f"bootstrap seed must be exactly {SEED}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_production_optimizer(
    model: AtomicNNUEV3,
) -> tuple[Ranger, torch.optim.lr_scheduler.StepLR]:
    if not isinstance(model, AtomicNNUEV3):
        raise TypeError("production optimizer accepts only AtomicNNUEV3")
    final_parameters = list(model.network.fc2.parameters())
    final_identities = {id(parameter) for parameter in final_parameters}
    main_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in final_identities
    ]
    if not main_parameters or not final_parameters:
        raise ExecutorError("AtomicNNUEV3 optimizer parameter groups are incomplete")
    if len({id(parameter) for parameter in main_parameters + final_parameters}) != len(
        list(model.parameters())
    ):
        raise ExecutorError("AtomicNNUEV3 optimizer parameters are duplicated")
    optimizer = Ranger(
        [
            {"params": main_parameters, "lr": MAIN_LEARNING_RATE},
            {"params": final_parameters, "lr": FINAL_LEARNING_RATE},
        ],
        lr=MAIN_LEARNING_RATE,
        betas=RANGER_BETAS,
        eps=RANGER_EPS,
        alpha=RANGER_ALPHA,
        k=RANGER_K,
        weight_decay=0.0,
        gc_loc=False,
        use_gc=False,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=SCHEDULER_GAMMA
    )
    return optimizer, scheduler


def validate_production_optimizer(
    model: AtomicNNUEV3,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    completed_epochs: int,
) -> None:
    """Fail closed if injected state is not the frozen historical stack."""

    if not isinstance(model, AtomicNNUEV3):
        raise TypeError("production optimizer validation requires AtomicNNUEV3")
    if not isinstance(optimizer, Ranger):
        raise ExecutorError("production optimizer must be the historical Ranger")
    if not isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
        raise ExecutorError("production scheduler must be StepLR")
    if scheduler.optimizer is not optimizer:
        raise ExecutorError("production scheduler is bound to a different optimizer")
    if scheduler.step_size != 1 or scheduler.gamma != SCHEDULER_GAMMA:
        raise ExecutorError("production StepLR settings differ from contract")
    if (
        isinstance(completed_epochs, bool)
        or not isinstance(completed_epochs, int)
        or completed_epochs < 0
    ):
        raise ExecutorError("completed epochs must be a non-negative integer")
    if scheduler.last_epoch != completed_epochs:
        raise ExecutorError("scheduler epoch cursor differs from checkpoint counters")
    if list(scheduler.base_lrs) != [MAIN_LEARNING_RATE, FINAL_LEARNING_RATE]:
        raise ExecutorError("scheduler base learning rates differ from contract")
    if optimizer.use_gc is not False or optimizer.gc_loc is not False:
        raise ExecutorError("Ranger gradient centralization must be disabled")
    if optimizer.alpha != RANGER_ALPHA or optimizer.k != RANGER_K:
        raise ExecutorError("Ranger Lookahead settings differ from contract")
    if len(optimizer.param_groups) != 2:
        raise ExecutorError("Ranger must contain exactly main and FC2 groups")
    final_ids = {id(parameter) for parameter in model.network.fc2.parameters()}
    main_ids = {id(parameter) for parameter in model.parameters()} - final_ids
    actual_ids = [
        {id(parameter) for parameter in group["params"]}
        for group in optimizer.param_groups
    ]
    if actual_ids != [main_ids, final_ids]:
        raise ExecutorError("Ranger parameter groups differ from main/FC2 contract")
    expected_lr = [MAIN_LEARNING_RATE, FINAL_LEARNING_RATE]
    for _ in range(completed_epochs):
        expected_lr = [value * SCHEDULER_GAMMA for value in expected_lr]
    for index, group in enumerate(optimizer.param_groups):
        if tuple(group["betas"]) != RANGER_BETAS:
            raise ExecutorError("Ranger betas differ from contract")
        if group["eps"] != RANGER_EPS or group["weight_decay"] != 0.0:
            raise ExecutorError("Ranger epsilon/weight decay differs from contract")
        if group["alpha"] != RANGER_ALPHA or group["k"] != RANGER_K:
            raise ExecutorError("Ranger group Lookahead settings differ from contract")
        if not math.isclose(
            float(group["lr"]), expected_lr[index], rel_tol=1.0e-15, abs_tol=0.0
        ):
            raise ExecutorError("Ranger learning-rate cursor differs from schedule")


def coalesce_sparse_gradients(model: torch.nn.Module) -> int:
    count = 0
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is not None and gradient.is_sparse:
            parameter.grad = gradient.coalesce()
            count += 1
    return count


def densify_sparse_gradients(model: torch.nn.Module) -> int:
    """Densify once, immediately before the historical Ranger optimizer."""

    count = 0
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is not None and gradient.is_sparse:
            parameter.grad = gradient.coalesce().to_dense()
            count += 1
    return count


def _check_finite_gradients(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        values = gradient.coalesce().values() if gradient.is_sparse else gradient
        if not torch.all(torch.isfinite(values)):
            raise FloatingPointError("Atomic V3 gradient is not finite")


def _default_loss(model: torch.nn.Module, payload: object, lambda_value: float) -> torch.Tensor:
    if not isinstance(model, AtomicNNUEV3):
        raise TypeError("default production loss requires AtomicNNUEV3")
    return batch_loss(model, payload, lambda_=lambda_value)  # type: ignore[arg-type]


def _clip_function(model: torch.nn.Module) -> ClipFunction:
    clip = getattr(model, "clip_weights", None)
    return clip if callable(clip) else (lambda: None)


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    provider: ResumableBatchProvider,
    *,
    lambda_value: float,
    sample_budget: int,
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE,
    microbatch_size: int = MICROBATCH_SIZE,
    loss_function: LossFunction = _default_loss,
) -> EpochMetrics:
    """Train one exact sample budget with sample-weighted microbatches."""

    lambda_value = _finite_unit("lambda", lambda_value)
    sample_budget = _positive_int("training sample budget", sample_budget)
    effective_batch_size = _positive_int(
        "effective batch size", effective_batch_size
    )
    microbatch_size = _positive_int("microbatch size", microbatch_size)
    if microbatch_size > effective_batch_size:
        raise ExecutorError("microbatch size exceeds effective batch size")
    if not callable(loss_function):
        raise TypeError("loss_function must be callable")
    clip = _clip_function(model)
    model.train()
    consumed = 0
    steps = 0
    weighted_loss = 0.0
    while consumed < sample_budget:
        step_samples = min(effective_batch_size, sample_budget - consumed)
        optimizer.zero_grad(set_to_none=True)
        clip()
        accumulated = 0
        while accumulated < step_samples:
            request = min(microbatch_size, step_samples - accumulated)
            batch = provider.next_batch(request)
            if not isinstance(batch, ProviderBatch):
                raise TypeError("provider.next_batch must return ProviderBatch")
            batch.validate(request)
            loss = loss_function(model, batch.payload, lambda_value)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise TypeError("loss_function must return a scalar tensor")
            if not torch.isfinite(loss):
                raise FloatingPointError("Atomic V3 loss is not finite")
            # Each loss is a microbatch mean.  Scale by its exact contribution
            # to this optimizer step, including the final partial microbatch.
            (loss * (batch.samples / step_samples)).backward()
            coalesce_sparse_gradients(model)
            loss_value = float(loss.detach())
            weighted_loss += loss_value * batch.samples
            accumulated += batch.samples
        _check_finite_gradients(model)
        densify_sparse_gradients(model)
        _check_finite_gradients(model)
        optimizer.step()
        clip()
        consumed += step_samples
        steps += 1
    return EpochMetrics(consumed, steps, weighted_loss / consumed)


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    provider: ResumableBatchProvider,
    *,
    lambda_value: float,
    sample_budget: int,
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE,
    microbatch_size: int = MICROBATCH_SIZE,
    loss_function: LossFunction = _default_loss,
) -> EpochMetrics:
    lambda_value = _finite_unit("lambda", lambda_value)
    sample_budget = _positive_int("validation sample budget", sample_budget)
    effective_batch_size = _positive_int(
        "validation effective batch size", effective_batch_size
    )
    microbatch_size = _positive_int("validation microbatch size", microbatch_size)
    if microbatch_size > effective_batch_size:
        raise ExecutorError("validation microbatch size exceeds effective batch size")
    if not callable(loss_function):
        raise TypeError("loss_function must be callable")
    model.eval()
    consumed = 0
    batches = 0
    weighted_loss = 0.0
    while consumed < sample_budget:
        logical_samples = min(effective_batch_size, sample_budget - consumed)
        accumulated = 0
        while accumulated < logical_samples:
            request = min(microbatch_size, logical_samples - accumulated)
            batch = provider.next_batch(request)
            if not isinstance(batch, ProviderBatch):
                raise TypeError("provider.next_batch must return ProviderBatch")
            batch.validate(request)
            loss = loss_function(model, batch.payload, lambda_value)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise TypeError("loss_function must return a scalar tensor")
            if not torch.isfinite(loss):
                raise FloatingPointError("Atomic V3 validation loss is not finite")
            weighted_loss += float(loss) * batch.samples
            accumulated += batch.samples
        consumed += logical_samples
        batches += 1
    return EpochMetrics(consumed, batches, weighted_loss / consumed)


def require_production_model(model: torch.nn.Module) -> None:
    if not isinstance(model, AtomicNNUEV3):
        raise TypeError("production execution accepts only AtomicNNUEV3")
    devices = {parameter.device for parameter in model.parameters()}
    dtypes = {parameter.dtype for parameter in model.parameters()}
    if len(devices) != 1 or next(iter(devices)).type != "cuda":
        raise ExecutorError("production execution requires one CUDA device")
    if dtypes != {torch.float32}:
        raise ExecutorError("production execution requires fp32 parameters")


def reset_validation_provider(
    factory: Callable[[], ResumableBatchProvider],
    start_cursor: Mapping[str, object],
) -> ResumableBatchProvider:
    """Create validation state and force it to the same logical origin."""

    if not callable(factory):
        raise TypeError("validation provider factory must be callable")
    if not isinstance(start_cursor, Mapping):
        raise TypeError("validation start cursor must be a mapping")
    provider = factory()
    provider.restore_logical_cursor(copy.deepcopy(start_cursor))
    restored = provider.logical_cursor_state()
    if not isinstance(restored, Mapping) or dict(restored) != dict(start_cursor):
        raise ExecutorError("validation provider did not restore its fixed cursor")
    return provider


def run_production(
    config: ProductionRunConfig,
    model: AtomicNNUEV3,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    training_provider: ResumableBatchProvider,
    validation_provider_factory: Callable[[], ResumableBatchProvider],
    validation_start_cursor: Mapping[str, object],
    checkpoint_binding: CheckpointBinding,
    output_directory: str,
    *,
    microbatch_size: int = MICROBATCH_SIZE,
    resume: bool = False,
    loss_function: LossFunction = _default_loss,
) -> TrainingCounters:
    """Execute all remaining epochs, writing only rolling ``last.ckpt``.

    Checkpoints are committed at epoch boundaries.  On a mid-epoch crash the
    previous checkpoint remains valid and resumption deterministically replays
    that entire epoch from its saved RNG and provider cursor.
    """

    config.validate()
    require_production_model(model)
    microbatch_size = _positive_int("microbatch size", microbatch_size)
    if microbatch_size > config.effective_batch_size:
        raise ExecutorError("microbatch size exceeds effective batch size")
    if microbatch_size != MICROBATCH_SIZE:
        raise ExecutorError(
            f"bootstrap microbatch size must be exactly {MICROBATCH_SIZE}"
        )
    if checkpoint_binding.config_document() != config.to_document(
        microbatch_size=microbatch_size
    ):
        raise ExecutorError("checkpoint binding config differs from run config")
    if not callable(validation_provider_factory):
        raise TypeError("validation_provider_factory must be callable")
    if not isinstance(validation_start_cursor, Mapping):
        raise TypeError("validation_start_cursor must be a mapping")
    torch.set_num_threads(config.threads)

    counters = TrainingCounters()
    validate_production_optimizer(
        model, optimizer, scheduler, completed_epochs=counters.completed_epochs
    )
    if resume:
        document = load_last_checkpoint(output_directory, checkpoint_binding)
        counters = restore_checkpoint(
            document, model, optimizer, scheduler, training_provider
        )
        validate_production_optimizer(
            model, optimizer, scheduler, completed_epochs=counters.completed_epochs
        )
    elif counters.completed_epochs != 0:
        raise ExecutorError("new production run has non-zero counters")

    if counters.completed_epochs > config.epochs:
        raise ExecutorError("checkpoint completed epoch count exceeds run")
    for epoch_index in range(counters.completed_epochs, config.epochs):
        lambda_value = config.lambda_schedule.value(epoch_index)
        training = train_epoch(
            model,
            optimizer,
            training_provider,
            lambda_value=lambda_value,
            sample_budget=TRAINING_SAMPLES_PER_EPOCH,
            effective_batch_size=config.effective_batch_size,
            microbatch_size=microbatch_size,
            loss_function=loss_function,
        )

        # A fresh provider is used and force-restored to the same authenticated
        # logical cursor every epoch.  Training's provider is never touched.
        validation_provider = reset_validation_provider(
            validation_provider_factory, validation_start_cursor
        )
        validation = validate_epoch(
            model,
            validation_provider,
            lambda_value=lambda_value,
            sample_budget=VALIDATION_SAMPLES_PER_EPOCH,
            effective_batch_size=config.effective_batch_size,
            microbatch_size=microbatch_size,
            loss_function=loss_function,
        )
        scheduler.step()
        counters = TrainingCounters(
            completed_epochs=epoch_index + 1,
            global_steps=counters.global_steps + training.steps,
            training_samples=counters.training_samples + training.samples,
            validation_samples=counters.validation_samples + validation.samples,
            last_epoch_training_samples=training.samples,
            last_epoch_validation_samples=validation.samples,
            last_train_loss=training.mean_loss,
            last_validation_loss=validation.mean_loss,
            last_lambda=lambda_value,
        )
        cursor = training_provider.logical_cursor_state()
        if not isinstance(cursor, Mapping):
            raise TypeError("provider.logical_cursor_state must return a mapping")
        document = checkpoint_document(
            model, optimizer, scheduler, cursor, counters, checkpoint_binding
        )
        save_last_checkpoint(output_directory, document)
    return counters


__all__ = [
    "CONFIG_FORMAT",
    "EFFECTIVE_BATCH_SIZE",
    "EPOCHS",
    "EPOCH_SIZE",
    "EpochMetrics",
    "ExecutorError",
    "FINAL_LEARNING_RATE",
    "LambdaSchedule",
    "MAIN_LEARNING_RATE",
    "MICROBATCH_SIZE",
    "PRECISION",
    "ProductionRunConfig",
    "ProviderBatch",
    "RANDOM_SKIP",
    "RUN_CONFIGS",
    "ResumableBatchProvider",
    "SCHEDULER_GAMMA",
    "SEED",
    "THREADS",
    "TRAINING_SAMPLES_PER_EPOCH",
    "TRAINING_STEPS_PER_EPOCH",
    "VALIDATION_SIZE",
    "VALIDATION_BATCHES_PER_EPOCH",
    "VALIDATION_SAMPLES_PER_EPOCH",
    "WORKERS",
    "coalesce_sparse_gradients",
    "create_production_optimizer",
    "densify_sparse_gradients",
    "production_config",
    "require_production_model",
    "reset_validation_provider",
    "run_production",
    "seed_training",
    "train_epoch",
    "validate_epoch",
    "validate_production_optimizer",
]
