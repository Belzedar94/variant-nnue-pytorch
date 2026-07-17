"""Explicit, non-production Atomic V3 performance benchmark harness.

Nothing in this module executes automatically on import.  It is intentionally
separate from the trainer and accepts callbacks instead of importing torch or
CUDA.  A real GPU caller must explicitly provide synchronization and VRAM
probes; unit tests can provide deterministic fakes.

The harness runs configurable warm-up steps followed by 10--20 measured,
production-shaped optimizer steps.  It reports full wall throughput and stage
timings through :mod:`atomic_v3.performance_gate` and must not be used as a
training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Optional

from .performance_gate import (
    DEFAULT_HARD_MAX_EPOCH_SECONDS,
    DEFAULT_NON_TRAINING_RESERVE_SECONDS,
    DEFAULT_TARGET_EPOCH_SECONDS,
    GateDecision,
    GateStatus,
    PerformanceGatePolicy,
    PerformanceGateRejected,
    StepMeasurement,
    aggregate_measurements,
    classify_metrics,
)


BENCHMARK_FORMAT = "atomic-v3-non-production-performance-benchmark-v1"
MINIMUM_MEASURED_STEPS = 10
MAXIMUM_MEASURED_STEPS = 20

STAGE_PROVIDER_FETCH_H2D = "provider_fetch_h2d"
STAGE_FORWARD_LOSS = "forward_loss"
STAGE_BACKWARD = "backward"
STAGE_OPTIMIZER_CLIPPING = "optimizer_clipping"


def _nonnegative_int(label: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _positive_int(label: str, value: object) -> int:
    result = _nonnegative_int(label, value)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_bytes(label: str, value: object) -> int:
    return _nonnegative_int(label, value)


def _callable(label: str, value: object) -> Callable[..., Any]:
    if not callable(value):
        raise TypeError(f"{label} must be callable")
    return value


@dataclass(frozen=True)
class BenchmarkBatch:
    """Opaque payload and accepted sample count returned by the provider hook."""

    payload: Any
    samples: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", _positive_int("samples", self.samples))


@dataclass(frozen=True)
class VramSnapshot:
    """One explicit CUDA memory probe result, expressed in bytes."""

    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    def __post_init__(self) -> None:
        for label in (
            "allocated_bytes",
            "reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        ):
            object.__setattr__(
                self, label, _nonnegative_bytes(label, getattr(self, label))
            )
        if self.peak_allocated_bytes < self.allocated_bytes:
            raise ValueError("peak_allocated_bytes is below allocated_bytes")
        if self.peak_reserved_bytes < self.reserved_bytes:
            raise ValueError("peak_reserved_bytes is below reserved_bytes")

    def to_document(self) -> dict[str, int]:
        return {
            "allocated_bytes": self.allocated_bytes,
            "reserved_bytes": self.reserved_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
        }


@dataclass(frozen=True)
class VramSummary:
    """Final and maximum memory observations across the whole benchmark."""

    final_allocated_bytes: int
    final_reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    def to_document(self) -> dict[str, int]:
        return {
            "final_allocated_bytes": self.final_allocated_bytes,
            "final_reserved_bytes": self.final_reserved_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
        }


@dataclass(frozen=True)
class BenchmarkOperations:
    """Explicit hooks for one production-shaped optimizer step.

    ``prepare_optimizer_step`` is intended for zero-grad and any pre-step
    clipping.  Its time is combined with ``optimizer_and_clipping`` in the
    optimizer/clipping stage.  ``synchronize`` must wait for all queued device
    work, otherwise asynchronous GPU timings are invalid.
    """

    prepare_optimizer_step: Callable[[], None]
    fetch_and_h2d: Callable[[], BenchmarkBatch]
    forward_loss: Callable[[Any], Any]
    backward: Callable[[Any], None]
    optimizer_and_clipping: Callable[[], None]
    synchronize: Callable[[], None]
    vram_probe: Callable[[], VramSnapshot]

    def __post_init__(self) -> None:
        for label in (
            "prepare_optimizer_step",
            "fetch_and_h2d",
            "forward_loss",
            "backward",
            "optimizer_and_clipping",
            "synchronize",
            "vram_probe",
        ):
            _callable(label, getattr(self, label))


@dataclass(frozen=True)
class PerformanceBenchmarkConfig:
    """Configuration for an explicit non-production benchmark invocation."""

    epoch_samples: int
    warmup_steps: int = 5
    measured_steps: int = MINIMUM_MEASURED_STEPS
    target_epoch_seconds: float = DEFAULT_TARGET_EPOCH_SECONDS
    hard_max_epoch_seconds: float = DEFAULT_HARD_MAX_EPOCH_SECONDS
    non_training_reserve_seconds: float = DEFAULT_NON_TRAINING_RESERVE_SECONDS
    expected_authentication_events_per_epoch: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "epoch_samples", _positive_int("epoch_samples", self.epoch_samples)
        )
        object.__setattr__(
            self, "warmup_steps", _nonnegative_int("warmup_steps", self.warmup_steps)
        )
        measured = _nonnegative_int("measured_steps", self.measured_steps)
        if not MINIMUM_MEASURED_STEPS <= measured <= MAXIMUM_MEASURED_STEPS:
            raise ValueError(
                f"measured_steps must be between {MINIMUM_MEASURED_STEPS} "
                f"and {MAXIMUM_MEASURED_STEPS}"
            )
        object.__setattr__(self, "measured_steps", measured)

        # Delegate finite-number and threshold-order validation to the shared
        # gate policy so the harness cannot disagree with production gating.
        policy = self.to_gate_policy(cold_authentication_seconds=0.0)
        object.__setattr__(
            self, "target_epoch_seconds", policy.target_epoch_seconds
        )
        object.__setattr__(
            self, "hard_max_epoch_seconds", policy.hard_max_epoch_seconds
        )
        object.__setattr__(
            self,
            "non_training_reserve_seconds",
            policy.non_training_reserve_seconds,
        )
        object.__setattr__(
            self,
            "expected_authentication_events_per_epoch",
            policy.expected_authentication_events_per_epoch,
        )

    def to_gate_policy(
        self, *, cold_authentication_seconds: float
    ) -> PerformanceGatePolicy:
        return PerformanceGatePolicy(
            epoch_samples=self.epoch_samples,
            warmup_steps=0,
            minimum_measured_steps=self.measured_steps,
            rolling_window_steps=self.measured_steps,
            target_total_epoch_seconds=self.target_epoch_seconds,
            hard_max_total_epoch_seconds=self.hard_max_epoch_seconds,
            non_training_reserve_seconds=self.non_training_reserve_seconds,
            cold_authentication_seconds=cold_authentication_seconds,
            expected_authentication_events_per_epoch=(
                self.expected_authentication_events_per_epoch
            ),
        )


@dataclass(frozen=True)
class PerformanceBenchmarkReport:
    """Machine-readable benchmark outcome.  This is never training evidence."""

    config: PerformanceBenchmarkConfig
    decision: GateDecision
    vram: VramSummary
    first_cold_measurement: Optional[StepMeasurement]

    @property
    def non_production(self) -> bool:
        return True

    def require_viable(self) -> "PerformanceBenchmarkReport":
        if self.decision.rejected:
            raise PerformanceGateRejected(self.decision)
        return self

    def to_document(self) -> dict[str, object]:
        policy = self.config.to_gate_policy(
            cold_authentication_seconds=(
                0.0
                if self.decision.metrics is None
                else self.decision.metrics.cold_authentication_seconds
            )
        )
        cold = self.first_cold_measurement
        return {
            "format": BENCHMARK_FORMAT,
            "non_production": True,
            "training_started": False,
            "config": {
                "epoch_samples": self.config.epoch_samples,
                "warmup_steps": self.config.warmup_steps,
                "measured_steps": self.config.measured_steps,
                "target_epoch_seconds": self.config.target_epoch_seconds,
                "hard_max_epoch_seconds": self.config.hard_max_epoch_seconds,
                "target_is_total_epoch": True,
                "hard_max_is_total_epoch": True,
                "non_training_reserve_seconds": (
                    self.config.non_training_reserve_seconds
                ),
                "expected_authentication_events_per_epoch": (
                    self.config.expected_authentication_events_per_epoch
                ),
                "target_training_budget_seconds": (
                    policy.target_training_budget_seconds
                ),
                "hard_max_training_budget_seconds": (
                    policy.hard_max_training_budget_seconds
                ),
            },
            "cold_authentication": None
            if cold is None
            else {
                "first_step_elapsed_seconds": cold.elapsed_seconds,
                "first_step_samples": cold.samples,
                "first_step_stage_seconds": dict(cold.stage_seconds),
                "estimated_cold_authentication_seconds": (
                    self.decision.metrics.cold_authentication_seconds
                ),
                "expected_events_per_epoch": (
                    self.config.expected_authentication_events_per_epoch
                ),
                "amortized_epoch_seconds": (
                    self.decision.metrics.amortized_authentication_seconds
                ),
            },
            "gate": self.decision.to_document(),
            "vram": self.vram.to_document(),
        }


def _timed_call(
    callback: Callable[[], Any],
    *,
    synchronize: Callable[[], None],
    clock: Callable[[], float],
) -> tuple[Any, float]:
    synchronize()
    started = clock()
    result = callback()
    synchronize()
    finished = clock()
    elapsed = finished - started
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise RuntimeError("benchmark clock produced an invalid stage duration")
    return result, elapsed


def _measure_step(
    operations: BenchmarkOperations, *, clock: Callable[[], float]
) -> StepMeasurement:
    operations.synchronize()
    wall_started = clock()

    _, optimizer_prepare_seconds = _timed_call(
        operations.prepare_optimizer_step,
        synchronize=operations.synchronize,
        clock=clock,
    )
    fetched, provider_seconds = _timed_call(
        operations.fetch_and_h2d,
        synchronize=operations.synchronize,
        clock=clock,
    )
    if not isinstance(fetched, BenchmarkBatch):
        raise TypeError("fetch_and_h2d must return BenchmarkBatch")
    loss, forward_seconds = _timed_call(
        lambda: operations.forward_loss(fetched.payload),
        synchronize=operations.synchronize,
        clock=clock,
    )
    _, backward_seconds = _timed_call(
        lambda: operations.backward(loss),
        synchronize=operations.synchronize,
        clock=clock,
    )
    _, optimizer_finish_seconds = _timed_call(
        operations.optimizer_and_clipping,
        synchronize=operations.synchronize,
        clock=clock,
    )

    operations.synchronize()
    wall_finished = clock()
    wall_seconds = wall_finished - wall_started
    if not math.isfinite(wall_seconds) or wall_seconds <= 0.0:
        raise RuntimeError("benchmark clock produced an invalid wall duration")
    return StepMeasurement(
        samples=fetched.samples,
        elapsed_seconds=wall_seconds,
        stage_seconds={
            STAGE_PROVIDER_FETCH_H2D: provider_seconds,
            STAGE_FORWARD_LOSS: forward_seconds,
            STAGE_BACKWARD: backward_seconds,
            STAGE_OPTIMIZER_CLIPPING: (
                optimizer_prepare_seconds + optimizer_finish_seconds
            ),
        },
    )


def _summarize_vram(snapshots: list[VramSnapshot]) -> VramSummary:
    if not snapshots:
        raise ValueError("at least one VRAM snapshot is required")
    final = snapshots[-1]
    return VramSummary(
        final_allocated_bytes=final.allocated_bytes,
        final_reserved_bytes=final.reserved_bytes,
        peak_allocated_bytes=max(item.peak_allocated_bytes for item in snapshots),
        peak_reserved_bytes=max(item.peak_reserved_bytes for item in snapshots),
    )


def run_performance_benchmark(
    operations: BenchmarkOperations,
    config: PerformanceBenchmarkConfig,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> PerformanceBenchmarkReport:
    """Run an explicit benchmark; never call this from the training executor.

    The function returns rejected reports rather than raising so diagnostics can
    always be persisted.  Call ``report.require_viable()`` at the launch gate.
    """

    if not isinstance(operations, BenchmarkOperations):
        raise TypeError("operations must be BenchmarkOperations")
    if not isinstance(config, PerformanceBenchmarkConfig):
        raise TypeError("config must be PerformanceBenchmarkConfig")
    _callable("clock", clock)

    snapshots: list[VramSnapshot] = []
    measurements: list[StepMeasurement] = []
    total_steps = config.warmup_steps + config.measured_steps
    for _ in range(total_steps):
        measurement = _measure_step(operations, clock=clock)
        measurements.append(measurement)
        snapshot = operations.vram_probe()
        if not isinstance(snapshot, VramSnapshot):
            raise TypeError("vram_probe must return VramSnapshot")
        snapshots.append(snapshot)

    measured = measurements[config.warmup_steps :]
    if len(measured) != config.measured_steps:
        raise RuntimeError("benchmark ended without the exact measured-step window")
    # Warm-up number one pays the provider's full snapshot/hash authentication.
    # It must not contaminate steady optimizer throughput, but it also must not
    # disappear from the epoch budget.  Estimate only its excess over one
    # steady step and amortize that cost by the expected shard events/epoch.
    steady = aggregate_measurements(measured, epoch_samples=config.epoch_samples)
    first_cold = measurements[0] if config.warmup_steps else None
    cold_authentication_seconds = 0.0
    if first_cold is not None:
        measured_provider_seconds = steady.stage_seconds.get(
            STAGE_PROVIDER_FETCH_H2D, 0.0
        )
        expected_steady_seconds = (
            measured_provider_seconds
            * first_cold.samples
            / steady.samples
        )
        cold_authentication_seconds = max(
            0.0,
            first_cold.stage_seconds.get(STAGE_PROVIDER_FETCH_H2D, 0.0)
            - expected_steady_seconds,
        )
    policy = config.to_gate_policy(
        cold_authentication_seconds=cold_authentication_seconds
    )
    metrics = aggregate_measurements(
        measured,
        epoch_samples=config.epoch_samples,
        **policy.projection_kwargs(),
    )
    status = classify_metrics(metrics, policy)
    if status is GateStatus.REJECTED:
        reason = (
            "conservative total epoch exceeds the hard "
            f"{policy.hard_max_total_epoch_seconds:.3f}-second limit"
        )
    elif status is GateStatus.TARGET:
        reason = "conservative total epoch meets the total-epoch target"
    else:
        reason = "conservative total epoch is above target but within hard limit"
    decision = GateDecision(
        status=status,
        observed_steps=total_steps,
        measured_steps=config.measured_steps,
        steps_until_evaluation=0,
        samples_until_evaluation=0,
        metrics=metrics,
        reason=reason,
    )
    return PerformanceBenchmarkReport(
        config=config,
        decision=decision,
        vram=_summarize_vram(snapshots),
        first_cold_measurement=first_cold,
    )


__all__ = [
    "BENCHMARK_FORMAT",
    "MAXIMUM_MEASURED_STEPS",
    "MINIMUM_MEASURED_STEPS",
    "STAGE_BACKWARD",
    "STAGE_FORWARD_LOSS",
    "STAGE_OPTIMIZER_CLIPPING",
    "STAGE_PROVIDER_FETCH_H2D",
    "BenchmarkBatch",
    "BenchmarkOperations",
    "PerformanceBenchmarkConfig",
    "PerformanceBenchmarkReport",
    "VramSnapshot",
    "VramSummary",
    "run_performance_benchmark",
]
