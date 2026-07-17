"""Dependency-light throughput gate for Atomic V3 training.

The gate deliberately has no torch, CUDA, NumPy, or provider dependency.  A
caller measures one or more production-shaped steps and supplies their wall
clock and optional stage timings.  This module aggregates those observations,
projects a full epoch, and latches a circuit breaker when the projection is
slower than the configured hard limit.

The default policy encodes the repair acceptance thresholds: five minutes is
the *total* epoch target and a conservative total projection strictly above
ten minutes is rejected.  Training-only throughput is always named as such;
validation, checkpointing, and periodic dataset authentication are explicit
parts of the total projection.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Deque, Iterable, Mapping, Optional


DEFAULT_TARGET_EPOCH_SECONDS = 5.0 * 60.0
DEFAULT_HARD_MAX_EPOCH_SECONDS = 10.0 * 60.0
DEFAULT_NON_TRAINING_RESERVE_SECONDS = 60.0


def _positive_finite(label: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _nonnegative_finite(label: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


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


def _stage_seconds(values: Mapping[str, float]) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError("stage_seconds must be a mapping")
    normalized: dict[str, float] = {}
    for name, seconds in values.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("stage timing names must be non-empty strings")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise TypeError(f"stage timing {name!r} must be a number")
        duration = float(seconds)
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError(
                f"stage timing {name!r} must be finite and non-negative"
            )
        normalized[name] = duration
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True)
class StepMeasurement:
    """One production-shaped observation supplied by the caller.

    Stage timers may overlap, so their sum is intentionally not required to be
    less than ``elapsed_seconds``.
    """

    samples: int
    elapsed_seconds: float
    stage_seconds: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", _positive_int("samples", self.samples))
        object.__setattr__(
            self,
            "elapsed_seconds",
            _positive_finite("elapsed_seconds", self.elapsed_seconds),
        )
        object.__setattr__(self, "stage_seconds", _stage_seconds(self.stage_seconds))


@dataclass(frozen=True)
class PerformanceMetrics:
    """Aggregated throughput and an honest conservative epoch projection."""

    steps: int
    samples: int
    elapsed_seconds: float
    projection_elapsed_seconds: float
    elapsed_exclusion_seconds: float
    steps_per_second: float
    samples_per_second: float
    estimated_training_seconds: float
    cold_authentication_seconds: float
    expected_authentication_events_per_epoch: float
    amortized_authentication_seconds: float
    non_training_reserve_seconds: float
    conservative_total_epoch_seconds: float
    stage_seconds: Mapping[str, float]
    stage_seconds_per_step: Mapping[str, float]

    def to_document(self) -> dict[str, object]:
        """Return a JSON-serializable metrics document."""

        return {
            "steps": self.steps,
            "samples": self.samples,
            "elapsed_seconds": self.elapsed_seconds,
            "projection_elapsed_seconds": self.projection_elapsed_seconds,
            "elapsed_exclusion_seconds": self.elapsed_exclusion_seconds,
            "steps_per_second": self.steps_per_second,
            "samples_per_second": self.samples_per_second,
            "estimated_training_seconds": self.estimated_training_seconds,
            "cold_authentication_seconds": self.cold_authentication_seconds,
            "expected_authentication_events_per_epoch": (
                self.expected_authentication_events_per_epoch
            ),
            "amortized_authentication_seconds": (
                self.amortized_authentication_seconds
            ),
            "non_training_reserve_seconds": self.non_training_reserve_seconds,
            "conservative_total_epoch_seconds": (
                self.conservative_total_epoch_seconds
            ),
            "stage_seconds": dict(self.stage_seconds),
            "stage_seconds_per_step": dict(self.stage_seconds_per_step),
        }

    @property
    def estimated_epoch_seconds(self) -> float:
        """Compatibility alias; machine reports use the honest field names."""

        return self.conservative_total_epoch_seconds


def aggregate_measurements(
    measurements: Iterable[StepMeasurement],
    *,
    epoch_samples: int,
    elapsed_exclusion_seconds: float = 0.0,
    cold_authentication_seconds: float = 0.0,
    expected_authentication_events_per_epoch: float = 0.0,
    non_training_reserve_seconds: float = 0.0,
) -> PerformanceMetrics:
    """Aggregate observations and project training plus known epoch overhead.

    ``elapsed_exclusion_seconds`` is used by the cumulative campaign breaker
    to remove the one-off initial shard authentication already measured by the
    mandatory preflight.  Later shard transitions remain in the cumulative
    wall rate and are therefore amortized naturally.
    """

    epoch_samples = _positive_int("epoch_samples", epoch_samples)
    observations = tuple(measurements)
    if not observations:
        raise ValueError("at least one measurement is required")
    if not all(isinstance(item, StepMeasurement) for item in observations):
        raise TypeError("measurements must contain StepMeasurement instances")

    steps = len(observations)
    samples = sum(item.samples for item in observations)
    elapsed = sum(item.elapsed_seconds for item in observations)
    exclusion = _nonnegative_finite(
        "elapsed_exclusion_seconds", elapsed_exclusion_seconds
    )
    if exclusion >= elapsed:
        raise ValueError("elapsed_exclusion_seconds must be below measured elapsed")
    projection_elapsed = elapsed - exclusion
    cold_authentication = _nonnegative_finite(
        "cold_authentication_seconds", cold_authentication_seconds
    )
    expected_authentications = _nonnegative_finite(
        "expected_authentication_events_per_epoch",
        expected_authentication_events_per_epoch,
    )
    reserve = _nonnegative_finite(
        "non_training_reserve_seconds", non_training_reserve_seconds
    )
    stages: dict[str, float] = {}
    for item in observations:
        for name, seconds in item.stage_seconds.items():
            stages[name] = stages.get(name, 0.0) + seconds

    samples_per_second = samples / projection_elapsed
    estimated_training = epoch_samples / samples_per_second
    amortized_authentication = cold_authentication * expected_authentications
    conservative_total = estimated_training + amortized_authentication + reserve
    stage_totals = MappingProxyType(dict(sorted(stages.items())))
    stage_per_step = MappingProxyType(
        {name: seconds / steps for name, seconds in stage_totals.items()}
    )
    return PerformanceMetrics(
        steps=steps,
        samples=samples,
        elapsed_seconds=elapsed,
        projection_elapsed_seconds=projection_elapsed,
        elapsed_exclusion_seconds=exclusion,
        steps_per_second=steps / projection_elapsed,
        samples_per_second=samples_per_second,
        estimated_training_seconds=estimated_training,
        cold_authentication_seconds=cold_authentication,
        expected_authentication_events_per_epoch=expected_authentications,
        amortized_authentication_seconds=amortized_authentication,
        non_training_reserve_seconds=reserve,
        conservative_total_epoch_seconds=conservative_total,
        stage_seconds=stage_totals,
        stage_seconds_per_step=stage_per_step,
    )


@dataclass(frozen=True)
class PerformanceGatePolicy:
    """Threshold and sampling policy for the rolling circuit breaker."""

    epoch_samples: int
    warmup_steps: int = 10
    minimum_measured_steps: int = 10
    rolling_window_steps: int = 20
    target_total_epoch_seconds: float = DEFAULT_TARGET_EPOCH_SECONDS
    hard_max_total_epoch_seconds: float = DEFAULT_HARD_MAX_EPOCH_SECONDS
    non_training_reserve_seconds: float = DEFAULT_NON_TRAINING_RESERVE_SECONDS
    cold_authentication_seconds: float = 0.0
    expected_authentication_events_per_epoch: float = 0.0
    elapsed_exclusion_seconds: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "epoch_samples", _positive_int("epoch_samples", self.epoch_samples)
        )
        object.__setattr__(
            self, "warmup_steps", _nonnegative_int("warmup_steps", self.warmup_steps)
        )
        object.__setattr__(
            self,
            "minimum_measured_steps",
            _positive_int("minimum_measured_steps", self.minimum_measured_steps),
        )
        object.__setattr__(
            self,
            "rolling_window_steps",
            _positive_int("rolling_window_steps", self.rolling_window_steps),
        )
        if self.rolling_window_steps < self.minimum_measured_steps:
            raise ValueError(
                "rolling_window_steps must be at least minimum_measured_steps"
            )
        target = _positive_finite(
            "target_total_epoch_seconds", self.target_total_epoch_seconds
        )
        hard_max = _positive_finite(
            "hard_max_total_epoch_seconds", self.hard_max_total_epoch_seconds
        )
        if hard_max < target:
            raise ValueError(
                "hard_max_total_epoch_seconds must be at least "
                "target_total_epoch_seconds"
            )
        reserve = _nonnegative_finite(
            "non_training_reserve_seconds", self.non_training_reserve_seconds
        )
        cold = _nonnegative_finite(
            "cold_authentication_seconds", self.cold_authentication_seconds
        )
        expected = _nonnegative_finite(
            "expected_authentication_events_per_epoch",
            self.expected_authentication_events_per_epoch,
        )
        exclusion = _nonnegative_finite(
            "elapsed_exclusion_seconds", self.elapsed_exclusion_seconds
        )
        object.__setattr__(self, "target_total_epoch_seconds", target)
        object.__setattr__(self, "hard_max_total_epoch_seconds", hard_max)
        object.__setattr__(self, "non_training_reserve_seconds", reserve)
        object.__setattr__(self, "cold_authentication_seconds", cold)
        object.__setattr__(self, "expected_authentication_events_per_epoch", expected)
        object.__setattr__(self, "elapsed_exclusion_seconds", exclusion)

    @property
    def target_epoch_seconds(self) -> float:
        return self.target_total_epoch_seconds

    @property
    def hard_max_epoch_seconds(self) -> float:
        return self.hard_max_total_epoch_seconds

    @property
    def amortized_authentication_seconds(self) -> float:
        return (
            self.cold_authentication_seconds
            * self.expected_authentication_events_per_epoch
        )

    @property
    def target_training_budget_seconds(self) -> float:
        return max(
            0.0,
            self.target_total_epoch_seconds
            - self.non_training_reserve_seconds
            - self.amortized_authentication_seconds,
        )

    @property
    def hard_max_training_budget_seconds(self) -> float:
        return max(
            0.0,
            self.hard_max_total_epoch_seconds
            - self.non_training_reserve_seconds
            - self.amortized_authentication_seconds,
        )

    def projection_kwargs(self) -> dict[str, float]:
        return {
            "elapsed_exclusion_seconds": self.elapsed_exclusion_seconds,
            "cold_authentication_seconds": self.cold_authentication_seconds,
            "expected_authentication_events_per_epoch": (
                self.expected_authentication_events_per_epoch
            ),
            "non_training_reserve_seconds": self.non_training_reserve_seconds,
        }


class GateStatus(str, Enum):
    WARMING_UP = "warming-up"
    MEASURING = "measuring"
    TARGET = "target"
    ACCEPTABLE = "acceptable"
    REJECTED = "rejected"


@dataclass(frozen=True)
class GateDecision:
    """Current circuit-breaker decision."""

    status: GateStatus
    observed_steps: int
    measured_steps: int
    steps_until_evaluation: int
    metrics: Optional[PerformanceMetrics]
    reason: str
    samples_until_evaluation: int = 0

    @property
    def rejected(self) -> bool:
        return self.status is GateStatus.REJECTED

    @property
    def meets_target(self) -> bool:
        return self.status is GateStatus.TARGET

    def to_document(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "observed_steps": self.observed_steps,
            "measured_steps": self.measured_steps,
            "steps_until_evaluation": self.steps_until_evaluation,
            "samples_until_evaluation": self.samples_until_evaluation,
            "reason": self.reason,
            "metrics": None if self.metrics is None else self.metrics.to_document(),
        }


class PerformanceGateRejected(RuntimeError):
    """Raised when the rolling estimate exceeds the hard epoch limit."""

    def __init__(self, decision: GateDecision) -> None:
        if not decision.rejected or decision.metrics is None:
            raise ValueError("PerformanceGateRejected requires a rejected decision")
        self.decision = decision
        estimate = decision.metrics.conservative_total_epoch_seconds
        super().__init__(
            f"Atomic V3 performance gate rejected an estimated "
            f"{estimate:.3f}-second conservative total epoch"
        )


def classify_metrics(
    metrics: PerformanceMetrics, policy: PerformanceGatePolicy
) -> GateStatus:
    """Classify a mature metrics window using the policy boundaries."""

    if not isinstance(metrics, PerformanceMetrics):
        raise TypeError("metrics must be PerformanceMetrics")
    if not isinstance(policy, PerformanceGatePolicy):
        raise TypeError("policy must be PerformanceGatePolicy")
    if (
        metrics.conservative_total_epoch_seconds
        > policy.hard_max_total_epoch_seconds
    ):
        return GateStatus.REJECTED
    if (
        metrics.conservative_total_epoch_seconds
        <= policy.target_total_epoch_seconds
    ):
        return GateStatus.TARGET
    return GateStatus.ACCEPTABLE


class RollingPerformanceGate:
    """Rolling, latched performance circuit breaker.

    Warm-up observations are discarded.  After
    ``minimum_measured_steps``, every new observation is classified using at
    most ``rolling_window_steps`` recent measured steps.  Rejection is latched:
    further calls return the same decision and consume no measurements until a
    new gate instance is constructed.
    """

    def __init__(self, policy: PerformanceGatePolicy) -> None:
        if not isinstance(policy, PerformanceGatePolicy):
            raise TypeError("policy must be PerformanceGatePolicy")
        self.policy = policy
        self._observed_steps = 0
        self._measured_steps = 0
        self._window: Deque[StepMeasurement] = deque(
            maxlen=policy.rolling_window_steps
        )
        self._decision: Optional[GateDecision] = None

    @property
    def decision(self) -> Optional[GateDecision]:
        return self._decision

    @property
    def tripped(self) -> bool:
        return self._decision is not None and self._decision.rejected

    def observe(self, measurement: StepMeasurement) -> GateDecision:
        if not isinstance(measurement, StepMeasurement):
            raise TypeError("measurement must be a StepMeasurement")
        if self.tripped:
            assert self._decision is not None
            return self._decision

        self._observed_steps += 1
        if self._observed_steps <= self.policy.warmup_steps:
            remaining_warmup = self.policy.warmup_steps - self._observed_steps
            decision = GateDecision(
                status=GateStatus.WARMING_UP,
                observed_steps=self._observed_steps,
                measured_steps=0,
                steps_until_evaluation=(
                    remaining_warmup + self.policy.minimum_measured_steps
                ),
                metrics=None,
                reason="warm-up observation excluded from performance metrics",
            )
            self._decision = decision
            return decision

        self._window.append(measurement)
        self._measured_steps += 1
        metrics = aggregate_measurements(
            self._window,
            epoch_samples=self.policy.epoch_samples,
            **self.policy.projection_kwargs(),
        )
        remaining = max(
            0, self.policy.minimum_measured_steps - self._measured_steps
        )
        if remaining:
            decision = GateDecision(
                status=GateStatus.MEASURING,
                observed_steps=self._observed_steps,
                measured_steps=self._measured_steps,
                steps_until_evaluation=remaining,
                metrics=metrics,
                reason="collecting the minimum measured-step window",
            )
            self._decision = decision
            return decision

        status = classify_metrics(metrics, self.policy)
        if status is GateStatus.REJECTED:
            reason = (
                "estimated epoch exceeds the hard "
                f"{self.policy.hard_max_total_epoch_seconds:.3f}-second limit"
            )
        elif status is GateStatus.TARGET:
            reason = (
                "estimated epoch meets the "
                f"{self.policy.target_total_epoch_seconds:.3f}-second target"
            )
        else:
            reason = (
                "estimated epoch is above target but within the hard limit"
            )
        decision = GateDecision(
            status=status,
            observed_steps=self._observed_steps,
            measured_steps=self._measured_steps,
            steps_until_evaluation=0,
            metrics=metrics,
            reason=reason,
        )
        self._decision = decision
        return decision

    def observe_or_raise(self, measurement: StepMeasurement) -> GateDecision:
        """Observe one step and raise immediately when the breaker is open."""

        decision = self.observe(measurement)
        if decision.rejected:
            raise PerformanceGateRejected(decision)
        return decision


class CumulativePerformanceGate:
    """Latched campaign breaker based on cumulative training-only wall time.

    The gate intentionally has no short rolling window.  It waits until at
    least one expected accepted-sample shard span has been observed, then
    projects from every training heartbeat accumulated so far.  Validation and
    checkpoint pauses are absent because the caller supplies only committed
    provider-plus-GPU training chunks.
    """

    def __init__(
        self, policy: PerformanceGatePolicy, *, minimum_measured_samples: int
    ) -> None:
        if not isinstance(policy, PerformanceGatePolicy):
            raise TypeError("policy must be PerformanceGatePolicy")
        self.policy = policy
        self.minimum_measured_samples = _positive_int(
            "minimum_measured_samples", minimum_measured_samples
        )
        self._measurements: list[StepMeasurement] = []
        self._samples = 0
        self._elapsed = 0.0
        self._decision: Optional[GateDecision] = None

    @property
    def decision(self) -> Optional[GateDecision]:
        return self._decision

    @property
    def tripped(self) -> bool:
        return self._decision is not None and self._decision.rejected

    def observe(self, measurement: StepMeasurement) -> GateDecision:
        if not isinstance(measurement, StepMeasurement):
            raise TypeError("measurement must be a StepMeasurement")
        if self.tripped:
            assert self._decision is not None
            return self._decision
        self._measurements.append(measurement)
        self._samples += measurement.samples
        self._elapsed += measurement.elapsed_seconds
        remaining = max(0, self.minimum_measured_samples - self._samples)
        # The exclusion is the separately measured first cold authentication.
        # Do not manufacture a rate until real cumulative wall exceeds it.
        if remaining or self._elapsed <= self.policy.elapsed_exclusion_seconds:
            decision = GateDecision(
                status=GateStatus.MEASURING,
                observed_steps=len(self._measurements),
                measured_steps=len(self._measurements),
                steps_until_evaluation=0,
                samples_until_evaluation=remaining,
                metrics=None,
                reason=(
                    "collecting at least one accepted-sample shard span for "
                    "a cumulative campaign projection"
                ),
            )
            self._decision = decision
            return decision
        metrics = aggregate_measurements(
            self._measurements,
            epoch_samples=self.policy.epoch_samples,
            **self.policy.projection_kwargs(),
        )
        status = classify_metrics(metrics, self.policy)
        if status is GateStatus.REJECTED:
            reason = (
                "cumulative conservative total epoch exceeds the hard "
                f"{self.policy.hard_max_total_epoch_seconds:.3f}-second limit"
            )
        elif status is GateStatus.TARGET:
            reason = "cumulative conservative total epoch meets target"
        else:
            reason = "cumulative total epoch is above target but within hard limit"
        decision = GateDecision(
            status=status,
            observed_steps=len(self._measurements),
            measured_steps=len(self._measurements),
            steps_until_evaluation=0,
            samples_until_evaluation=0,
            metrics=metrics,
            reason=reason,
        )
        self._decision = decision
        return decision

    def observe_or_raise(self, measurement: StepMeasurement) -> GateDecision:
        decision = self.observe(measurement)
        if decision.rejected:
            raise PerformanceGateRejected(decision)
        return decision


__all__ = [
    "CumulativePerformanceGate",
    "DEFAULT_HARD_MAX_EPOCH_SECONDS",
    "DEFAULT_NON_TRAINING_RESERVE_SECONDS",
    "DEFAULT_TARGET_EPOCH_SECONDS",
    "GateDecision",
    "GateStatus",
    "PerformanceGatePolicy",
    "PerformanceGateRejected",
    "PerformanceMetrics",
    "RollingPerformanceGate",
    "StepMeasurement",
    "aggregate_measurements",
    "classify_metrics",
]
