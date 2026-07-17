import math

import pytest

from atomic_v3.performance_gate import (
    CumulativePerformanceGate,
    DEFAULT_HARD_MAX_EPOCH_SECONDS,
    DEFAULT_TARGET_EPOCH_SECONDS,
    GateStatus,
    PerformanceGatePolicy,
    PerformanceGateRejected,
    RollingPerformanceGate,
    StepMeasurement,
    aggregate_measurements,
    classify_metrics,
)


def _measurement(*, samples=100, seconds=1.0, **stages):
    return StepMeasurement(samples, seconds, stages)


def _metrics_for_estimate(estimated_epoch_seconds):
    return aggregate_measurements(
        [_measurement(samples=1_000, seconds=estimated_epoch_seconds)],
        epoch_samples=1_000,
    )


def test_measurements_aggregate_throughput_epoch_projection_and_stages():
    provider = {"provider": 0.10, "h2d": 0.05}
    first = StepMeasurement(1_000, 0.5, provider)
    provider["provider"] = 99.0
    second = _measurement(samples=1_000, seconds=0.5, provider=0.20, forward=0.30)

    metrics = aggregate_measurements([first, second], epoch_samples=20_000)

    assert metrics.steps == 2
    assert metrics.samples == 2_000
    assert metrics.elapsed_seconds == pytest.approx(1.0)
    assert metrics.steps_per_second == pytest.approx(2.0)
    assert metrics.samples_per_second == pytest.approx(2_000.0)
    assert metrics.estimated_training_seconds == pytest.approx(10.0)
    assert metrics.conservative_total_epoch_seconds == pytest.approx(10.0)
    assert dict(metrics.stage_seconds) == {
        "forward": pytest.approx(0.30),
        "h2d": pytest.approx(0.05),
        "provider": pytest.approx(0.30),
    }
    assert metrics.stage_seconds_per_step["provider"] == pytest.approx(0.15)
    assert metrics.to_document()["stage_seconds"]["forward"] == pytest.approx(0.30)


@pytest.mark.parametrize(
    ("estimate", "expected"),
    [
        (DEFAULT_TARGET_EPOCH_SECONDS, GateStatus.TARGET),
        (DEFAULT_TARGET_EPOCH_SECONDS + 0.001, GateStatus.ACCEPTABLE),
        (DEFAULT_HARD_MAX_EPOCH_SECONDS, GateStatus.ACCEPTABLE),
        (DEFAULT_HARD_MAX_EPOCH_SECONDS + 0.001, GateStatus.REJECTED),
    ],
)
def test_default_threshold_boundaries_are_exact(estimate, expected):
    policy = PerformanceGatePolicy(epoch_samples=1_000)
    assert classify_metrics(_metrics_for_estimate(estimate), policy) is expected


def test_gate_waits_for_configured_warmup_and_measurement_steps_before_rejecting():
    policy = PerformanceGatePolicy(
        epoch_samples=601,
        warmup_steps=2,
        minimum_measured_steps=3,
        rolling_window_steps=3,
    )
    gate = RollingPerformanceGate(policy)
    slow = _measurement(samples=1, seconds=1.0)

    first = gate.observe(slow)
    second = gate.observe(slow)
    third = gate.observe(slow)
    fourth = gate.observe(slow)
    fifth = gate.observe(slow)

    assert first.status is second.status is GateStatus.WARMING_UP
    assert first.steps_until_evaluation == 4
    assert second.steps_until_evaluation == 3
    assert third.status is fourth.status is GateStatus.MEASURING
    assert third.steps_until_evaluation == 2
    assert fourth.steps_until_evaluation == 1
    assert fifth.status is GateStatus.REJECTED
    assert fifth.metrics.estimated_training_seconds == pytest.approx(601.0)
    assert fifth.metrics.conservative_total_epoch_seconds == pytest.approx(661.0)
    assert gate.tripped


def test_rejection_is_latched_and_consumes_no_later_measurements():
    policy = PerformanceGatePolicy(
        epoch_samples=601,
        warmup_steps=0,
        minimum_measured_steps=1,
        rolling_window_steps=1,
    )
    gate = RollingPerformanceGate(policy)
    rejected = gate.observe(_measurement(samples=1, seconds=1.0))

    later = gate.observe(_measurement(samples=1_000, seconds=0.001))

    assert later is rejected
    assert later.observed_steps == 1
    assert later.measured_steps == 1
    assert later.status is GateStatus.REJECTED


def test_rolling_window_reclassifies_and_trips_only_on_recent_steps():
    policy = PerformanceGatePolicy(
        epoch_samples=1_000,
        warmup_steps=0,
        minimum_measured_steps=2,
        rolling_window_steps=2,
    )
    gate = RollingPerformanceGate(policy)
    fast = _measurement(samples=100, seconds=1.0)
    slow = _measurement(samples=1, seconds=1.0)

    assert gate.observe(fast).status is GateStatus.MEASURING
    assert gate.observe(fast).status is GateStatus.TARGET
    assert gate.observe(slow).status is GateStatus.TARGET
    rejected = gate.observe(slow)

    assert rejected.status is GateStatus.REJECTED
    assert rejected.metrics.steps == 2
    assert rejected.metrics.samples == 2
    assert rejected.metrics.estimated_training_seconds == pytest.approx(1_000.0)
    assert rejected.metrics.conservative_total_epoch_seconds == pytest.approx(1_060.0)


def test_observe_or_raise_exposes_the_latched_decision():
    gate = RollingPerformanceGate(
        PerformanceGatePolicy(
            epoch_samples=601,
            warmup_steps=0,
            minimum_measured_steps=1,
            rolling_window_steps=1,
        )
    )

    with pytest.raises(
        PerformanceGateRejected,
        match="661.000-second conservative total epoch",
    ) as raised:
        gate.observe_or_raise(_measurement(samples=1, seconds=1.0))

    assert raised.value.decision is gate.decision
    assert gate.tripped


@pytest.mark.parametrize(
    "arguments",
    [
        {"samples": 0, "elapsed_seconds": 1.0},
        {"samples": 1, "elapsed_seconds": 0.0},
        {"samples": 1, "elapsed_seconds": math.inf},
        {"samples": 1, "elapsed_seconds": 1.0, "stage_seconds": {"x": -1.0}},
        {"samples": 1, "elapsed_seconds": 1.0, "stage_seconds": {"": 1.0}},
    ],
)
def test_measurement_rejects_invalid_values(arguments):
    with pytest.raises((TypeError, ValueError)):
        StepMeasurement(**arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        {"epoch_samples": 0},
        {"epoch_samples": 1, "warmup_steps": -1},
        {"epoch_samples": 1, "minimum_measured_steps": 0},
        {
            "epoch_samples": 1,
            "minimum_measured_steps": 3,
            "rolling_window_steps": 2,
        },
        {
            "epoch_samples": 1,
            "target_total_epoch_seconds": 601.0,
            "hard_max_total_epoch_seconds": 600.0,
        },
    ],
)
def test_policy_rejects_invalid_values(arguments):
    with pytest.raises((TypeError, ValueError)):
        PerformanceGatePolicy(**arguments)


def test_aggregate_rejects_empty_or_non_measurement_inputs():
    with pytest.raises(ValueError, match="at least one"):
        aggregate_measurements([], epoch_samples=1)
    with pytest.raises(TypeError, match="StepMeasurement"):
        aggregate_measurements([object()], epoch_samples=1)


def test_projection_names_training_auth_reserve_and_conservative_total_honestly():
    metrics = aggregate_measurements(
        [_measurement(samples=1_000, seconds=150.0)],
        epoch_samples=1_000,
        cold_authentication_seconds=10.0,
        expected_authentication_events_per_epoch=6.0,
        non_training_reserve_seconds=60.0,
    )
    policy = PerformanceGatePolicy(
        epoch_samples=1_000,
        target_total_epoch_seconds=300.0,
        hard_max_total_epoch_seconds=600.0,
        cold_authentication_seconds=10.0,
        expected_authentication_events_per_epoch=6.0,
        non_training_reserve_seconds=60.0,
    )

    assert metrics.estimated_training_seconds == pytest.approx(150.0)
    assert metrics.amortized_authentication_seconds == pytest.approx(60.0)
    assert metrics.non_training_reserve_seconds == pytest.approx(60.0)
    assert metrics.conservative_total_epoch_seconds == pytest.approx(270.0)
    assert policy.target_training_budget_seconds == pytest.approx(180.0)
    assert policy.hard_max_training_budget_seconds == pytest.approx(480.0)
    document = metrics.to_document()
    assert "estimated_epoch_seconds" not in document
    assert document["estimated_training_seconds"] == pytest.approx(150.0)


def test_classification_uses_conservative_total_not_training_only():
    metrics = aggregate_measurements(
        [_measurement(samples=1_000, seconds=550.0)],
        epoch_samples=1_000,
        non_training_reserve_seconds=60.0,
    )
    policy = PerformanceGatePolicy(
        epoch_samples=1_000,
        target_total_epoch_seconds=300.0,
        hard_max_total_epoch_seconds=600.0,
    )

    assert metrics.estimated_training_seconds == pytest.approx(550.0)
    assert metrics.conservative_total_epoch_seconds == pytest.approx(610.0)
    assert classify_metrics(metrics, policy) is GateStatus.REJECTED


def test_cumulative_gate_ignores_short_windows_and_latches_sustained_failure():
    policy = PerformanceGatePolicy(
        epoch_samples=1_000,
        warmup_steps=0,
        minimum_measured_steps=1,
        rolling_window_steps=1,
        target_total_epoch_seconds=20.0,
        hard_max_total_epoch_seconds=50.0,
        non_training_reserve_seconds=0.0,
        elapsed_exclusion_seconds=10.0,
    )
    gate = CumulativePerformanceGate(policy, minimum_measured_samples=200)

    assert gate.observe(_measurement(samples=100, seconds=11.0)).status is GateStatus.MEASURING
    assert gate.observe(_measurement(samples=100, seconds=1.0)).status is GateStatus.TARGET
    # A single later shard transition is amortized with all prior chunks; this
    # is acceptable rather than an immediate short-window rejection.
    transition = gate.observe(_measurement(samples=100, seconds=5.0))
    assert transition.status is GateStatus.ACCEPTABLE
    rejected = gate.observe(_measurement(samples=100, seconds=50.0))
    assert rejected.status is GateStatus.REJECTED
    assert gate.observe(_measurement(samples=1000, seconds=0.001)) is rejected
