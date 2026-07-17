import pytest

from atomic_v3.performance_benchmark import (
    BENCHMARK_FORMAT,
    STAGE_BACKWARD,
    STAGE_FORWARD_LOSS,
    STAGE_OPTIMIZER_CLIPPING,
    STAGE_PROVIDER_FETCH_H2D,
    BenchmarkBatch,
    BenchmarkOperations,
    PerformanceBenchmarkConfig,
    VramSnapshot,
    run_performance_benchmark,
)
from atomic_v3.performance_gate import GateStatus, PerformanceGateRejected


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeStep:
    def __init__(self, *, samples=100, durations=None, first_provider_extra=0.0):
        self.clock = FakeClock()
        self.samples = samples
        self.durations = durations or {
            "prepare": 0.1,
            "provider": 0.2,
            "forward": 0.3,
            "backward": 0.4,
            "optimizer": 0.5,
        }
        self.events = []
        self.step = 0
        self.first_provider_extra = first_provider_extra

    def prepare(self):
        self.events.append("prepare")
        self.clock.advance(self.durations["prepare"])

    def fetch(self):
        self.events.append("fetch")
        self.clock.advance(self.durations["provider"])
        if self.step == 0:
            self.clock.advance(self.first_provider_extra)
        return BenchmarkBatch({"step": self.step}, self.samples)

    def forward(self, payload):
        self.events.append("forward")
        assert payload == {"step": self.step}
        self.clock.advance(self.durations["forward"])
        return f"loss-{self.step}"

    def backward(self, loss):
        self.events.append("backward")
        assert loss == f"loss-{self.step}"
        self.clock.advance(self.durations["backward"])

    def optimizer(self):
        self.events.append("optimizer")
        self.clock.advance(self.durations["optimizer"])

    def synchronize(self):
        pass

    def vram(self):
        self.events.append("vram")
        self.step += 1
        allocated = 1_000 + self.step
        reserved = 2_000 + self.step
        return VramSnapshot(
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            peak_allocated_bytes=3_000 + self.step,
            peak_reserved_bytes=4_000 + self.step,
        )

    def operations(self):
        return BenchmarkOperations(
            prepare_optimizer_step=self.prepare,
            fetch_and_h2d=self.fetch,
            forward_loss=self.forward,
            backward=self.backward,
            optimizer_and_clipping=self.optimizer,
            synchronize=self.synchronize,
            vram_probe=self.vram,
        )


def test_harness_reports_measured_wall_stages_vram_and_epoch_eta():
    fake = FakeStep()
    config = PerformanceBenchmarkConfig(
        epoch_samples=10_000,
        warmup_steps=2,
        measured_steps=10,
    )

    report = run_performance_benchmark(
        fake.operations(), config, clock=fake.clock
    )

    metrics = report.decision.metrics
    assert report.non_production
    assert report.decision.status is GateStatus.TARGET
    assert report.decision.observed_steps == 12
    assert report.decision.measured_steps == 10
    assert metrics.steps == 10
    assert metrics.samples == 1_000
    assert metrics.elapsed_seconds == pytest.approx(15.0)
    assert metrics.samples_per_second == pytest.approx(1_000 / 15.0)
    assert metrics.estimated_training_seconds == pytest.approx(150.0)
    assert metrics.conservative_total_epoch_seconds == pytest.approx(210.0)
    assert dict(metrics.stage_seconds) == {
        STAGE_PROVIDER_FETCH_H2D: pytest.approx(2.0),
        STAGE_FORWARD_LOSS: pytest.approx(3.0),
        STAGE_BACKWARD: pytest.approx(4.0),
        STAGE_OPTIMIZER_CLIPPING: pytest.approx(6.0),
    }
    assert fake.events[:6] == [
        "prepare",
        "fetch",
        "forward",
        "backward",
        "optimizer",
        "vram",
    ]
    assert len(fake.events) == 12 * 6
    assert report.vram.final_allocated_bytes == 1_012
    assert report.vram.final_reserved_bytes == 2_012
    assert report.vram.peak_allocated_bytes == 3_012
    assert report.vram.peak_reserved_bytes == 4_012

    document = report.to_document()
    assert document["format"] == BENCHMARK_FORMAT
    assert document["non_production"] is True
    assert document["training_started"] is False
    assert document["gate"]["metrics"]["estimated_training_seconds"] == pytest.approx(
        150.0
    )


def test_harness_returns_diagnostics_before_launch_gate_raises_on_rejection():
    fake = FakeStep(
        samples=1,
        durations={
            "prepare": 0.1,
            "provider": 0.2,
            "forward": 0.2,
            "backward": 0.2,
            "optimizer": 0.3,
        },
    )
    config = PerformanceBenchmarkConfig(
        epoch_samples=601,
        warmup_steps=0,
        measured_steps=10,
    )

    report = run_performance_benchmark(
        fake.operations(), config, clock=fake.clock
    )

    assert report.decision.status is GateStatus.REJECTED
    assert report.decision.metrics.estimated_training_seconds == pytest.approx(601.0)
    assert report.decision.metrics.conservative_total_epoch_seconds == pytest.approx(661.0)
    with pytest.raises(
        PerformanceGateRejected,
        match="661.000-second conservative total epoch",
    ):
        report.require_viable()


@pytest.mark.parametrize("measured_steps", [0, 9, 21])
def test_config_requires_ten_to_twenty_measured_steps(measured_steps):
    with pytest.raises(ValueError, match="between 10 and 20"):
        PerformanceBenchmarkConfig(
            epoch_samples=1,
            measured_steps=measured_steps,
        )


def test_config_delegates_threshold_validation_to_shared_gate():
    with pytest.raises(ValueError, match="at least target"):
        PerformanceBenchmarkConfig(
            epoch_samples=1,
            target_epoch_seconds=601.0,
            hard_max_epoch_seconds=600.0,
        )


def test_batch_and_vram_values_are_validated():
    with pytest.raises(ValueError, match="samples must be positive"):
        BenchmarkBatch(None, 0)
    with pytest.raises(ValueError, match="peak_allocated"):
        VramSnapshot(
            allocated_bytes=2,
            reserved_bytes=3,
            peak_allocated_bytes=1,
            peak_reserved_bytes=3,
        )


def test_harness_rejects_malformed_callback_results():
    fake = FakeStep()
    operations = BenchmarkOperations(
        prepare_optimizer_step=fake.prepare,
        fetch_and_h2d=lambda: object(),
        forward_loss=fake.forward,
        backward=fake.backward,
        optimizer_and_clipping=fake.optimizer,
        synchronize=fake.synchronize,
        vram_probe=fake.vram,
    )
    config = PerformanceBenchmarkConfig(
        epoch_samples=1,
        warmup_steps=0,
        measured_steps=10,
    )

    with pytest.raises(TypeError, match="fetch_and_h2d"):
        run_performance_benchmark(operations, config, clock=fake.clock)


def test_operations_require_explicit_sync_and_vram_hooks():
    with pytest.raises(TypeError, match="synchronize must be callable"):
        BenchmarkOperations(
            prepare_optimizer_step=lambda: None,
            fetch_and_h2d=lambda: BenchmarkBatch(None, 1),
            forward_loss=lambda payload: None,
            backward=lambda loss: None,
            optimizer_and_clipping=lambda: None,
            synchronize=None,
            vram_probe=lambda: VramSnapshot(0, 0, 0, 0),
        )


def test_first_cold_step_is_reported_and_amortized_by_expected_shards():
    fake = FakeStep(first_provider_extra=9.0)
    expected_events = 20_004_864 * 4 / 12_500_000
    config = PerformanceBenchmarkConfig(
        epoch_samples=10_000,
        warmup_steps=5,
        measured_steps=10,
        expected_authentication_events_per_epoch=expected_events,
    )

    report = run_performance_benchmark(fake.operations(), config, clock=fake.clock)
    metrics = report.decision.metrics
    assert metrics.estimated_training_seconds == pytest.approx(150.0)
    assert metrics.cold_authentication_seconds == pytest.approx(9.0)
    assert metrics.expected_authentication_events_per_epoch == pytest.approx(
        expected_events
    )
    assert metrics.amortized_authentication_seconds == pytest.approx(
        9.0 * expected_events
    )
    assert metrics.conservative_total_epoch_seconds == pytest.approx(
        150.0 + 9.0 * expected_events + 60.0
    )
    document = report.to_document()
    assert document["cold_authentication"]["first_step_elapsed_seconds"] == pytest.approx(
        10.5
    )
    assert document["config"]["target_training_budget_seconds"] == pytest.approx(
        300.0 - 60.0 - 9.0 * expected_events
    )
    assert "estimated_epoch_seconds" not in document["gate"]["metrics"]
