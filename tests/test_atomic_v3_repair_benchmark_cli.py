import io
import json

import pytest

from atomic_v3.performance_benchmark import (
    BenchmarkBatch,
    BenchmarkOperations,
    VramSnapshot,
)
from atomic_v3.performance_gate import GateStatus
from scripts import benchmark_atomic_v3_repair as benchmark_cli


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeBackend:
    def __init__(self, *, samples=100, seconds_per_stage=0.1, fail_forward=False):
        self.clock = FakeClock()
        self.samples = samples
        self.seconds_per_stage = seconds_per_stage
        self.fail_forward = fail_forward
        self.step = 0
        self.events = []
        self.cleanup_calls = 0
        self.load_calls = 0

    def _stage(self, name):
        self.events.append(name)
        self.clock.advance(self.seconds_per_stage)

    def prepare(self):
        self._stage("prepare")

    def fetch(self):
        self._stage("fetch")
        return BenchmarkBatch({"step": self.step}, self.samples)

    def forward(self, payload):
        self._stage("forward")
        if self.fail_forward:
            raise RuntimeError("synthetic forward failure")
        return payload["step"]

    def backward(self, loss):
        self._stage("backward")
        assert loss == self.step

    def optimizer(self):
        self._stage("optimizer")

    def synchronize(self):
        pass

    def vram(self):
        self.events.append("vram")
        self.step += 1
        return VramSnapshot(
            allocated_bytes=1_000 + self.step,
            reserved_bytes=2_000 + self.step,
            peak_allocated_bytes=3_000 + self.step,
            peak_reserved_bytes=4_000 + self.step,
        )

    def cleanup(self):
        self.cleanup_calls += 1

    def loader(self, arguments):
        self.load_calls += 1
        assert arguments.bootstrap_source == ["receipt.json", "a" * 64]
        return benchmark_cli.BenchmarkSession(
            operations=BenchmarkOperations(
                prepare_optimizer_step=self.prepare,
                fetch_and_h2d=self.fetch,
                forward_loss=self.forward,
                backward=self.backward,
                optimizer_and_clipping=self.optimizer,
                synchronize=self.synchronize,
                vram_probe=self.vram,
            ),
            epoch_samples=10_000,
            identity={"backend": "fake", "run_id": arguments.run},
            cleanup=self.cleanup,
        )


def _argv(*extra):
    return [
        "--bootstrap-source",
        "receipt.json",
        "a" * 64,
        "--provider-library",
        "provider.dll",
        "--shared-initial-state",
        "shared.pt",
        *extra,
    ]


def test_real_command_adapter_runs_bounded_steps_and_always_cleans_up():
    fake = FakeBackend()
    arguments = benchmark_cli.build_parser().parse_args(
        _argv("--warmup-steps", "5", "--measured-steps", "10")
    )

    document, report = benchmark_cli.run_benchmark_command(
        arguments, backend_loader=fake.loader, clock=fake.clock
    )

    assert report.decision.status is GateStatus.TARGET
    assert report.decision.metrics.steps == 10
    assert report.decision.metrics.samples == 1_000
    assert report.decision.metrics.estimated_training_seconds == pytest.approx(50.0)
    assert report.decision.metrics.conservative_total_epoch_seconds == pytest.approx(
        110.0
    )
    assert fake.load_calls == 1
    assert fake.cleanup_calls == 1
    assert fake.step == 15
    assert document["non_production"] is True
    assert document["training_campaign_started"] is False
    assert document["epochs_started"] == 0
    assert document["checkpoints_written"] == 0
    assert document["ephemeral_optimizer_steps"] == 15
    assert document["identity"] == {"backend": "fake", "run_id": "lambda-0"}


def test_cli_prints_one_report_and_returns_zero_for_a_passing_gate():
    fake = FakeBackend()
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = benchmark_cli.main(
        _argv("--warmup-steps", "0", "--measured-steps", "10"),
        backend_loader=fake.loader,
        clock=fake.clock,
        stdout=stdout,
        stderr=stderr,
    )

    document = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert document["format"] == benchmark_cli.REAL_BENCHMARK_FORMAT
    assert document["status"] == "target"
    assert document["gate_passed"] is True
    assert document["benchmark"]["training_started"] is False
    assert fake.cleanup_calls == 1


def test_cli_returns_two_but_preserves_diagnostics_for_hard_rejection():
    fake = FakeBackend(samples=1, seconds_per_stage=0.2)
    stdout = io.StringIO()

    exit_code = benchmark_cli.main(
        _argv("--warmup-steps", "0", "--measured-steps", "10"),
        backend_loader=fake.loader,
        clock=fake.clock,
        stdout=stdout,
        stderr=io.StringIO(),
    )

    document = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert document["status"] == "rejected"
    assert document["gate_passed"] is False
    assert (
        document["benchmark"]["gate"]["metrics"][
            "conservative_total_epoch_seconds"
        ]
        == pytest.approx(10_060.0)
    )
    assert fake.cleanup_calls == 1


def test_cli_reports_failure_and_cleans_up_without_any_checkpoint_path():
    fake = FakeBackend(fail_forward=True)
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = benchmark_cli.main(
        _argv("--warmup-steps", "0", "--measured-steps", "10"),
        backend_loader=fake.loader,
        clock=fake.clock,
        stdout=stdout,
        stderr=stderr,
    )

    document = json.loads(stderr.getvalue())
    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert document["status"] == "failed"
    assert document["error_type"] == "RuntimeError"
    assert document["epochs_started"] == 0
    assert document["checkpoints_written"] == 0
    assert fake.cleanup_calls == 1


def test_import_and_parser_construction_do_not_load_a_backend():
    fake = FakeBackend()
    parser = benchmark_cli.build_parser()

    assert parser.prog
    assert fake.load_calls == 0
    assert fake.step == 0


def test_session_rejects_invalid_cleanup_or_epoch_size():
    operations = BenchmarkOperations(
        prepare_optimizer_step=lambda: None,
        fetch_and_h2d=lambda: BenchmarkBatch(None, 1),
        forward_loss=lambda payload: None,
        backward=lambda loss: None,
        optimizer_and_clipping=lambda: None,
        synchronize=lambda: None,
        vram_probe=lambda: VramSnapshot(0, 0, 0, 0),
    )
    with pytest.raises(ValueError, match="epoch_samples"):
        benchmark_cli.BenchmarkSession(operations, 0, {}, lambda: None)
    with pytest.raises(TypeError, match="cleanup"):
        benchmark_cli.BenchmarkSession(operations, 1, {}, None)
