"""Explicit real-GPU benchmark gate for the repaired Atomic V3 trainer.

This command is diagnostic only.  It prepares the authenticated production
model/provider stack, runs a bounded set of ephemeral optimizer steps, prints a
single JSON report, and discards all mutated state.  It never calls the epoch or
campaign runners and has no checkpoint/output-directory option.

Heavy trainer, torch, CUDA, and native-provider imports are deliberately lazy:
importing this module cannot allocate a model, open a dataset, or start work.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Sequence, TextIO


# Permit direct execution from scripts/ without installing the repository.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))


from atomic_v3.performance_benchmark import (
    BenchmarkBatch,
    BenchmarkOperations,
    PerformanceBenchmarkConfig,
    PerformanceBenchmarkReport,
    VramSnapshot,
    run_performance_benchmark,
)


REAL_BENCHMARK_FORMAT = "atomic-v3-repair-real-performance-benchmark-v2"
RUN_IDS = (
    "lambda-0",
    "lambda-025",
    "lambda-050",
    "lambda-linear-015-050",
)


@dataclass(frozen=True)
class BenchmarkSession:
    """Prepared ephemeral backend returned by a real or test adapter."""

    operations: BenchmarkOperations
    epoch_samples: int
    identity: Mapping[str, object]
    cleanup: Callable[[], None]
    expected_authentication_events_per_epoch: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.operations, BenchmarkOperations):
            raise TypeError("session operations must be BenchmarkOperations")
        if (
            isinstance(self.epoch_samples, bool)
            or not isinstance(self.epoch_samples, int)
            or self.epoch_samples <= 0
        ):
            raise ValueError("session epoch_samples must be a positive integer")
        if not isinstance(self.identity, Mapping):
            raise TypeError("session identity must be a mapping")
        if not callable(self.cleanup):
            raise TypeError("session cleanup must be callable")
        expected = self.expected_authentication_events_per_epoch
        if (
            isinstance(expected, bool)
            or not isinstance(expected, (int, float))
            or not float(expected) >= 0.0
        ):
            raise ValueError(
                "expected_authentication_events_per_epoch must be non-negative"
            )
        object.__setattr__(
            self, "expected_authentication_events_per_epoch", float(expected)
        )
        object.__setattr__(
            self, "identity", MappingProxyType(dict(self.identity))
        )


BackendLoader = Callable[[argparse.Namespace], BenchmarkSession]


def _nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_number(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not parsed > 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "NON-PRODUCTION: benchmark the repaired Atomic V3 trainer with "
            "bounded ephemeral optimizer steps; no epochs or checkpoints"
        )
    )
    parser.add_argument(
        "--bootstrap-source",
        required=True,
        nargs=2,
        metavar=("RECEIPT", "RECEIPT_SHA256"),
        help="authenticated Atomic V3 bootstrap receipt and expected SHA-256",
    )
    parser.add_argument(
        "--provider-library",
        required=True,
        help="compiled native Atomic V3 provider DLL/shared object",
    )
    parser.add_argument(
        "--shared-initial-state",
        required=True,
        help="existing authenticated shared-initial-state.pt (read-only)",
    )
    parser.add_argument("--run", choices=RUN_IDS, default="lambda-0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--warmup-steps",
        type=_nonnegative_integer,
        default=5,
        help="ephemeral warm-up optimizer steps excluded from throughput",
    )
    parser.add_argument(
        "--measured-steps",
        type=int,
        default=10,
        help="measured optimizer steps; the harness accepts 10 through 20",
    )
    parser.add_argument(
        "--target-epoch-seconds",
        type=_positive_number,
        default=300.0,
        help="target conservative total epoch time (default: 300 seconds)",
    )
    parser.add_argument(
        "--hard-max-epoch-seconds",
        type=_positive_number,
        default=600.0,
        help=(
            "strict conservative total-epoch rejection boundary "
            "(default: above 600 seconds)"
        ),
    )
    return parser


def _load_real_session(arguments: argparse.Namespace) -> BenchmarkSession:
    """Lazily prepare the current repaired production stack without output IO."""

    import torch

    from atomic_v3.bootstrap_dataset import (
        BOOTSTRAP_RECORDS_PER_MANIFEST,
        inspect_bootstrap_roles,
    )
    from atomic_v3.executor import (
        MICROBATCH_SIZE,
        RANDOM_SKIP,
        TRAINING_SAMPLES_PER_EPOCH,
        ProviderBatch,
        densify_sparse_gradients,
        prepare_production_run,
        production_config,
    )
    from atomic_v3.production import (
        _load_shared_initial_state,
        _provider_factory,
        _validate_cuda_device,
        cleanup_prepared,
        sha256_file,
    )
    from atomic_v3.training import batch_loss

    provider_path = Path(os.path.abspath(os.fspath(arguments.provider_library)))
    if not provider_path.is_file():
        raise FileNotFoundError(f"provider library is not a file: {provider_path}")
    shared_path = Path(os.path.abspath(os.fspath(arguments.shared_initial_state)))
    if not shared_path.is_file():
        raise FileNotFoundError(
            f"shared initial state is not an existing file: {shared_path}"
        )

    resolved_device = _validate_cuda_device(arguments.device)
    receipt_path, receipt_sha256 = arguments.bootstrap_source
    snapshot = inspect_bootstrap_roles(receipt_path, receipt_sha256)
    provider_sha256 = sha256_file(provider_path)
    shared = _load_shared_initial_state(shared_path)
    run_config = production_config(arguments.run)
    torch.set_num_threads(run_config.threads)
    lambda_value = run_config.lambda_schedule.value(0)
    training_provider_factory = _provider_factory(
        snapshot,
        "train",
        provider_library=provider_path,
        provider_sha256=provider_sha256,
        device=resolved_device,
    )
    validation_provider_factory = _provider_factory(
        snapshot,
        "validation",
        provider_library=provider_path,
        provider_sha256=provider_sha256,
        device=resolved_device,
    )

    prepared = None
    try:
        prepared = prepare_production_run(
            run_config,
            training_provider_factory,
            validation_provider_factory,
            provider_library_sha256=provider_sha256,
            shared_initial_state=shared,
            device=resolved_device,
        )
        prepared.model.train()
        torch.cuda.reset_peak_memory_stats(resolved_device)

        def prepare_optimizer_step() -> None:
            prepared.optimizer.zero_grad(set_to_none=True)

        def fetch_and_h2d() -> BenchmarkBatch:
            provider_batch = prepared.training_provider.next_batch(MICROBATCH_SIZE)
            if not isinstance(provider_batch, ProviderBatch):
                raise TypeError("production provider did not return ProviderBatch")
            provider_batch.validate(MICROBATCH_SIZE, exact=True)
            return BenchmarkBatch(provider_batch.payload, provider_batch.samples)

        def forward_loss(payload):
            return batch_loss(
                prepared.model,
                payload,
                lambda_=lambda_value,
                validate=False,
                check_finite=False,
            )

        def backward(loss) -> None:
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise TypeError("Atomic V3 benchmark loss must be a scalar tensor")
            loss.backward()

        def optimizer_and_clipping() -> None:
            densify_sparse_gradients(prepared.model)
            prepared.optimizer.step()
            prepared.model.clip_training_weights()
            prepared.training_provider.commit()

        def synchronize() -> None:
            torch.cuda.synchronize(resolved_device)

        def vram_probe() -> VramSnapshot:
            return VramSnapshot(
                allocated_bytes=int(torch.cuda.memory_allocated(resolved_device)),
                reserved_bytes=int(torch.cuda.memory_reserved(resolved_device)),
                peak_allocated_bytes=int(
                    torch.cuda.max_memory_allocated(resolved_device)
                ),
                peak_reserved_bytes=int(
                    torch.cuda.max_memory_reserved(resolved_device)
                ),
            )

        operations = BenchmarkOperations(
            prepare_optimizer_step=prepare_optimizer_step,
            fetch_and_h2d=fetch_and_h2d,
            forward_loss=forward_loss,
            backward=backward,
            optimizer_and_clipping=optimizer_and_clipping,
            synchronize=synchronize,
            vram_probe=vram_probe,
        )
        identity = {
            "receipt_path": os.fspath(snapshot.receipt_path),
            "receipt_sha256": snapshot.receipt_sha256,
            "selection_sha256": snapshot.selection_sha256,
            "training_manifests": len(snapshot.train),
            "validation_manifests": len(snapshot.validation),
            "training_manifest_records": sum(item.records for item in snapshot.train),
            "validation_manifest_records": sum(
                item.records for item in snapshot.validation
            ),
            "provider_library_path": os.fspath(provider_path),
            "provider_library_sha256": provider_sha256,
            "shared_initial_state_path": os.fspath(shared_path),
            "shared_initial_state_file_sha256": sha256_file(shared_path),
            "shared_initial_state_state_sha256": shared.sha256,
            "device": resolved_device,
            "run_id": arguments.run,
            "lambda": lambda_value,
            "physical_batch_size": MICROBATCH_SIZE,
            "model": type(prepared.model).__name__,
            "model_parameters": sum(
                parameter.numel() for parameter in prepared.model.parameters()
            ),
            "run_config": run_config.to_document(
                microbatch_size=MICROBATCH_SIZE
            ),
        }

        def cleanup() -> None:
            cleanup_prepared(prepared)

        return BenchmarkSession(
            operations=operations,
            epoch_samples=TRAINING_SAMPLES_PER_EPOCH,
            identity=identity,
            cleanup=cleanup,
            expected_authentication_events_per_epoch=(
                TRAINING_SAMPLES_PER_EPOCH
                * (RANDOM_SKIP + 1)
                / BOOTSTRAP_RECORDS_PER_MANIFEST
            ),
        )
    except BaseException:
        cleanup_prepared(prepared)
        raise


def run_benchmark_command(
    arguments: argparse.Namespace,
    *,
    backend_loader: BackendLoader = _load_real_session,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, object], PerformanceBenchmarkReport]:
    """Run exactly one bounded benchmark session and always discard its state."""

    if not callable(backend_loader):
        raise TypeError("backend_loader must be callable")
    session = backend_loader(arguments)
    if not isinstance(session, BenchmarkSession):
        raise TypeError("backend_loader must return BenchmarkSession")
    try:
        config = PerformanceBenchmarkConfig(
            epoch_samples=session.epoch_samples,
            warmup_steps=arguments.warmup_steps,
            measured_steps=arguments.measured_steps,
            target_epoch_seconds=arguments.target_epoch_seconds,
            hard_max_epoch_seconds=arguments.hard_max_epoch_seconds,
            expected_authentication_events_per_epoch=(
                session.expected_authentication_events_per_epoch
            ),
        )
        report = run_performance_benchmark(
            session.operations, config, clock=clock
        )
        document = {
            "format": REAL_BENCHMARK_FORMAT,
            "status": report.decision.status.value,
            "gate_passed": not report.decision.rejected,
            "non_production": True,
            "training_campaign_started": False,
            "epochs_started": 0,
            "checkpoints_written": 0,
            "ephemeral_optimizer_steps": (
                config.warmup_steps + config.measured_steps
            ),
            "identity": dict(session.identity),
            "benchmark": report.to_document(),
        }
        return document, report
    finally:
        session.cleanup()


def _print_json(document: object, stream: TextIO) -> None:
    print(
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        file=stream,
        flush=True,
    )


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    backend_loader: Optional[BackendLoader] = None,
    clock: Callable[[], float] = time.perf_counter,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr
    loader = _load_real_session if backend_loader is None else backend_loader
    try:
        document, report = run_benchmark_command(
            arguments, backend_loader=loader, clock=clock
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as error:
        _print_json(
            {
                "format": REAL_BENCHMARK_FORMAT,
                "status": "failed",
                "non_production": True,
                "training_campaign_started": False,
                "epochs_started": 0,
                "checkpoints_written": 0,
                "error_type": type(error).__name__,
                "error": str(error),
            },
            errors,
        )
        return 1
    _print_json(document, output)
    return 2 if report.decision.rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
