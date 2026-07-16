"""Two-step restore-and-continue GPU canary for AtomicNNUEV3 production.

This diagnostic intentionally does not call ``run_production`` and therefore
cannot start a 37-epoch run. It checkpoints one exact effective optimizer step,
restores into fresh state, executes the next step, checkpoints again, then uses
the same strict wire serializer as production.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Optional, Sequence

import torch

# Permit direct execution from scripts/ without installing the repository.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from atomic_v3.bootstrap_dataset import inspect_bootstrap_roles
from atomic_v3.checkpoint import (
    CheckpointBinding,
    CommitBinding,
    DatasetBinding,
    TrainingCounters,
    checkpoint_document,
    checkpoint_sha256,
    load_last_checkpoint,
    restore_checkpoint,
    save_last_checkpoint,
)
from atomic_v3.executor import (
    EFFECTIVE_BATCH_SIZE,
    MICROBATCH_SIZE,
    SharedInitialState,
    prepare_production_run,
    production_config,
    train_epoch,
)
from atomic_v3.production import (
    _provider_factory,
    _validate_cuda_device,
    cleanup_prepared,
    load_or_create_shared_initial_state,
    sha256_file,
)
from atomic_v3.serialization import check_nnue, read_nnue, save_nnue


CANARY_FORMAT = "atomic-v3-real-gpu-resume-canary-v2"


def _set_frozen_threads() -> None:
    threads = production_config("lambda-0").threads
    torch.set_num_threads(threads)
    if torch.get_num_threads() != threads:
        raise RuntimeError("canary failed to apply the frozen torch thread count")


def _run_canary_optimizer_step(prepared):
    """Apply the frozen CPU-thread contract and execute one exact GPU step."""

    _set_frozen_threads()
    return train_epoch(
        prepared.model,
        prepared.optimizer,
        prepared.training_provider,
        lambda_value=0.0,
        sample_budget=EFFECTIVE_BATCH_SIZE,
        effective_batch_size=EFFECTIVE_BATCH_SIZE,
        microbatch_size=MICROBATCH_SIZE,
        require_full_microbatches=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one 16,384-position GPU optimizer step, checkpoint/restore, "
            "a second resumed step and V3 serialize/load without starting "
            "production epochs"
        )
    )
    parser.add_argument(
        "--bootstrap-source",
        required=True,
        nargs=2,
        metavar=("RECEIPT", "RECEIPT_SHA256"),
    )
    parser.add_argument("--provider-library", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--shared-initial-state", required=True)
    parser.add_argument("--trainer-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _validate_commit(value: str) -> str:
    if (
        len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("trainer commit must be a lowercase 40-digit SHA-1")
    return value


def _prepare(
    *,
    snapshot,
    provider_library: Path,
    provider_sha256: str,
    shared: SharedInitialState,
    device: str,
):
    config = production_config("lambda-0")
    return prepare_production_run(
        config,
        _provider_factory(
            snapshot,
            "train",
            provider_library=provider_library,
            device=device,
        ),
        _provider_factory(
            snapshot,
            "validation",
            provider_library=provider_library,
            device=device,
        ),
        provider_library_sha256=provider_sha256,
        shared_initial_state=shared,
        device=device,
    )


def run_canary(arguments: argparse.Namespace) -> dict[str, object]:
    trainer_commit = _validate_commit(arguments.trainer_commit)
    device = _validate_cuda_device(arguments.device)
    provider_library = Path(os.path.abspath(arguments.provider_library))
    if not provider_library.is_file():
        raise FileNotFoundError(f"provider library is not a file: {provider_library}")
    provider_sha256 = sha256_file(provider_library)
    work_directory = Path(os.path.abspath(arguments.work_dir))
    if work_directory.exists() and any(work_directory.iterdir()):
        raise FileExistsError(f"canary work directory is not empty: {work_directory}")
    work_directory.mkdir(parents=True, exist_ok=True)
    receipt_path, receipt_sha256 = arguments.bootstrap_source
    snapshot = inspect_bootstrap_roles(receipt_path, receipt_sha256)
    shared, shared_artifact = load_or_create_shared_initial_state(
        arguments.shared_initial_state
    )

    first = None
    second = None
    imported = None
    try:
        first = _prepare(
            snapshot=snapshot,
            provider_library=provider_library,
            provider_sha256=provider_sha256,
            shared=shared,
            device=device,
        )
        binding = CheckpointBinding(
            config=first.checkpoint_config_document(),
            dataset=DatasetBinding.from_bootstrap(snapshot),
            commits=CommitBinding(trainer_commit),
        )
        metric = _run_canary_optimizer_step(first)
        if metric.samples != EFFECTIVE_BATCH_SIZE or metric.steps != 1:
            raise RuntimeError("canary did not complete exactly one optimizer step")
        cursor = first.training_provider.logical_cursor_state()
        if (
            cursor.get("accepted_samples") != EFFECTIVE_BATCH_SIZE
            or cursor.get("next_batch_sequence")
            != EFFECTIVE_BATCH_SIZE // MICROBATCH_SIZE
        ):
            raise RuntimeError("canary provider cursor differs after one step")
        counters = TrainingCounters(
            global_steps=1,
            training_samples=EFFECTIVE_BATCH_SIZE,
            last_epoch_training_samples=EFFECTIVE_BATCH_SIZE,
            last_train_loss=metric.mean_loss,
            last_lambda=0.0,
        )
        checkpoint_directory = work_directory / "checkpoint"
        checkpoint_path = save_last_checkpoint(
            checkpoint_directory,
            checkpoint_document(
                first.model,
                first.optimizer,
                first.scheduler,
                cursor,
                counters,
                binding,
            ),
        )
        restore_checkpoint_hash = checkpoint_sha256(checkpoint_path)
        cleanup_prepared(first)
        first = None

        second = _prepare(
            snapshot=snapshot,
            provider_library=provider_library,
            provider_sha256=provider_sha256,
            shared=shared,
            device=device,
        )
        second_binding = CheckpointBinding(
            config=second.checkpoint_config_document(),
            dataset=DatasetBinding.from_bootstrap(snapshot),
            commits=CommitBinding(trainer_commit),
        )
        document = load_last_checkpoint(checkpoint_directory, second_binding)
        restored = restore_checkpoint(
            document,
            second.model,
            second.optimizer,
            second.scheduler,
            second.training_provider,
        )
        if restored != counters:
            raise RuntimeError("canary checkpoint counters changed on restore")
        if second.training_provider.logical_cursor_state() != cursor:
            raise RuntimeError("canary provider cursor changed on restore")

        resumed_metric = _run_canary_optimizer_step(second)
        if resumed_metric.samples != EFFECTIVE_BATCH_SIZE or resumed_metric.steps != 1:
            raise RuntimeError("resumed canary did not complete exactly one optimizer step")
        final_cursor = second.training_provider.logical_cursor_state()
        expected_batches = 2 * EFFECTIVE_BATCH_SIZE // MICROBATCH_SIZE
        if (
            final_cursor.get("accepted_samples") != 2 * EFFECTIVE_BATCH_SIZE
            or final_cursor.get("next_batch_sequence") != expected_batches
        ):
            raise RuntimeError("resumed canary cursor differs after the second step")
        final_counters = TrainingCounters(
            global_steps=2,
            training_samples=2 * EFFECTIVE_BATCH_SIZE,
            last_epoch_training_samples=EFFECTIVE_BATCH_SIZE,
            last_train_loss=resumed_metric.mean_loss,
            last_lambda=0.0,
        )
        checkpoint_path = save_last_checkpoint(
            checkpoint_directory,
            checkpoint_document(
                second.model,
                second.optimizer,
                second.scheduler,
                final_cursor,
                final_counters,
                second_binding,
            ),
        )
        final_checkpoint_hash = checkpoint_sha256(checkpoint_path)

        description = json.dumps(
            {
                "format": CANARY_FORMAT,
                "trainer_commit": trainer_commit,
                "receipt_sha256": snapshot.receipt_sha256,
                "initial_state_sha256": shared.sha256,
                "restore_checkpoint_sha256": restore_checkpoint_hash,
                "final_checkpoint_sha256": final_checkpoint_hash,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        network_path = work_directory / "atomic-v3-resume-canary.nnue"
        metadata = save_nnue(network_path, second.model, description)
        with network_path.open("rb") as stream:
            checked = check_nnue(stream)
        with network_path.open("rb") as stream:
            imported, imported_description = read_nnue(stream)
        if imported_description != description or checked != metadata:
            raise RuntimeError("canary serialized network changed on strict load")
        result = {
            "format": CANARY_FORMAT,
            "status": "passed",
            "training_started": False,
            "optimizer_steps": 2,
            "samples": metric.samples + resumed_metric.samples,
            "first_step_mean_loss": metric.mean_loss,
            "resumed_step_mean_loss": resumed_metric.mean_loss,
            "device": device,
            "receipt_sha256": snapshot.receipt_sha256,
            "provider_library": {
                "path": os.fspath(provider_library),
                "sha256": provider_sha256,
            },
            "shared_initial_state": shared_artifact,
            "checkpoint": {
                "path": os.fspath(checkpoint_path),
                "sha256": final_checkpoint_hash,
            },
            "restore_checkpoint_sha256": restore_checkpoint_hash,
            "network": {
                "path": os.fspath(network_path),
                "bytes": metadata.size,
                "sha256": metadata.sha256.lower(),
            },
            "restored_cursor": cursor,
            "final_cursor": final_cursor,
        }
        result_path = work_directory / "canary-result.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
            newline="\n",
        )
        return result
    finally:
        del imported
        cleanup_prepared(first)
        cleanup_prepared(second)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = run_canary(arguments)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as error:
        print(
            json.dumps(
                {
                    "format": CANARY_FORMAT,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
