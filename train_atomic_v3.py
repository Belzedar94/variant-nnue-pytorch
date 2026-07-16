"""Validate one AtomicNNUEV3 bootstrap run without starting training.

The native provider and final NNUE serializer are intentionally not selected
by this Slice2 entry point.  Real execution is exposed through
``atomic_v3.executor.run_production`` once those audited dependencies are
injected; this CLI is fail-closed and dry-run-only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

from atomic_v3.bootstrap_dataset import inspect_bootstrap_roles
from atomic_v3.checkpoint import CheckpointBinding, CommitBinding, DatasetBinding
from atomic_v3.executor import (
    EFFECTIVE_BATCH_SIZE,
    MICROBATCH_SIZE,
    RUN_CONFIGS,
    production_config,
)


def _positive_microbatch(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed != MICROBATCH_SIZE:
        raise argparse.ArgumentTypeError(
            f"must be exactly {MICROBATCH_SIZE} for the bootstrap runs"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a strict AtomicNNUEV3 bootstrap training run"
    )
    parser.add_argument("--run", required=True, choices=tuple(RUN_CONFIGS))
    parser.add_argument(
        "--bootstrap-source",
        required=True,
        nargs=2,
        metavar=("RECEIPT", "RECEIPT_SHA256"),
        help="authenticated non-publication bootstrap receipt",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--trainer-commit",
        required=True,
        help="lowercase 40-digit trainer commit recorded in checkpoints",
    )
    parser.add_argument(
        "--microbatch-size",
        type=_positive_microbatch,
        default=MICROBATCH_SIZE,
        help=(
            "physical provider batch; gradients are weighted to the frozen "
            f"effective batch {EFFECTIVE_BATCH_SIZE}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="authenticate inputs and print the frozen configuration",
    )
    return parser


def dry_run_document(arguments: argparse.Namespace) -> dict[str, object]:
    if not arguments.dry_run:
        raise ValueError(
            "this entry point is dry-run-only until the audited native provider is injected"
        )
    receipt_path, receipt_sha256 = arguments.bootstrap_source
    snapshot = inspect_bootstrap_roles(receipt_path, receipt_sha256)
    config = production_config(arguments.run)
    binding = CheckpointBinding(
        config=config.to_document(microbatch_size=arguments.microbatch_size),
        dataset=DatasetBinding.from_bootstrap(snapshot),
        commits=CommitBinding(arguments.trainer_commit),
    )
    output_directory = Path(os.path.abspath(os.fspath(arguments.output_dir)))
    return {
        "status": "validated-dry-run",
        "training_started": False,
        "output_directory": os.fspath(output_directory),
        "microbatch_size": arguments.microbatch_size,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "checkpoint_name": "last.ckpt",
        "config": binding.config_document(),
        "dataset": binding.dataset.to_document(),
        "commits": binding.commits.to_document(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if not arguments.dry_run:
        parser.error(
            "execution is unavailable until the audited native provider is injected; "
            "pass --dry-run"
        )
    document = dry_run_document(arguments)
    print(
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
