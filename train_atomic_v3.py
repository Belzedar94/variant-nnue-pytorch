"""Normative command-line launcher for the four AtomicNNUEV3 bootstrap runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Optional, Sequence

from atomic_v3.bootstrap_dataset import inspect_bootstrap_roles
from atomic_v3.checkpoint import CheckpointBinding, CommitBinding, DatasetBinding
from atomic_v3.executor import (
    EFFECTIVE_BATCH_SIZE,
    MICROBATCH_SIZE,
    RUN_CONFIGS,
    production_config,
)
from atomic_v3.production import execute_runs


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
        description=(
            "Train one or all four frozen AtomicNNUEV3 bootstrap networks "
            "through the authenticated native provider"
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run", choices=tuple(RUN_CONFIGS))
    selection.add_argument(
        "--all-runs",
        action="store_true",
        help="run lambda-0, lambda-025, lambda-050 and the linear schedule",
    )
    parser.add_argument(
        "--bootstrap-source",
        required=True,
        nargs=2,
        metavar=("RECEIPT", "RECEIPT_SHA256"),
        help="authenticated non-publication bootstrap receipt and expected SHA-256",
    )
    parser.add_argument(
        "--provider-library",
        help="absolute path to the compiled training_data_loader DLL/shared object",
    )
    parser.add_argument(
        "--output-root",
        "--output-dir",
        dest="output_root",
        required=True,
        help="root containing runs/, artifacts/, status and the shared state",
    )
    parser.add_argument(
        "--shared-initial-state",
        help=(
            "persistent seed-42 state file; defaults to "
            "<output-root>/shared-initial-state.pt"
        ),
    )
    parser.add_argument(
        "--trainer-commit",
        required=True,
        help="lowercase 40-digit trainer commit recorded in checkpoints and receipts",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="single CUDA device, for example cuda or cuda:0 (default: cuda)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="restore each unfinished run from its authenticated rolling last.ckpt",
    )
    parser.add_argument(
        "--microbatch-size",
        type=_positive_microbatch,
        default=MICROBATCH_SIZE,
        help=(
            "physical provider batch; frozen to 128 and accumulated to "
            f"{EFFECTIVE_BATCH_SIZE}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="authenticate the receipt and print bindings without model/provider creation",
    )
    return parser


def _selected_runs(arguments: argparse.Namespace) -> tuple[str, ...]:
    if getattr(arguments, "all_runs", False):
        return tuple(RUN_CONFIGS)
    run = getattr(arguments, "run", None)
    if run not in RUN_CONFIGS:
        raise ValueError("exactly one --run or --all-runs selection is required")
    return (run,)


def dry_run_document(arguments: argparse.Namespace) -> dict[str, object]:
    if not arguments.dry_run:
        raise ValueError("dry_run_document requires --dry-run")
    receipt_path, receipt_sha256 = arguments.bootstrap_source
    snapshot = inspect_bootstrap_roles(receipt_path, receipt_sha256)
    output_value = getattr(arguments, "output_root", None)
    if output_value is None:
        output_value = getattr(arguments, "output_dir")
    output_root = Path(os.path.abspath(os.fspath(output_value)))
    configs = []
    for run_id in _selected_runs(arguments):
        config = production_config(run_id)
        binding = CheckpointBinding(
            config=config.to_document(microbatch_size=arguments.microbatch_size),
            dataset=DatasetBinding.from_bootstrap(snapshot),
            commits=CommitBinding(arguments.trainer_commit),
        )
        configs.append(
            {
                "run_id": run_id,
                "config": binding.config_document(),
                "commits": binding.commits.to_document(),
            }
        )
    document = {
        "status": "validated-dry-run",
        "training_started": False,
        "output_root": os.fspath(output_root),
        "output_directory": os.fspath(output_root),
        "shared_initial_state": (
            os.path.abspath(os.fspath(arguments.shared_initial_state))
            if getattr(arguments, "shared_initial_state", None)
            else os.fspath(output_root / "shared-initial-state.pt")
        ),
        "microbatch_size": arguments.microbatch_size,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "checkpoint_name": "last.ckpt",
        "runs": configs,
        "dataset": DatasetBinding.from_bootstrap(snapshot).to_document(),
    }
    if len(configs) == 1:
        document["config"] = configs[0]["config"]
        document["commits"] = configs[0]["commits"]
    return document


def _print_json(document: object, *, stream=None) -> None:
    # Resolve stdout at call time so redirection/capture preserves the CLI's
    # single-document contract instead of retaining an import-time stream.
    if stream is None:
        stream = sys.stdout
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.dry_run:
        _print_json(dry_run_document(arguments))
        return 0
    if not arguments.provider_library:
        parser.error("--provider-library is required unless --dry-run is used")
    try:
        receipt_path, receipt_sha256 = arguments.bootstrap_source
        summary = execute_runs(
            receipt_path=receipt_path,
            receipt_sha256=receipt_sha256,
            provider_library=arguments.provider_library,
            output_root=arguments.output_root,
            trainer_commit=arguments.trainer_commit,
            run_ids=_selected_runs(arguments),
            shared_initial_state=arguments.shared_initial_state,
            device=arguments.device,
            resume=arguments.resume,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as error:
        _print_json(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
            stream=sys.stderr,
        )
        return 1
    _print_json(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
