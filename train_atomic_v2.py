#!/usr/bin/env python3
"""Train AtomicNNUEV2 from an authenticated atomic-bin-v2 manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from atomic_v2.checkpoint import checkpoint_document
from atomic_v2.dataset import (
    create_provider,
    validate_batch,
    validate_train_validation_manifests,
)
from atomic_v2.model import AtomicNNUEV2
from atomic_v2.serialization import write_nnue
from atomic_v2.training import atomic_loss, create_optimizer, train_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the isolated AtomicNNUEV2 SFNNv15 backend from scratch."
    )
    parser.add_argument("--train", required=True, type=Path, help="atomic-bin-v2 training manifest")
    parser.add_argument(
        "--validation", required=True, type=Path, help="distinct atomic-bin-v2 validation manifest"
    )
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--output-nnue", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="positions per step; default 128 on CPU and 16384 on CUDA",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=1_000_000,
        help="maximum validation positions; actual count is persisted",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.5e-3)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--description", default="AtomicNNUEV2 initial trainer export")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("steps", "num_workers", "validation_samples"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if not 0.0 <= args.lambda_ <= 1.0:
        raise ValueError("--lambda must be in [0, 1]")
    if args.output_checkpoint == args.output_nnue:
        raise ValueError("checkpoint and NNUE outputs must be different paths")
    for output in (args.output_checkpoint, args.output_nnue):
        if output.exists() or output.with_name(output.name + ".tmp").exists():
            raise FileExistsError(f"refusing to overwrite output: {output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is unavailable")


def default_batch_size(device: str) -> int:
    return 16_384 if torch.device(device).type == "cuda" else 128


def _validation_loss(
    model: AtomicNNUEV2,
    batches,
    lambda_: float,
    *,
    max_samples: int,
) -> tuple[float, int]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    weighted_loss = 0.0
    sample_count = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                values = validate_batch(batch)
                remaining = max_samples - sample_count
                if remaining <= 0:
                    break
                batch_samples = values[0].shape[0]
                if batch_samples > remaining:
                    values = tuple(value[:remaining] for value in values)
                    batch_samples = remaining
                inputs = values[:6] + values[8:]
                prediction = model(*inputs)
                loss = atomic_loss(prediction, values[6], values[7], lambda_=lambda_)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Atomic V2 validation loss is not finite")
                weighted_loss += float(loss) * batch_samples
                sample_count += batch_samples
                if sample_count >= max_samples:
                    break
    finally:
        model.train(was_training)
    if sample_count == 0:
        raise ValueError("validation manifest yielded no records")
    return weighted_loss / sample_count, sample_count


def main() -> int:
    args = parse_args()
    _validate_args(args)
    validate_train_validation_manifests(args.train, args.validation)
    batch_size = args.batch_size or default_batch_size(args.device)

    # Seed before model construction; official upstream currently seeds too
    # late for this deterministic Atomic contract.
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = AtomicNNUEV2().to(args.device)
    optimizer = create_optimizer(model, args.learning_rate)
    training = create_provider(
        args.train,
        batch_size=batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        cyclic=True,
    )

    losses = []
    for _ in range(args.steps):
        losses.append(train_batch(model, optimizer, next(training), lambda_=args.lambda_))

    validation = create_provider(
        args.validation,
        batch_size=batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        cyclic=False,
    )
    validation_loss, validation_samples = _validation_loss(
        model,
        validation,
        args.lambda_,
        max_samples=args.validation_samples,
    )

    checkpoint_tmp = args.output_checkpoint.with_name(args.output_checkpoint.name + ".tmp")
    nnue_tmp = args.output_nnue.with_name(args.output_nnue.name + ".tmp")
    committed: list[Path] = []
    try:
        checkpoint = checkpoint_document(model, step=args.steps, optimizer=optimizer)
        checkpoint["validation"] = {
            "sample_budget": args.validation_samples,
            "samples": validation_samples,
            "loss": validation_loss,
        }
        torch.save(checkpoint, checkpoint_tmp)
        with nnue_tmp.open("xb") as output:
            write_nnue(output, model, args.description)
        os.replace(checkpoint_tmp, args.output_checkpoint)
        committed.append(args.output_checkpoint)
        os.replace(nnue_tmp, args.output_nnue)
        committed.append(args.output_nnue)
    except BaseException:
        checkpoint_tmp.unlink(missing_ok=True)
        nnue_tmp.unlink(missing_ok=True)
        # Outputs did not exist before the command, so removing a successfully
        # renamed first member restores the all-or-nothing contract if the
        # second rename fails.
        for output in committed:
            output.unlink(missing_ok=True)
        raise

    print(
        json.dumps(
            {
                "backend": "atomic-nnue-v2",
                "steps": args.steps,
                "last_train_loss": losses[-1],
                "validation_loss": validation_loss,
                "validation_sample_budget": args.validation_samples,
                "validation_samples": validation_samples,
                "checkpoint": str(args.output_checkpoint),
                "network": str(args.output_nnue),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
