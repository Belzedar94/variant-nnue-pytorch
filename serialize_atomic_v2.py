#!/usr/bin/env python3
"""Explicit AtomicNNUEV2 checkpoint/network converter (never V1)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from atomic_v2.checkpoint import load_checkpoint, save_checkpoint
from atomic_v2.serialization import read_nnue, write_nnue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--description", default="AtomicNNUEV2 explicit conversion")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target.exists():
        raise FileExistsError(f"refusing to overwrite target: {args.target}")
    source_suffix = args.source.suffix.lower()
    target_suffix = args.target.suffix.lower()
    if source_suffix in (".pt", ".pth") and target_suffix == ".nnue":
        model, _, _ = load_checkpoint(args.source)
        temporary = args.target.with_name(args.target.name + ".tmp")
        if temporary.exists():
            raise FileExistsError(f"refusing to overwrite temporary target: {temporary}")
        try:
            with temporary.open("xb") as output:
                write_nnue(output, model, args.description)
            os.replace(temporary, args.target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    elif source_suffix == ".nnue" and target_suffix in (".pt", ".pth"):
        with args.source.open("rb") as source:
            model, _ = read_nnue(source)
        save_checkpoint(args.target, model, step=0)
    else:
        raise ValueError("V2 conversion must be .pt/.pth -> .nnue or .nnue -> .pt/.pth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
