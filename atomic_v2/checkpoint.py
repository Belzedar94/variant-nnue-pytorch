"""Backend-tagged V2 checkpoints; never accepts legacy model objects."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .contract import BACKEND_KEY, FILE_VERSION, NETWORK_HASH
from .model import AtomicNNUEV2


class CheckpointError(ValueError):
    pass


def validate_checkpoint_document(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise CheckpointError("Atomic V2 checkpoint must be a dictionary")
    expected = {
        "backend": BACKEND_KEY,
        "file_version": FILE_VERSION,
        "network_hash": NETWORK_HASH,
    }
    for key, expected_value in expected.items():
        if document.get(key) != expected_value:
            raise CheckpointError(
                f"checkpoint {key} is {document.get(key)!r}, expected {expected_value!r}"
            )
    if not isinstance(document.get("model_state"), dict):
        raise CheckpointError("checkpoint is missing model_state")
    step = document.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise CheckpointError("checkpoint step must be a non-negative integer")
    return document


def checkpoint_document(
    model: AtomicNNUEV2,
    *,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    if not isinstance(model, AtomicNNUEV2):
        raise TypeError("Atomic V2 checkpoints accept only AtomicNNUEV2 models")
    document: dict[str, Any] = {
        "backend": BACKEND_KEY,
        "file_version": FILE_VERSION,
        "network_hash": NETWORK_HASH,
        "step": step,
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        document["optimizer_state"] = optimizer.state_dict()
    return validate_checkpoint_document(document)


def save_checkpoint(
    path: str | Path,
    model: AtomicNNUEV2,
    *,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    if target.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
    try:
        torch.save(checkpoint_document(model, step=step, optimizer=optimizer), temporary)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: str | Path,
    *,
    load_optimizer: bool = False,
) -> tuple[AtomicNNUEV2, int, dict[str, Any] | None]:
    # `weights_only=True` limits deserialization to tensors and primitive
    # containers; V1's pickled model object is therefore not an implicit path.
    document = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint_document(document)
    model = AtomicNNUEV2(initialize=False)
    try:
        model.load_state_dict(document["model_state"], strict=True)
    except RuntimeError as error:
        raise CheckpointError("checkpoint model_state does not match AtomicNNUEV2") from error
    optimizer_state = document.get("optimizer_state") if load_optimizer else None
    return model, document["step"], optimizer_state
