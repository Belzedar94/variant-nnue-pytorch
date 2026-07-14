"""Strict adapter from the existing Atomic native loader to V2 batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch


ATOMIC_BIN_V2_SCHEMA_SHA256 = "0352b036f2a140c609e3eb9c9d635dc553e8d77253d8faa92437390f5cf93cb6"
ATOMIC_BIN_V2_MANIFEST_SCHEMA_SHA256 = (
    "83d63922df3ac4a0c81a21ec9d9fd9e180efe50f26efee62fe01710e09da5b42"
)
EXPECTED_V2_CAPABILITY = {
    "read": True,
    "write": False,
    "entrypoint": "manifest",
    "header_size": 96,
    "record_size": 64,
    "schema_sha256": ATOMIC_BIN_V2_SCHEMA_SHA256,
    "manifest_schema_sha256": ATOMIC_BIN_V2_MANIFEST_SCHEMA_SHA256,
}


class DatasetContractError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise DatasetContractError(f"duplicate manifest property: {key}")
        document[key] = value
    return document


def validate_v2_manifest_entrypoint(manifest: str | Path) -> Path:
    """Fail closed before handing a dataset path to the dual-format loader.

    The native provider deliberately retains Legacy Atomic V1 support for the
    historical trainer.  V2 must therefore authenticate its manifest boundary
    explicitly instead of relying on format auto-detection in that provider.
    Full schema, shard and record validation remains native and SHA-pinned.
    """

    path = Path(manifest)
    if not path.name.endswith(".atbin.manifest.json"):
        raise DatasetContractError(
            "AtomicNNUEV2 requires an .atbin.manifest.json entrypoint"
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        raw = path.read_bytes()
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except DatasetContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetContractError(f"invalid atomic-bin-v2 manifest: {error}") from error
    if not isinstance(document, dict):
        raise DatasetContractError("atomic-bin-v2 manifest root must be an object")
    if type(document.get("manifest_version")) is not int or document["manifest_version"] != 1:
        raise DatasetContractError("atomic-bin-v2 manifest_version must be 1")
    expected_fields = {
        "format": "atomic-bin-v2",
        "data_schema_sha256": ATOMIC_BIN_V2_SCHEMA_SHA256,
        "manifest_schema_sha256": ATOMIC_BIN_V2_MANIFEST_SCHEMA_SHA256,
    }
    for field, expected in expected_fields.items():
        if document.get(field) != expected:
            raise DatasetContractError(
                f"atomic-bin-v2 manifest {field} mismatch: {document.get(field)!r}"
            )
    return path


def validate_loader_capabilities(document: dict[str, Any]) -> None:
    if not isinstance(document, dict):
        raise DatasetContractError("native loader capabilities must be an object")
    if document.get("capability_version") != 2:
        raise DatasetContractError("native loader capability_version must be 2")
    formats = document.get("formats")
    if not isinstance(formats, dict):
        raise DatasetContractError("native loader capabilities are missing formats")
    actual = formats.get("atomic-bin-v2")
    if actual != EXPECTED_V2_CAPABILITY:
        raise DatasetContractError(
            f"native atomic-bin-v2 capability mismatch: {actual!r}"
        )


def validate_batch(batch: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    if not isinstance(batch, (tuple, list)) or len(batch) != 10:
        raise DatasetContractError("Atomic loader batch must contain exactly 10 tensors")
    tensors = tuple(batch)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise DatasetContractError("Atomic loader batch entries must all be tensors")
    (
        us,
        them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        outcome,
        score,
        psqt_indices,
        layer_stack_indices,
    ) = tensors
    batch_size = us.shape[0]
    if us.shape != (batch_size, 1) or them.shape != us.shape:
        raise DatasetContractError("us/them must have shape [batch, 1]")
    if white_indices.shape != black_indices.shape:
        raise DatasetContractError("white/black feature index shapes differ")
    if white_values.shape != white_indices.shape or black_values.shape != black_indices.shape:
        raise DatasetContractError("feature value shapes do not match feature indices")
    if white_indices.shape[0] != batch_size:
        raise DatasetContractError("feature batch size does not match us/them")
    if outcome.shape != (batch_size, 1) or score.shape != (batch_size, 1):
        raise DatasetContractError("outcome/score must have shape [batch, 1]")
    if psqt_indices.shape != (batch_size,) or layer_stack_indices.shape != (batch_size,):
        raise DatasetContractError("bucket tensors must have shape [batch]")
    if white_indices.dtype != torch.int32 or black_indices.dtype != torch.int32:
        raise DatasetContractError("feature indices must use torch.int32")
    if white_values.dtype != torch.float32 or black_values.dtype != torch.float32:
        raise DatasetContractError("feature values must use torch.float32")
    if any(value.dtype != torch.float32 for value in (us, them, outcome, score)):
        raise DatasetContractError("us/them/outcome/score must use torch.float32")
    if psqt_indices.dtype != torch.long or layer_stack_indices.dtype != torch.long:
        raise DatasetContractError("bucket indices must use torch.long")
    return tensors


def create_provider(
    manifest: str | Path,
    *,
    batch_size: int,
    num_workers: int = 1,
    device: str = "cpu",
    seed: int = 0,
    cyclic: bool = True,
):
    manifest_path = validate_v2_manifest_entrypoint(manifest)
    # Import lazily so contract/model/serializer unit tests do not require a
    # platform-specific native loader binary.
    import nnue_dataset

    validate_loader_capabilities(nnue_dataset.atomic_training_data_schemas())
    return nnue_dataset.SparseBatchProvider(
        "HalfKAv2",
        str(manifest_path),
        batch_size,
        cyclic=cyclic,
        num_workers=num_workers,
        filtered=False,
        random_fen_skipping=0,
        device=device,
        seed=seed,
    )


def validate_train_validation_manifests(training: str | Path, validation: str | Path) -> None:
    training_path = validate_v2_manifest_entrypoint(training)
    validation_path = validate_v2_manifest_entrypoint(validation)
    import nnue_dataset

    validate_loader_capabilities(nnue_dataset.atomic_training_data_schemas())
    nnue_dataset.validate_training_validation_data_paths(
        str(training_path), str(validation_path)
    )
