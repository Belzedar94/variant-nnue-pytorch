"""Atomic, backend-bound checkpoints for the AtomicNNUEV3 bootstrap runs."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from .bootstrap_dataset import (
    BOOTSTRAP_PROVENANCE_CLASS,
    BOOTSTRAP_RECEIPT_TYPE,
    BOOTSTRAP_TRAIN_MANIFESTS,
    BOOTSTRAP_VALIDATION_MANIFESTS,
    BootstrapReceiptSnapshot,
)
from .contract import BACKEND_KEY, FILE_VERSION, NETWORK_HASH


CHECKPOINT_FORMAT = "atomic-v3-bootstrap-checkpoint-v1"
ENGINE_PIN = "420c9f35266fbdc2167dc5b9d8d20d90281c60c9"
BOOTSTRAP_VERIFIER_COMMIT = "dc453080721d97640948cf5a2c0cf0d536570a70"
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DOCUMENT_KEYS = {
    "checkpoint_format",
    "backend",
    "file_version",
    "network_hash",
    "config",
    "dataset",
    "commits",
    "counters",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "logical_cursor",
    "rng_state",
}
_COUNTER_KEYS = {
    "completed_epochs",
    "global_steps",
    "training_samples",
    "validation_samples",
    "last_epoch_training_samples",
    "last_epoch_validation_samples",
    "last_train_loss",
    "last_validation_loss",
    "last_lambda",
}
_RNG_KEYS = {"python", "numpy", "torch_cpu", "torch_cuda"}
_DATASET_KEYS = {
    "receipt_type",
    "receipt_path",
    "receipt_sha256",
    "selection_sha256",
    "semantic_validation_jsonl_path",
    "semantic_validation_jsonl_sha256",
    "semantic_validation_jsonl_domain_sha256",
    "provenance_class",
    "dataset_publication_ready",
    "release_candidate_eligible",
    "train_manifests",
    "validation_manifests",
}
_MANIFEST_KEYS = {"path", "sha256", "records"}
_COMMIT_KEYS = {"trainer_commit", "engine_commit", "bootstrap_verifier_commit"}


class CheckpointError(ValueError):
    pass


def _plain_uint(label: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointError(f"{label} must be a non-negative integer")
    return value


def _sha1(label: str, value: object) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise CheckpointError(f"{label} must be lowercase 40-digit git hex")
    return value


def _sha256(label: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CheckpointError(f"{label} must be lowercase SHA-256 hex")
    return value


@dataclass(frozen=True)
class DatasetBinding:
    receipt_path: str
    receipt_sha256: str
    selection_sha256: str
    semantic_validation_jsonl_path: str
    semantic_validation_jsonl_sha256: str
    semantic_validation_jsonl_domain_sha256: str
    train_manifests: tuple[tuple[str, str, int], ...]
    validation_manifests: tuple[tuple[str, str, int], ...]
    provenance_class: str = BOOTSTRAP_PROVENANCE_CLASS
    dataset_publication_ready: bool = False
    release_candidate_eligible: bool = False

    @classmethod
    def from_bootstrap(cls, snapshot: BootstrapReceiptSnapshot) -> "DatasetBinding":
        if not isinstance(snapshot, BootstrapReceiptSnapshot):
            raise TypeError("dataset binding requires a BootstrapReceiptSnapshot")
        return cls(
            receipt_path=str(snapshot.receipt_path),
            receipt_sha256=snapshot.receipt_sha256,
            selection_sha256=snapshot.selection_sha256,
            semantic_validation_jsonl_path=str(snapshot.semantic_validation_jsonl_path),
            semantic_validation_jsonl_sha256=(
                snapshot.semantic_validation_jsonl_sha256
            ),
            semantic_validation_jsonl_domain_sha256=(
                snapshot.semantic_validation_jsonl_domain_sha256
            ),
            train_manifests=tuple(
                (str(item.path), item.sha256, item.records) for item in snapshot.train
            ),
            validation_manifests=tuple(
                (str(item.path), item.sha256, item.records)
                for item in snapshot.validation
            ),
            provenance_class=snapshot.provenance_class,
            dataset_publication_ready=snapshot.dataset_publication_ready,
            release_candidate_eligible=snapshot.release_candidate_eligible,
        )

    def _manifest_documents(
        self, role: str, values: Sequence[tuple[str, str, int]], expected_count: int
    ) -> list[dict[str, object]]:
        if len(values) != expected_count:
            raise CheckpointError(f"{role} manifest count differs from bootstrap contract")
        documents = []
        for index, value in enumerate(values):
            if not isinstance(value, tuple) or len(value) != 3:
                raise CheckpointError(f"{role} manifest {index} binding is malformed")
            path, digest, records = value
            if not isinstance(path, str) or not Path(path).is_absolute():
                raise CheckpointError(f"{role} manifest {index} path must be absolute")
            documents.append(
                {
                    "path": path,
                    "sha256": _sha256(f"{role} manifest {index}", digest),
                    "records": _plain_uint(f"{role} manifest {index} records", records),
                }
            )
        return documents

    def to_document(self) -> dict[str, object]:
        if not isinstance(self.receipt_path, str) or not Path(self.receipt_path).is_absolute():
            raise CheckpointError("bootstrap receipt path must be absolute")
        if (
            not isinstance(self.semantic_validation_jsonl_path, str)
            or not Path(self.semantic_validation_jsonl_path).is_absolute()
        ):
            raise CheckpointError("semantic validation JSONL path must be absolute")
        if self.provenance_class != BOOTSTRAP_PROVENANCE_CLASS:
            raise CheckpointError("checkpoint dataset is not non-publication-bootstrap")
        if self.dataset_publication_ready is not False:
            raise CheckpointError("bootstrap checkpoint cannot be publication-ready")
        if self.release_candidate_eligible is not False:
            raise CheckpointError("bootstrap checkpoint cannot be release-candidate-eligible")
        return {
            "receipt_type": BOOTSTRAP_RECEIPT_TYPE,
            "receipt_path": self.receipt_path,
            "receipt_sha256": _sha256("receipt_sha256", self.receipt_sha256),
            "selection_sha256": _sha256("selection_sha256", self.selection_sha256),
            "semantic_validation_jsonl_path": self.semantic_validation_jsonl_path,
            "semantic_validation_jsonl_sha256": _sha256(
                "semantic_validation_jsonl_sha256",
                self.semantic_validation_jsonl_sha256,
            ),
            "semantic_validation_jsonl_domain_sha256": _sha256(
                "semantic_validation_jsonl_domain_sha256",
                self.semantic_validation_jsonl_domain_sha256,
            ),
            "provenance_class": self.provenance_class,
            "dataset_publication_ready": False,
            "release_candidate_eligible": False,
            "train_manifests": self._manifest_documents(
                "train", self.train_manifests, BOOTSTRAP_TRAIN_MANIFESTS
            ),
            "validation_manifests": self._manifest_documents(
                "validation",
                self.validation_manifests,
                BOOTSTRAP_VALIDATION_MANIFESTS,
            ),
        }


@dataclass(frozen=True)
class CommitBinding:
    trainer_commit: str
    engine_commit: str = ENGINE_PIN
    bootstrap_verifier_commit: str = BOOTSTRAP_VERIFIER_COMMIT

    def to_document(self) -> dict[str, str]:
        trainer = _sha1("trainer_commit", self.trainer_commit)
        engine = _sha1("engine_commit", self.engine_commit)
        verifier = _sha1("bootstrap_verifier_commit", self.bootstrap_verifier_commit)
        if engine != ENGINE_PIN:
            raise CheckpointError("checkpoint engine commit differs from audited pin")
        if verifier != BOOTSTRAP_VERIFIER_COMMIT:
            raise CheckpointError("checkpoint bootstrap verifier commit differs")
        return {
            "trainer_commit": trainer,
            "engine_commit": engine,
            "bootstrap_verifier_commit": verifier,
        }


@dataclass(frozen=True)
class CheckpointBinding:
    config: Mapping[str, object]
    dataset: DatasetBinding
    commits: CommitBinding

    def config_document(self) -> dict[str, object]:
        if not isinstance(self.config, Mapping):
            raise CheckpointError("checkpoint config binding must be a mapping")
        return dict(self.config)


@dataclass(frozen=True)
class TrainingCounters:
    completed_epochs: int = 0
    global_steps: int = 0
    training_samples: int = 0
    validation_samples: int = 0
    last_epoch_training_samples: int = 0
    last_epoch_validation_samples: int = 0
    last_train_loss: Optional[float] = None
    last_validation_loss: Optional[float] = None
    last_lambda: Optional[float] = None

    def to_document(self) -> dict[str, object]:
        document = {
            "completed_epochs": _plain_uint("completed_epochs", self.completed_epochs),
            "global_steps": _plain_uint("global_steps", self.global_steps),
            "training_samples": _plain_uint("training_samples", self.training_samples),
            "validation_samples": _plain_uint(
                "validation_samples", self.validation_samples
            ),
            "last_epoch_training_samples": _plain_uint(
                "last_epoch_training_samples", self.last_epoch_training_samples
            ),
            "last_epoch_validation_samples": _plain_uint(
                "last_epoch_validation_samples", self.last_epoch_validation_samples
            ),
            "last_train_loss": self.last_train_loss,
            "last_validation_loss": self.last_validation_loss,
            "last_lambda": self.last_lambda,
        }
        for field in ("last_train_loss", "last_validation_loss", "last_lambda"):
            value = document[field]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
            ):
                raise CheckpointError(f"{field} must be finite or null")
            if value is not None:
                document[field] = float(value)
        return document

    @classmethod
    def from_document(cls, value: object) -> "TrainingCounters":
        if not isinstance(value, Mapping) or set(value) != _COUNTER_KEYS:
            raise CheckpointError("checkpoint counters differ from contract")
        counters = cls(**dict(value))
        counters.to_document()
        return counters


def capture_rng_state() -> dict[str, object]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    return {
        "python": {
            "version": python_state[0],
            "state": list(python_state[1]),
            "gaussian": python_state[2],
        },
        "numpy": {
            "algorithm": numpy_state[0],
            "state": numpy_state[1].astype(np.uint32, copy=False).tolist(),
            "position": numpy_state[2],
            "has_gaussian": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.random.get_rng_state().clone(),
        "torch_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized()
            else []
        ),
    }


def _validate_rng_state(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _RNG_KEYS:
        raise CheckpointError("checkpoint RNG state differs from contract")
    python_state = value["python"]
    numpy_state = value["numpy"]
    if not isinstance(python_state, Mapping) or set(python_state) != {
        "version",
        "state",
        "gaussian",
    }:
        raise CheckpointError("checkpoint Python RNG state is malformed")
    if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
        "algorithm",
        "state",
        "position",
        "has_gaussian",
        "cached_gaussian",
    }:
        raise CheckpointError("checkpoint NumPy RNG state is malformed")
    if not isinstance(value["torch_cpu"], torch.Tensor):
        raise CheckpointError("checkpoint torch CPU RNG state is missing")
    cuda_states = value["torch_cuda"]
    if not isinstance(cuda_states, list) or any(
        not isinstance(state, torch.Tensor) for state in cuda_states
    ):
        raise CheckpointError("checkpoint torch CUDA RNG state is malformed")
    return value


def restore_rng_state(value: object) -> None:
    state = _validate_rng_state(value)
    python_state = state["python"]
    numpy_state = state["numpy"]
    assert isinstance(python_state, Mapping)
    assert isinstance(numpy_state, Mapping)
    random.setstate(
        (
            int(python_state["version"]),
            tuple(int(item) for item in python_state["state"]),
            python_state["gaussian"],
        )
    )
    np.random.set_state(
        (
            str(numpy_state["algorithm"]),
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gaussian"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.random.set_rng_state(state["torch_cpu"])
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise CheckpointError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_states)


def _validate_dataset_document(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _DATASET_KEYS:
        raise CheckpointError("checkpoint dataset fields differ from bootstrap contract")
    if value["receipt_type"] != BOOTSTRAP_RECEIPT_TYPE:
        raise CheckpointError("checkpoint dataset receipt type differs")
    if value["provenance_class"] != BOOTSTRAP_PROVENANCE_CLASS:
        raise CheckpointError("checkpoint dataset is not non-publication-bootstrap")
    if value["dataset_publication_ready"] is not False:
        raise CheckpointError("bootstrap checkpoint cannot be publication-ready")
    if value["release_candidate_eligible"] is not False:
        raise CheckpointError("bootstrap checkpoint cannot be release-candidate-eligible")
    for field in ("receipt_path", "semantic_validation_jsonl_path"):
        path = value[field]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise CheckpointError(f"checkpoint dataset {field} must be absolute")
    for field in (
        "receipt_sha256",
        "selection_sha256",
        "semantic_validation_jsonl_sha256",
        "semantic_validation_jsonl_domain_sha256",
    ):
        _sha256(field, value[field])
    for role, expected_count in (
        ("train_manifests", BOOTSTRAP_TRAIN_MANIFESTS),
        ("validation_manifests", BOOTSTRAP_VALIDATION_MANIFESTS),
    ):
        manifests = value[role]
        if not isinstance(manifests, list) or len(manifests) != expected_count:
            raise CheckpointError(f"checkpoint {role} count differs")
        for index, manifest in enumerate(manifests):
            if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_KEYS:
                raise CheckpointError(f"checkpoint {role} manifest {index} is malformed")
            path = manifest["path"]
            if not isinstance(path, str) or not Path(path).is_absolute():
                raise CheckpointError(
                    f"checkpoint {role} manifest {index} path must be absolute"
                )
            _sha256(f"checkpoint {role} manifest {index}", manifest["sha256"])
            _plain_uint(
                f"checkpoint {role} manifest {index} records", manifest["records"]
            )
    return value


def _validate_commit_document(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _COMMIT_KEYS:
        raise CheckpointError("checkpoint commit fields differ from contract")
    _sha1("trainer_commit", value["trainer_commit"])
    if _sha1("engine_commit", value["engine_commit"]) != ENGINE_PIN:
        raise CheckpointError("checkpoint engine commit differs from audited pin")
    if (
        _sha1("bootstrap_verifier_commit", value["bootstrap_verifier_commit"])
        != BOOTSTRAP_VERIFIER_COMMIT
    ):
        raise CheckpointError("checkpoint bootstrap verifier commit differs")
    return value


def checkpoint_document(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    logical_cursor: Mapping[str, object],
    counters: TrainingCounters,
    binding: CheckpointBinding,
) -> dict[str, object]:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("checkpoint model must be a torch module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("checkpoint optimizer must be a torch optimizer")
    if not isinstance(logical_cursor, Mapping):
        raise TypeError("logical cursor must be a mapping")
    document = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "backend": BACKEND_KEY,
        "file_version": FILE_VERSION,
        "network_hash": NETWORK_HASH,
        "config": binding.config_document(),
        "dataset": binding.dataset.to_document(),
        "commits": binding.commits.to_document(),
        "counters": counters.to_document(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "logical_cursor": dict(logical_cursor),
        "rng_state": capture_rng_state(),
    }
    return validate_checkpoint_document(document)


def validate_checkpoint_document(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _DOCUMENT_KEYS:
        raise CheckpointError("Atomic V3 checkpoint fields differ from contract")
    expected = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "backend": BACKEND_KEY,
        "file_version": FILE_VERSION,
        "network_hash": NETWORK_HASH,
    }
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            raise CheckpointError(f"checkpoint {field} differs from AtomicNNUEV3")
    for field in (
        "config",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "logical_cursor",
    ):
        if not isinstance(value[field], Mapping):
            raise CheckpointError(f"checkpoint {field} must be a mapping")
    _validate_dataset_document(value["dataset"])
    _validate_commit_document(value["commits"])
    TrainingCounters.from_document(value["counters"])
    _validate_rng_state(value["rng_state"])
    return value


def _atomic_replace(target: Path, document: Mapping[str, object]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".last.ckpt.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(document), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def save_last_checkpoint(output_directory: Union[str, Path], document: object) -> Path:
    validated = validate_checkpoint_document(document)
    target = Path(output_directory) / "last.ckpt"
    _atomic_replace(target, validated)
    return target


def load_last_checkpoint(
    output_directory: Union[str, Path], expected: CheckpointBinding
) -> dict[str, object]:
    target = Path(output_directory) / "last.ckpt"
    document = torch.load(target, map_location="cpu", weights_only=True)
    validate_checkpoint_document(document)
    expected_documents = {
        "config": expected.config_document(),
        "dataset": expected.dataset.to_document(),
        "commits": expected.commits.to_document(),
    }
    for field, expected_value in expected_documents.items():
        if document[field] != expected_value:
            raise CheckpointError(f"checkpoint {field} is incompatible with this run")
    return document


def restore_checkpoint(
    document: object,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    provider: Any,
) -> TrainingCounters:
    validated = validate_checkpoint_document(document)
    try:
        model.load_state_dict(validated["model_state"], strict=True)
    except RuntimeError as error:
        raise CheckpointError("checkpoint model state is incompatible") from error
    optimizer.load_state_dict(validated["optimizer_state"])
    scheduler.load_state_dict(validated["scheduler_state"])
    restore = getattr(provider, "restore_logical_cursor", None)
    if not callable(restore):
        raise TypeError("provider must implement restore_logical_cursor")
    restore(validated["logical_cursor"])
    restore_rng_state(validated["rng_state"])
    return TrainingCounters.from_document(validated["counters"])


def checkpoint_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "BOOTSTRAP_VERIFIER_COMMIT",
    "CHECKPOINT_FORMAT",
    "ENGINE_PIN",
    "CheckpointBinding",
    "CheckpointError",
    "CommitBinding",
    "DatasetBinding",
    "TrainingCounters",
    "capture_rng_state",
    "checkpoint_document",
    "checkpoint_sha256",
    "load_last_checkpoint",
    "restore_checkpoint",
    "restore_rng_state",
    "save_last_checkpoint",
    "validate_checkpoint_document",
]
