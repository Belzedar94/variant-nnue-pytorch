"""Normative AtomicNNUEV3 bootstrap training orchestration.

This module deliberately has one execution graph: authenticate the fixed
bootstrap receipt, prepare the canonical native providers/model, execute the
frozen trainer, and publish exactly one epoch-37 V3 network.  It contains no
alternate reader, optimizer, loss, or serializer path.
"""

from __future__ import annotations

import gc
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Optional, Sequence

import torch

from .bootstrap_dataset import BootstrapReceiptSnapshot, inspect_bootstrap_roles
from .checkpoint import (
    CheckpointBinding,
    CommitBinding,
    DatasetBinding,
    TrainingCounters,
    checkpoint_sha256,
)
from .executor import (
    EPOCHS,
    MICROBATCH_SIZE,
    RUN_CONFIGS,
    PreparedProductionRun,
    SharedInitialState,
    create_shared_initial_state,
    prepare_production_run,
    production_config,
    run_production,
)
from .native_provider import NativeAtomicV3Provider
from .serialization import WireMetadata, check_nnue, save_nnue


SHARED_INITIAL_STATE_FORMAT = "atomic-v3-shared-initial-state-v1"
FINAL_RECEIPT_FORMAT = "atomic-v3-training-final-receipt-v1"
SUMMARY_FORMAT = "atomic-v3-training-summary-v1"
STATUS_FORMAT = "atomic-v3-training-status-v1"


class ProductionLaunchError(RuntimeError):
    """The production launcher could not satisfy an authenticated invariant."""


def _canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            dict(document),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def sha256_file(path: object) -> str:
    digest = hashlib.sha256()
    with Path(os.fspath(path)).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _publish_temporary_no_overwrite(temporary: Path, target: Path) -> None:
    if os.path.lexists(target):
        raise FileExistsError(f"production artifact already exists: {target}")
    try:
        if os.name == "nt":
            os.rename(temporary, target)
        else:
            os.link(temporary, target)
            temporary.unlink()
        _fsync_directory(target.parent)
    except FileExistsError:
        raise FileExistsError(f"production artifact already exists: {target}") from None


def _publish_bytes_no_overwrite(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_temporary_no_overwrite(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _shared_initial_document(shared: SharedInitialState) -> dict[str, object]:
    shared.validate()
    return {
        "format": SHARED_INITIAL_STATE_FORMAT,
        "seed": 42,
        "state_sha256": shared.sha256,
        "state": {
            name: tensor.detach().cpu().clone()
            for name, tensor in shared.state.items()
        },
    }


def _load_shared_initial_state(path: Path) -> SharedInitialState:
    try:
        document = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ProductionLaunchError("shared initial state cannot be loaded safely") from error
    expected_keys = {"format", "seed", "state_sha256", "state"}
    if not isinstance(document, Mapping) or set(document) != expected_keys:
        raise ProductionLaunchError("shared initial state fields differ from contract")
    if document["format"] != SHARED_INITIAL_STATE_FORMAT or document["seed"] != 42:
        raise ProductionLaunchError("shared initial state identity differs from contract")
    state = document["state"]
    sha256 = document["state_sha256"]
    if not isinstance(state, Mapping) or not isinstance(sha256, str):
        raise ProductionLaunchError("shared initial state payload is malformed")
    shared = SharedInitialState(state=state, sha256=sha256)
    try:
        return shared.validate()
    except Exception as error:
        raise ProductionLaunchError("shared initial state SHA-256 mismatch") from error


def load_or_create_shared_initial_state(
    path: object,
) -> tuple[SharedInitialState, dict[str, object]]:
    """Load or atomically create the one seed-42 state shared by all runs."""

    target = Path(os.path.abspath(os.fspath(path)))
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shared = create_shared_initial_state()
        document = _shared_initial_document(shared)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary: Optional[Path] = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                torch.save(document, stream)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                _publish_temporary_no_overwrite(temporary, target)
                temporary = None
            except FileExistsError:
                # A concurrent launcher may have won.  Its authenticated state
                # must be loaded and will be compared below; it is never replaced.
                pass
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    shared = _load_shared_initial_state(target)
    return shared, {
        "path": os.fspath(target),
        "bytes": target.stat().st_size,
        "file_sha256": sha256_file(target),
        "state_sha256": shared.sha256,
    }


def _provider_factory(
    snapshot: BootstrapReceiptSnapshot,
    role: str,
    *,
    provider_library: Path,
    device: str,
):
    manifests = snapshot.manifests(role)

    def factory() -> NativeAtomicV3Provider:
        return NativeAtomicV3Provider(
            backend="atomic-nnue-v3",
            role=role,
            manifests=tuple(os.fspath(item.path) for item in manifests),
            manifest_sha256=tuple(item.sha256 for item in manifests),
            manifest_records=tuple(item.records for item in manifests),
            manifest_payloads=tuple(item.payload for item in manifests),
            batch_size=MICROBATCH_SIZE,
            random_fen_skipping=3,
            seed=42,
            native_workers=1,
            device=device,
            library_path=provider_library,
            dataset_source="bootstrap",
            receipt_path=os.fspath(snapshot.receipt_path),
            receipt_sha256=snapshot.receipt_sha256,
            selection_sha256=snapshot.selection_sha256,
            semantic_validation_jsonl_sha256=(
                snapshot.semantic_validation_jsonl_sha256
            ),
            semantic_validation_jsonl_domain_sha256=(
                snapshot.semantic_validation_jsonl_domain_sha256
            ),
            provenance_class=snapshot.provenance_class,
            dataset_publication_ready=False,
            release_candidate_eligible=False,
        )

    return factory


def _write_status(run_directory: Path, document: Mapping[str, object]) -> None:
    payload = {
        "format": STATUS_FORMAT,
        **dict(document),
    }
    _atomic_replace_bytes(run_directory / "status.json", _canonical_json_bytes(payload))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class _ProgressReporter:
    """Atomically persist low-frequency executor progress and an ETA."""

    def __init__(self, run_directory: Path, run_id: str) -> None:
        self.run_directory = run_directory
        self.run_id = run_id
        self.started_at = _utc_now()
        self.started_monotonic = time.monotonic()
        self.start_global_steps: Optional[int] = None
        self.last_document: Optional[dict[str, object]] = None

    def __call__(self, event: Mapping[str, object]) -> None:
        if not isinstance(event, Mapping):
            raise TypeError("progress event must be a mapping")
        now = _utc_now()
        elapsed = max(0.0, time.monotonic() - self.started_monotonic)
        global_steps = event.get("global_steps")
        total_steps = event.get("total_steps")
        if isinstance(global_steps, bool) or not isinstance(global_steps, int):
            raise ProductionLaunchError("progress global_steps is malformed")
        if isinstance(total_steps, bool) or not isinstance(total_steps, int):
            raise ProductionLaunchError("progress total_steps is malformed")
        if global_steps < 0 or total_steps <= 0 or global_steps > total_steps:
            raise ProductionLaunchError("progress step counters are outside bounds")
        if self.start_global_steps is None:
            self.start_global_steps = global_steps
        delta_steps = global_steps - self.start_global_steps
        steps_per_second: Optional[float] = None
        eta_seconds: Optional[float] = None
        estimated_completion: Optional[str] = None
        if delta_steps > 0 and elapsed > 0.0:
            steps_per_second = delta_steps / elapsed
            eta_seconds = (total_steps - global_steps) / steps_per_second
            estimated_completion = _utc_text(now + timedelta(seconds=eta_seconds))
        document: dict[str, object] = {
            **dict(event),
            "status": "training" if event.get("phase") != "completed" else "trained",
            "run_id": self.run_id,
            "training_started": True,
            "invocation_started_at_utc": _utc_text(self.started_at),
            "updated_at_utc": _utc_text(now),
            "elapsed_seconds_this_invocation": elapsed,
            "steps_per_second_this_invocation": steps_per_second,
            "estimated_remaining_seconds": eta_seconds,
            "estimated_completion_utc": estimated_completion,
        }
        self.last_document = document
        _write_status(self.run_directory, document)


def _network_description(
    run_id: str,
    trainer_commit: str,
    snapshot: BootstrapReceiptSnapshot,
    initial_state_sha256: str,
) -> bytes:
    return _canonical_json_bytes(
        {
            "format": "atomic-v3-final-network-v1",
            "run_id": run_id,
            "completed_epoch": EPOCHS,
            "trainer_commit": trainer_commit,
            "bootstrap_receipt_sha256": snapshot.receipt_sha256,
            "initial_state_sha256": initial_state_sha256,
        }
    ).rstrip(b"\n")


def _counters_document(counters: TrainingCounters) -> dict[str, object]:
    return counters.to_document()


def _artifact_paths(output_root: Path, run_id: str) -> tuple[Path, Path, Path]:
    run_directory = output_root / "runs" / run_id
    network_path = output_root / "artifacts" / f"atomic-v3-{run_id}-epoch-37.nnue"
    receipt_path = run_directory / "final-receipt.json"
    return run_directory, network_path, receipt_path


def _verify_completed_receipt(
    receipt_path: Path,
    *,
    run_id: str,
    snapshot: BootstrapReceiptSnapshot,
    shared_artifact: Mapping[str, object],
    provider_sha256: str,
    trainer_commit: str,
) -> dict[str, object]:
    try:
        document = json.loads(receipt_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionLaunchError("existing final receipt is unreadable") from error
    if not isinstance(document, dict) or document.get("format") != FINAL_RECEIPT_FORMAT:
        raise ProductionLaunchError("existing final receipt identity differs")
    expected = {
        "run_id": run_id,
        "bootstrap_receipt_sha256": snapshot.receipt_sha256,
        "selection_sha256": snapshot.selection_sha256,
        "semantic_validation_jsonl_sha256": (
            snapshot.semantic_validation_jsonl_sha256
        ),
        "semantic_validation_jsonl_domain_sha256": (
            snapshot.semantic_validation_jsonl_domain_sha256
        ),
        "trainer_commit": trainer_commit,
        "initial_state_sha256": shared_artifact["state_sha256"],
        "provider_library_sha256": provider_sha256,
    }
    for field, value in expected.items():
        if document.get(field) != value:
            raise ProductionLaunchError(f"existing final receipt {field} differs")
    if document.get("shared_initial_state") != dict(shared_artifact):
        raise ProductionLaunchError("existing final receipt shared state differs")
    network = document.get("network")
    checkpoint = document.get("checkpoint")
    if not isinstance(network, Mapping) or not isinstance(checkpoint, Mapping):
        raise ProductionLaunchError("existing final receipt artifacts are malformed")
    network_value = network.get("path")
    checkpoint_value = checkpoint.get("path")
    if not isinstance(network_value, str) or not isinstance(checkpoint_value, str):
        raise ProductionLaunchError("existing final receipt artifact path is malformed")
    network_path = Path(network_value)
    checkpoint_path = Path(checkpoint_value)
    expected_root = receipt_path.parent.parent.parent
    expected_run_directory, expected_network, _ = _artifact_paths(expected_root, run_id)
    if network_path != expected_network.absolute():
        raise ProductionLaunchError("existing final receipt network path differs")
    if checkpoint_path != (expected_run_directory / "last.ckpt").absolute():
        raise ProductionLaunchError("existing final receipt checkpoint path differs")
    if not network_path.is_file() or not checkpoint_path.is_file():
        raise ProductionLaunchError("existing final receipt artifact is missing")
    if network.get("bytes") != network_path.stat().st_size:
        raise ProductionLaunchError("existing final network size differs")
    if checkpoint.get("bytes") != checkpoint_path.stat().st_size:
        raise ProductionLaunchError("existing final checkpoint size differs")
    if sha256_file(network_path) != network.get("sha256"):
        raise ProductionLaunchError("existing final network SHA-256 mismatch")
    if checkpoint_sha256(checkpoint_path) != checkpoint.get("sha256"):
        raise ProductionLaunchError("existing final checkpoint SHA-256 mismatch")
    with network_path.open("rb") as stream:
        metadata = check_nnue(stream)
    if metadata.sha256.lower() != network.get("sha256"):
        raise ProductionLaunchError("existing final network wire hash differs")
    return document


def cleanup_prepared(prepared: Optional[PreparedProductionRun]) -> None:
    """Release native provider and CUDA allocations before the next run."""

    if prepared is not None:
        close = getattr(prepared.training_provider, "close", None)
        if callable(close):
            close()
        # The preparation object owns the model, optimizer (including Ranger
        # state), scheduler and provider.  Clear those references before the
        # cache flush; merely deleting the caller's local after this function
        # returns would leave the current GPU allocation live during cleanup.
        prepared.training_provider = None  # type: ignore[assignment]
        prepared.validation_provider_factory = None  # type: ignore[assignment]
        prepared.optimizer = None  # type: ignore[assignment]
        prepared.scheduler = None  # type: ignore[assignment]
        prepared.model = None  # type: ignore[assignment]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def execute_one_run(
    *,
    run_id: str,
    snapshot: BootstrapReceiptSnapshot,
    output_root: Path,
    provider_library: Path,
    provider_sha256: str,
    trainer_commit: str,
    shared: SharedInitialState,
    shared_artifact: Mapping[str, object],
    device: str,
    resume: bool,
) -> dict[str, object]:
    """Execute or verify one run and return its machine-readable receipt."""

    config = production_config(run_id)
    run_directory, network_path, receipt_path = _artifact_paths(output_root, run_id)
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "last.ckpt"
    if receipt_path.exists():
        if not resume:
            raise FileExistsError(f"final training receipt already exists: {receipt_path}")
        verified = _verify_completed_receipt(
            receipt_path,
            run_id=run_id,
            snapshot=snapshot,
            shared_artifact=shared_artifact,
            provider_sha256=provider_sha256,
            trainer_commit=trainer_commit,
        )
        _write_status(
            run_directory,
            {"status": "already-complete", "run_id": run_id, "receipt": os.fspath(receipt_path)},
        )
        return verified
    if network_path.exists():
        raise FileExistsError(
            f"orphan final network exists without its receipt: {network_path}"
        )
    if os.path.lexists(checkpoint_path) and not checkpoint_path.is_file():
        raise FileExistsError(f"rolling checkpoint path is not a file: {checkpoint_path}")
    checkpoint_exists = checkpoint_path.is_file()
    if checkpoint_exists and not resume:
        raise FileExistsError(
            f"rolling checkpoint already exists; pass --resume: {checkpoint_path}"
        )
    resume_from_checkpoint = bool(resume and checkpoint_exists)

    _write_status(
        run_directory,
        {
            "status": "preparing",
            "run_id": run_id,
            "training_started": False,
            "resume_requested": bool(resume),
            "resume_from_checkpoint": resume_from_checkpoint,
            "updated_at_utc": _utc_text(_utc_now()),
        },
    )
    training_factory = _provider_factory(
        snapshot, "train", provider_library=provider_library, device=device
    )
    validation_factory = _provider_factory(
        snapshot, "validation", provider_library=provider_library, device=device
    )
    prepared: Optional[PreparedProductionRun] = None
    progress: Optional[_ProgressReporter] = None
    try:
        prepared = prepare_production_run(
            config,
            training_factory,
            validation_factory,
            provider_library_sha256=provider_sha256,
            shared_initial_state=shared,
            device=device,
        )
        binding = CheckpointBinding(
            config=prepared.checkpoint_config_document(),
            dataset=DatasetBinding.from_bootstrap(snapshot),
            commits=CommitBinding(trainer_commit),
        )
        progress = _ProgressReporter(run_directory, run_id)
        _write_status(
            run_directory,
            {
                "status": "training",
                "run_id": run_id,
                "training_started": True,
                "resume_requested": bool(resume),
                "resume_from_checkpoint": resume_from_checkpoint,
                "updated_at_utc": _utc_text(_utc_now()),
            },
        )
        counters = run_production(
            prepared,
            binding,
            os.fspath(run_directory),
            resume=resume_from_checkpoint,
            progress_callback=progress,
        )
        if counters.completed_epochs != EPOCHS:
            raise ProductionLaunchError("production run ended before epoch 37")
        if not checkpoint_path.is_file():
            raise ProductionLaunchError("completed production run has no last.ckpt")
        description = _network_description(
            run_id, trainer_commit, snapshot, shared.sha256
        )
        metadata: WireMetadata = save_nnue(
            network_path, prepared.model, description
        )
        network_sha256 = metadata.sha256.lower()
        if sha256_file(network_path) != network_sha256:
            raise ProductionLaunchError("published final network SHA-256 changed")
        receipt: dict[str, object] = {
            "format": FINAL_RECEIPT_FORMAT,
            "status": "completed",
            "run_id": run_id,
            "completed_epoch": EPOCHS,
            "trainer_commit": trainer_commit,
            "engine_commit": binding.commits.engine_commit,
            "bootstrap_verifier_commit": binding.commits.bootstrap_verifier_commit,
            "bootstrap_receipt_path": os.fspath(snapshot.receipt_path),
            "bootstrap_receipt_sha256": snapshot.receipt_sha256,
            "selection_sha256": snapshot.selection_sha256,
            "semantic_validation_jsonl_path": os.fspath(
                snapshot.semantic_validation_jsonl_path
            ),
            "semantic_validation_jsonl_sha256": (
                snapshot.semantic_validation_jsonl_sha256
            ),
            "semantic_validation_jsonl_domain_sha256": (
                snapshot.semantic_validation_jsonl_domain_sha256
            ),
            "provider_library_path": os.fspath(provider_library),
            "provider_library_sha256": provider_sha256,
            "shared_initial_state": dict(shared_artifact),
            "initial_state_sha256": shared.sha256,
            "config": binding.config_document(),
            "counters": _counters_document(counters),
            "checkpoint": {
                "path": os.fspath(checkpoint_path.absolute()),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": checkpoint_sha256(checkpoint_path),
            },
            "network": {
                "path": os.fspath(network_path.absolute()),
                "bytes": metadata.size,
                "sha256": network_sha256,
                "description_sha256": hashlib.sha256(description).hexdigest(),
            },
            "dataset_publication_ready": False,
            "release_candidate_eligible": False,
        }
        _publish_bytes_no_overwrite(receipt_path, _canonical_json_bytes(receipt))
        _write_status(
            run_directory,
            {
                "status": "completed",
                "run_id": run_id,
                "training_started": True,
                "receipt": os.fspath(receipt_path),
                "network_sha256": network_sha256,
                "updated_at_utc": _utc_text(_utc_now()),
            },
        )
        return receipt
    except BaseException as error:
        _write_status(
            run_directory,
            {
                "status": "failed",
                "run_id": run_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "updated_at_utc": _utc_text(_utc_now()),
                "last_progress": (
                    progress.last_document if progress is not None else None
                ),
            },
        )
        raise
    finally:
        cleanup_prepared(prepared)


def _validate_cuda_device(device: str) -> str:
    try:
        parsed = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ProductionLaunchError("device is not a valid torch device") from error
    if parsed.type != "cuda":
        raise ProductionLaunchError("production training requires a CUDA device")
    if not torch.cuda.is_available():
        raise ProductionLaunchError("CUDA is unavailable")
    index = parsed.index if parsed.index is not None else torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise ProductionLaunchError("requested CUDA device does not exist")
    torch.cuda.set_device(index)
    return f"cuda:{index}"


def execute_runs(
    *,
    receipt_path: object,
    receipt_sha256: str,
    provider_library: object,
    output_root: object,
    trainer_commit: str,
    run_ids: Sequence[str],
    shared_initial_state: Optional[object] = None,
    device: str = "cuda",
    resume: bool = False,
) -> dict[str, object]:
    """Execute the selected runs sequentially with one authenticated init."""

    if not run_ids or any(run_id not in RUN_CONFIGS for run_id in run_ids):
        raise ProductionLaunchError("run selection differs from the four-run contract")
    if len(set(run_ids)) != len(run_ids):
        raise ProductionLaunchError("run selection contains duplicates")
    if (
        not isinstance(trainer_commit, str)
        or len(trainer_commit) != 40
        or any(character not in "0123456789abcdef" for character in trainer_commit)
    ):
        raise ProductionLaunchError("trainer commit must be a lowercase 40-digit SHA-1")
    resolved_device = _validate_cuda_device(device)
    provider_path = Path(os.path.abspath(os.fspath(provider_library)))
    if not provider_path.is_file():
        raise FileNotFoundError(f"provider library is not a file: {provider_path}")
    provider_sha256 = sha256_file(provider_path)
    snapshot = inspect_bootstrap_roles(receipt_path, receipt_sha256)
    root = Path(os.path.abspath(os.fspath(output_root)))
    root.mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    (root / "artifacts").mkdir(exist_ok=True)
    shared_path = (
        Path(os.path.abspath(os.fspath(shared_initial_state)))
        if shared_initial_state is not None
        else root / "shared-initial-state.pt"
    )
    shared, shared_artifact = load_or_create_shared_initial_state(shared_path)
    summary_path = root / "training-summary.json"
    summary: dict[str, object] = {
        "format": SUMMARY_FORMAT,
        "status": "running",
        "receipt_path": os.fspath(snapshot.receipt_path),
        "receipt_sha256": snapshot.receipt_sha256,
        "provider_library_path": os.fspath(provider_path),
        "provider_library_sha256": provider_sha256,
        "shared_initial_state": shared_artifact,
        "device": resolved_device,
        "resume": bool(resume),
        "selected_runs": list(run_ids),
        "results": [],
    }
    _atomic_replace_bytes(summary_path, _canonical_json_bytes(summary))
    try:
        for run_id in run_ids:
            result = execute_one_run(
                run_id=run_id,
                snapshot=snapshot,
                output_root=root,
                provider_library=provider_path,
                provider_sha256=provider_sha256,
                trainer_commit=trainer_commit,
                shared=shared,
                shared_artifact=shared_artifact,
                device=resolved_device,
                resume=resume,
            )
            summary["results"].append(result)
            _atomic_replace_bytes(summary_path, _canonical_json_bytes(summary))
    except BaseException as error:
        summary["status"] = "failed"
        summary["error_type"] = type(error).__name__
        summary["error"] = str(error)
        _atomic_replace_bytes(summary_path, _canonical_json_bytes(summary))
        raise
    summary["status"] = "completed"
    _atomic_replace_bytes(summary_path, _canonical_json_bytes(summary))
    return summary


__all__ = [
    "FINAL_RECEIPT_FORMAT",
    "ProductionLaunchError",
    "SHARED_INITIAL_STATE_FORMAT",
    "STATUS_FORMAT",
    "SUMMARY_FORMAT",
    "cleanup_prepared",
    "execute_one_run",
    "execute_runs",
    "load_or_create_shared_initial_state",
    "sha256_file",
]
