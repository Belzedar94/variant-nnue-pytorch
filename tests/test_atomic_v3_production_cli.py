import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import atomic_v3.executor as executor_module
import atomic_v3.production as production
from atomic_v3.bootstrap_dataset import (
    BOOTSTRAP_PROVENANCE_CLASS,
    BOOTSTRAP_RECORDS_PER_MANIFEST,
    BootstrapReceiptSnapshot,
)
from atomic_v3.checkpoint import TrainingCounters
from atomic_v3.dataset import RoleManifest
from atomic_v3.executor import SharedInitialState
from atomic_v3.serialization import WireMetadata
import train_atomic_v3


SHA_A = "a" * 64
SHA_B = "b" * 64
TRAINER_COMMIT = "c" * 40


def _shared(value=1.0):
    state = {"weight": torch.tensor([value], dtype=torch.float32)}
    return SharedInitialState(state, executor_module._state_sha256(state))


def _snapshot(tmp_path):
    def manifest(index):
        return RoleManifest(
            chunk_index=index,
            first_record=index * BOOTSTRAP_RECORDS_PER_MANIFEST,
            records=BOOTSTRAP_RECORDS_PER_MANIFEST,
            path=(tmp_path / f"chunk-{index}.manifest.json").absolute(),
            sha256=SHA_A,
            payload=b"manifest",
        )

    return BootstrapReceiptSnapshot(
        receipt_path=(tmp_path / "receipt.json").absolute(),
        receipt_sha256=SHA_A,
        receipt_payload=b"receipt",
        semantic_validation_jsonl_path=(tmp_path / "semantic.jsonl").absolute(),
        semantic_validation_jsonl_sha256=SHA_B,
        semantic_validation_jsonl_domain_sha256=SHA_A,
        semantic_validation_jsonl_payload=b"semantic",
        selection_sha256=SHA_B,
        provenance_class=BOOTSTRAP_PROVENANCE_CLASS,
        dataset_publication_ready=False,
        release_candidate_eligible=False,
        train=tuple(manifest(index) for index in range(29)),
        validation=(manifest(29),),
    )


def _completed_checkpoint_state(run_id, completed_epochs=executor_module.EPOCHS):
    config = production.production_config(run_id)
    counters = TrainingCounters(
        completed_epochs=completed_epochs,
        global_steps=completed_epochs * executor_module.TRAINING_STEPS_PER_EPOCH,
        training_samples=(
            completed_epochs * executor_module.TRAINING_SAMPLES_PER_EPOCH
        ),
        validation_samples=(
            completed_epochs * executor_module.VALIDATION_SAMPLES_PER_EPOCH
        ),
        validation_batches=(
            completed_epochs * executor_module.VALIDATION_BATCHES_PER_EPOCH
        ),
        last_epoch_training_samples=(
            executor_module.TRAINING_SAMPLES_PER_EPOCH if completed_epochs else 0
        ),
        last_epoch_validation_samples=(
            executor_module.VALIDATION_SAMPLES_PER_EPOCH if completed_epochs else 0
        ),
        last_epoch_validation_batches=(
            executor_module.VALIDATION_BATCHES_PER_EPOCH if completed_epochs else 0
        ),
        last_train_loss=0.125 if completed_epochs else None,
        last_validation_loss=0.25 if completed_epochs else None,
        last_lambda=(
            config.lambda_schedule.value(completed_epochs - 1)
            if completed_epochs
            else None
        ),
    )
    cursor = {
        "accepted_samples": counters.training_samples,
        "next_batch_sequence": (
            completed_epochs
            * executor_module.TRAINING_STEPS_PER_EPOCH
            * executor_module.ACCUMULATION_STEPS
        ),
    }
    return counters, cursor


def test_shared_initial_state_is_atomic_persistent_and_hash_checked(tmp_path, monkeypatch):
    calls = []

    def create():
        calls.append("create")
        return _shared()

    monkeypatch.setattr(production, "create_shared_initial_state", create)
    path = tmp_path / "shared" / "initial.pt"
    first, first_artifact = production.load_or_create_shared_initial_state(path)
    second, second_artifact = production.load_or_create_shared_initial_state(path)
    assert calls == ["create"]
    assert first.sha256 == second.sha256 == first_artifact["state_sha256"]
    assert first_artifact == second_artifact
    assert first_artifact["file_sha256"] == production.sha256_file(path)
    assert not list(path.parent.glob(".initial.pt.*.tmp"))

    document = torch.load(path, map_location="cpu", weights_only=True)
    document["state_sha256"] = "0" * 64
    torch.save(document, path)
    with pytest.raises(production.ProductionLaunchError, match="SHA-256"):
        production.load_or_create_shared_initial_state(path)


class _CloseProbe:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_one_run_calls_only_prepare_run_and_final_serializer(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path)
    provider_library = tmp_path / "provider.dll"
    provider_library.write_bytes(b"provider")
    shared = _shared()
    shared_artifact = {
        "path": str(tmp_path / "shared.pt"),
        "bytes": 1,
        "file_sha256": SHA_A,
        "state_sha256": shared.sha256,
    }
    events = []
    provider = _CloseProbe()
    completed_counters, completed_cursor = _completed_checkpoint_state("lambda-0")

    class Prepared:
        model = object()
        training_provider = provider

        def checkpoint_config_document(self):
            return {"normative": True}

    def prepare(*args, **kwargs):
        events.append(("prepare", args, kwargs))
        return Prepared()

    def run(prepared, binding, output_directory, *, resume, progress_callback):
        events.append(
            ("run", prepared, binding, output_directory, resume, progress_callback)
        )
        Path(output_directory, "last.ckpt").write_bytes(b"checkpoint")
        return completed_counters

    def serialize(path, model, description):
        events.append(("serialize", path, model, description))
        payload = b"strict-v3-final"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(payload)
        return WireMetadata(
            description=description,
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest().upper(),
        )

    monkeypatch.setattr(production, "prepare_production_run", prepare)
    monkeypatch.setattr(production, "run_production", run)
    monkeypatch.setattr(production, "save_nnue", serialize)
    result = production.execute_one_run(
        run_id="lambda-0",
        snapshot=snapshot,
        output_root=tmp_path / "output",
        provider_library=provider_library,
        provider_sha256=production.sha256_file(provider_library),
        trainer_commit=TRAINER_COMMIT,
        shared=shared,
        shared_artifact=shared_artifact,
        device="cuda:0",
        resume=False,
    )
    assert [event[0] for event in events] == ["prepare", "run", "serialize"]
    assert events[0][2]["provider_library_sha256"] == production.sha256_file(
        provider_library
    )
    assert result["completed_epoch"] == 37
    assert result["network"]["sha256"] == hashlib.sha256(
        b"strict-v3-final"
    ).hexdigest()
    assert result["initial_state_sha256"] == shared.sha256
    assert provider.closed
    receipt = json.loads(
        (tmp_path / "output" / "runs" / "lambda-0" / "final-receipt.json").read_text()
    )
    assert receipt == result
    status = json.loads(
        (tmp_path / "output" / "runs" / "lambda-0" / "status.json").read_text()
    )
    assert status["status"] == "completed"

    events.clear()
    with pytest.raises(FileExistsError, match="final training receipt"):
        production.execute_one_run(
            run_id="lambda-0",
            snapshot=snapshot,
            output_root=tmp_path / "output",
            provider_library=provider_library,
            provider_sha256=production.sha256_file(provider_library),
            trainer_commit=TRAINER_COMMIT,
            shared=shared,
            shared_artifact=shared_artifact,
            device="cuda:0",
            resume=False,
        )
    assert events == []

    def check(stream):
        payload = stream.read()
        return WireMetadata(
            description=b"checked",
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest().upper(),
        )

    monkeypatch.setattr(production, "check_nnue", check)
    authenticated = []

    def load_checkpoint(output_directory, binding):
        authenticated.append((Path(output_directory), binding))
        return {
            "counters": completed_counters.to_document(),
            "logical_cursor": completed_cursor,
        }

    monkeypatch.setattr(production, "load_last_checkpoint", load_checkpoint)
    events.clear()
    resumed = production.execute_one_run(
        run_id="lambda-0",
        snapshot=snapshot,
        output_root=tmp_path / "output",
        provider_library=provider_library,
        provider_sha256=production.sha256_file(provider_library),
        trainer_commit=TRAINER_COMMIT,
        shared=shared,
        shared_artifact=shared_artifact,
        device="cuda:0",
        resume=True,
    )
    assert resumed == result
    assert [event[0] for event in events] == ["prepare"]
    assert len(authenticated) == 1
    assert authenticated[0][0] == tmp_path / "output" / "runs" / "lambda-0"
    assert authenticated[0][1].config_document() == {"normative": True}
    assert authenticated[0][1].commits.trainer_commit == TRAINER_COMMIT
    assert not any(event[0] in {"run", "serialize"} for event in events)


@pytest.mark.parametrize(
    "failure",
    ("incompatible-checkpoint", "incomplete-checkpoint", "receipt-counters"),
)
def test_completed_checkpoint_authentication_fails_closed(
    tmp_path, monkeypatch, failure
):
    snapshot = _snapshot(tmp_path)
    binding = production.CheckpointBinding(
        config={"normative": True},
        dataset=production.DatasetBinding.from_bootstrap(snapshot),
        commits=production.CommitBinding(TRAINER_COMMIT),
    )
    completed_counters, completed_cursor = _completed_checkpoint_state("lambda-0")
    receipt = {
        "status": "completed",
        "completed_epoch": executor_module.EPOCHS,
        "engine_commit": binding.commits.engine_commit,
        "bootstrap_verifier_commit": binding.commits.bootstrap_verifier_commit,
        "config": binding.config_document(),
        "counters": completed_counters.to_document(),
    }

    if failure == "incompatible-checkpoint":

        def load_checkpoint(*args, **kwargs):
            raise ValueError("checkpoint config is incompatible with this run")

    else:
        checkpoint_counters = completed_counters
        checkpoint_cursor = completed_cursor
        if failure == "incomplete-checkpoint":
            checkpoint_counters, checkpoint_cursor = _completed_checkpoint_state(
                "lambda-0", executor_module.EPOCHS - 1
            )
        elif failure == "receipt-counters":
            receipt_counters = dict(completed_counters.to_document())
            receipt_counters["global_steps"] -= 1
            receipt["counters"] = receipt_counters

        def load_checkpoint(*args, **kwargs):
            return {
                "counters": checkpoint_counters.to_document(),
                "logical_cursor": checkpoint_cursor,
            }

    monkeypatch.setattr(production, "load_last_checkpoint", load_checkpoint)
    with pytest.raises(production.ProductionLaunchError):
        production._authenticate_completed_checkpoint(
            tmp_path,
            binding=binding,
            config=production.production_config("lambda-0"),
            receipt=receipt,
        )


def test_failed_run_closes_provider_and_writes_machine_status(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path)
    provider_library = tmp_path / "provider.dll"
    provider_library.write_bytes(b"provider")
    shared = _shared()
    provider = _CloseProbe()

    class Prepared:
        model = object()
        training_provider = provider

        def checkpoint_config_document(self):
            return {"normative": True}

    monkeypatch.setattr(production, "prepare_production_run", lambda *a, **k: Prepared())

    def fail(*args, **kwargs):
        raise RuntimeError("injected training failure")

    monkeypatch.setattr(production, "run_production", fail)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="injected"):
        production.execute_one_run(
            run_id="lambda-025",
            snapshot=snapshot,
            output_root=output,
            provider_library=provider_library,
            provider_sha256=production.sha256_file(provider_library),
            trainer_commit=TRAINER_COMMIT,
            shared=shared,
            shared_artifact={
                "path": str(tmp_path / "shared.pt"),
                "bytes": 1,
                "file_sha256": SHA_A,
                "state_sha256": shared.sha256,
            },
            device="cuda:0",
            resume=False,
        )
    assert provider.closed
    status = json.loads(
        (output / "runs" / "lambda-025" / "status.json").read_text()
    )
    assert status["status"] == "failed"
    assert status["error_type"] == "RuntimeError"


def test_existing_checkpoint_without_resume_fails_before_prepare_and_preserves_bytes(
    tmp_path, monkeypatch
):
    snapshot = _snapshot(tmp_path)
    provider_library = tmp_path / "provider.dll"
    provider_library.write_bytes(b"provider")
    output = tmp_path / "output"
    checkpoint = output / "runs" / "lambda-0" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"advanced-checkpoint-must-survive")
    calls = []
    monkeypatch.setattr(
        production,
        "prepare_production_run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    with pytest.raises(FileExistsError, match="pass --resume"):
        production.execute_one_run(
            run_id="lambda-0",
            snapshot=snapshot,
            output_root=output,
            provider_library=provider_library,
            provider_sha256=production.sha256_file(provider_library),
            trainer_commit=TRAINER_COMMIT,
            shared=_shared(),
            shared_artifact={
                "path": str(tmp_path / "shared.pt"),
                "bytes": 1,
                "file_sha256": SHA_A,
                "state_sha256": _shared().sha256,
            },
            device="cuda:0",
            resume=False,
        )
    assert calls == []
    assert checkpoint.read_bytes() == b"advanced-checkpoint-must-survive"


@pytest.mark.parametrize(
    ("checkpoint_exists", "expected_resume"), ((False, False), (True, True))
)
def test_resume_starts_fresh_only_when_checkpoint_is_absent(
    tmp_path, monkeypatch, checkpoint_exists, expected_resume
):
    snapshot = _snapshot(tmp_path)
    provider_library = tmp_path / "provider.dll"
    provider_library.write_bytes(b"provider")
    output = tmp_path / "output"
    run_directory = output / "runs" / "lambda-025"
    run_directory.mkdir(parents=True)
    if checkpoint_exists:
        (run_directory / "last.ckpt").write_bytes(b"checkpoint")
    shared = _shared()
    provider = _CloseProbe()

    class Prepared:
        model = object()
        training_provider = provider
        optimizer = object()
        scheduler = object()
        validation_provider_factory = object()

        def checkpoint_config_document(self):
            return {"normative": True}

    observed = []

    def run(prepared, binding, output_directory, *, resume, progress_callback):
        del prepared, binding, progress_callback
        observed.append(resume)
        Path(output_directory, "last.ckpt").write_bytes(b"completed-checkpoint")
        return TrainingCounters(completed_epochs=37)

    def serialize(path, model, description):
        del model
        payload = b"strict-v3-final"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(payload)
        return WireMetadata(
            description=description,
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest().upper(),
        )

    monkeypatch.setattr(production, "prepare_production_run", lambda *a, **k: Prepared())
    monkeypatch.setattr(production, "run_production", run)
    monkeypatch.setattr(production, "save_nnue", serialize)
    result = production.execute_one_run(
        run_id="lambda-025",
        snapshot=snapshot,
        output_root=output,
        provider_library=provider_library,
        provider_sha256=production.sha256_file(provider_library),
        trainer_commit=TRAINER_COMMIT,
        shared=shared,
        shared_artifact={
            "path": str(tmp_path / "shared.pt"),
            "bytes": 1,
            "file_sha256": SHA_A,
            "state_sha256": shared.sha256,
        },
        device="cuda:0",
        resume=True,
    )
    assert result["completed_epoch"] == 37
    assert observed == [expected_resume]
    assert provider.closed


def test_progress_reporter_persists_atomic_eta_and_checkpoint_payload(tmp_path):
    reporter = production._ProgressReporter(tmp_path, "lambda-0")
    reporter(
        {
            "event": "run-ready",
            "phase": "training",
            "global_steps": 10,
            "total_steps": 100,
        }
    )
    reporter.started_monotonic -= 32.0
    reporter(
        {
            "event": "epoch-checkpoint",
            "phase": "checkpointed",
            "completed_epochs": 1,
            "global_steps": 42,
            "total_steps": 100,
            "checkpoint": {"path": "D:/last.ckpt", "bytes": 123},
        }
    )
    status = json.loads((tmp_path / "status.json").read_text(encoding="ascii"))
    assert status["format"] == production.STATUS_FORMAT
    assert status["event"] == "epoch-checkpoint"
    assert status["checkpoint"]["bytes"] == 123
    assert status["steps_per_second_this_invocation"] > 0
    assert status["estimated_remaining_seconds"] > 0
    assert status["estimated_completion_utc"].endswith("Z")
    assert status["updated_at_utc"].endswith("Z")


def test_provider_factory_passes_the_receipted_sha_to_native_binding(
    tmp_path, monkeypatch
):
    snapshot = _snapshot(tmp_path)
    provider_library = tmp_path / "provider.dll"
    calls = []
    sentinel = object()

    def create(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(production, "NativeAtomicV3Provider", create)
    factory = production._provider_factory(
        snapshot,
        "train",
        provider_library=provider_library,
        provider_sha256=SHA_A,
        device="cuda:0",
    )
    assert factory() is sentinel
    assert calls[0]["library_path"] == provider_library
    assert calls[0]["library_sha256"] == SHA_A


def test_execute_all_runs_reuses_one_shared_state_and_updates_summary(
    tmp_path, monkeypatch
):
    snapshot = _snapshot(tmp_path)
    provider_library = tmp_path / "provider.dll"
    provider_library.write_bytes(b"provider")
    shared = _shared()
    shared_artifact = {
        "path": str(tmp_path / "shared.pt"),
        "bytes": 1,
        "file_sha256": SHA_A,
        "state_sha256": shared.sha256,
    }
    monkeypatch.setattr(production, "_validate_cuda_device", lambda value: "cuda:0")
    monkeypatch.setattr(production, "inspect_bootstrap_roles", lambda *args: snapshot)
    monkeypatch.setattr(
        production,
        "load_or_create_shared_initial_state",
        lambda path: (shared, shared_artifact),
    )
    calls = []

    def one(**kwargs):
        calls.append(kwargs)
        return {"format": production.FINAL_RECEIPT_FORMAT, "run_id": kwargs["run_id"]}

    monkeypatch.setattr(production, "execute_one_run", one)
    selected = tuple(executor_module.RUN_CONFIGS)
    output = tmp_path / "output"
    summary = production.execute_runs(
        receipt_path=snapshot.receipt_path,
        receipt_sha256=snapshot.receipt_sha256,
        provider_library=provider_library,
        output_root=output,
        trainer_commit=TRAINER_COMMIT,
        run_ids=selected,
        device="cuda",
    )
    assert [call["run_id"] for call in calls] == list(selected)
    assert all(call["shared"] is shared for call in calls)
    assert summary["status"] == "completed"
    assert json.loads((output / "training-summary.json").read_text()) == summary


def test_cli_routes_non_dry_execution_to_normative_orchestrator(tmp_path, monkeypatch):
    provider = tmp_path / "provider.dll"
    provider.write_bytes(b"provider")
    calls = []

    def execute(**kwargs):
        calls.append(kwargs)
        return {"status": "completed", "results": []}

    monkeypatch.setattr(train_atomic_v3, "execute_runs", execute)
    assert train_atomic_v3.main(
        [
            "--all-runs",
            "--bootstrap-source",
            "receipt.json",
            SHA_A,
            "--provider-library",
            str(provider),
            "--output-root",
            str(tmp_path / "output"),
            "--trainer-commit",
            TRAINER_COMMIT,
            "--shared-initial-state",
            str(tmp_path / "shared.pt"),
            "--device",
            "cuda:0",
            "--resume",
        ]
    ) == 0
    assert calls[0]["run_ids"] == tuple(executor_module.RUN_CONFIGS)
    assert calls[0]["resume"] is True
    assert calls[0]["device"] == "cuda:0"
