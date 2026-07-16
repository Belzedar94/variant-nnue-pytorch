import argparse
import copy
from dataclasses import replace
import inspect
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn

import atomic_v3.checkpoint as checkpoint_module
import atomic_v3.executor as executor_module
from atomic_v3.bootstrap_dataset import (
    BOOTSTRAP_PROVENANCE_CLASS,
    BOOTSTRAP_RECORDS_PER_MANIFEST,
    BootstrapReceiptSnapshot,
)
from atomic_v3.checkpoint import (
    CheckpointBinding,
    CheckpointError,
    CommitBinding,
    DatasetBinding,
    TrainingCounters,
    checkpoint_document,
    load_last_checkpoint,
    restore_checkpoint,
    save_last_checkpoint,
)
from atomic_v3.dataset import (
    AtomicV3Batch,
    PerspectiveBatch,
    RoleManifest,
    SparseSliceBatch,
    load_canonical_fixture,
)
from atomic_v3.executor import (
    ACCUMULATION_STEPS,
    EFFECTIVE_BATCH_SIZE,
    EPOCHS,
    EPOCH_SIZE,
    FINAL_LEARNING_RATE,
    MAIN_LEARNING_RATE,
    MICROBATCH_SIZE,
    PROGRESS_INTERVAL_STEPS,
    ProviderBatch,
    RANGER_N_SMA_THRESHOLD,
    RUN_CONFIGS,
    SharedInitialState,
    TRAINING_SAMPLES_PER_EPOCH,
    TRAINING_STEPS_PER_EPOCH,
    VALIDATION_SIZE,
    VALIDATION_BATCHES_PER_EPOCH,
    VALIDATION_SAMPLES_PER_EPOCH,
    create_production_optimizer,
    densify_sparse_gradients,
    prepare_production_run,
    production_config,
    reset_validation_provider,
    run_production,
    seed_training,
    train_epoch,
    validate_production_counters,
    validate_production_optimizer,
)
from ranger import Ranger
import train_atomic_v3
import scripts.canary_atomic_v3_production as canary_module


SHA_A = "a" * 64
SHA_B = "b" * 64


class FakeProvider:
    def __init__(self, records, *, cap=10_000, role="train"):
        self.records = tuple((float(x), float(y)) for x, y in records)
        self.cap = cap
        self.role = role
        self.batch_size = 128
        self.random_fen_skipping = 3
        self.seed = 42
        self.native_workers = 1
        self.cyclic = role == "train"
        self.binding_sha256 = ("a" if role == "train" else "b") * 64
        self.cursor = 0
        self.batch_sequence = 0

    def next_batch(self, maximum_samples):
        count = min(maximum_samples, self.cap)
        selected = [
            self.records[(self.cursor + offset) % len(self.records)]
            for offset in range(count)
        ]
        self.cursor += count
        self.batch_sequence += 1
        x = torch.tensor([[item[0]] for item in selected], dtype=torch.float32)
        y = torch.tensor([[item[1]] for item in selected], dtype=torch.float32)
        return ProviderBatch((x, y), count)

    def logical_cursor_state(self):
        return {
            "provider": "atomic-v3-sequential-v1",
            "binding_sha256": self.binding_sha256,
            "epoch": 0,
            "manifest_index": 0,
            "record_index": self.cursor,
            "accepted_samples": self.cursor,
            "next_batch_sequence": self.batch_sequence,
            "eof": False,
        }

    def restore_logical_cursor(self, state):
        if set(state) != {
            "provider",
            "binding_sha256",
            "epoch",
            "manifest_index",
            "record_index",
            "accepted_samples",
            "next_batch_sequence",
            "eof",
        }:
            raise ValueError("bad fake cursor")
        if state["provider"] != "atomic-v3-sequential-v1":
            raise ValueError("bad fake cursor identity")
        self.binding_sha256 = str(state["binding_sha256"])
        self.cursor = int(state["accepted_samples"])
        self.batch_sequence = int(state["next_batch_sequence"])

    def commit(self):
        pass


def _repeat_batch(batch, copies=64):
    def repeat(tensor):
        return tensor.repeat((copies,) + (1,) * (tensor.ndim - 1))

    def sparse(value):
        return SparseSliceBatch(repeat(value.indices), repeat(value.values))

    def perspective(value):
        return PerspectiveBatch(
            repeat(value.own_king_squares),
            sparse(value.hm),
            sparse(value.capture_pair),
            sparse(value.king_blast_ep),
            sparse(value.blast_ring),
        )

    return AtomicV3Batch(
        repeat(batch.side_to_move_white),
        repeat(batch.piece_counts),
        perspective(batch.white),
        perspective(batch.black),
        repeat(batch.outcome),
        repeat(batch.score),
        repeat(batch.bucket_indices),
    )


class CanaryProvider(FakeProvider):
    def __init__(self, payload, *, role, events=None):
        super().__init__([(0, 0)], role=role)
        self.payload = payload
        self.closed = False
        if events is not None:
            events.append(role)

    def next_batch(self, maximum_samples):
        samples = self.payload.batch_size
        assert samples <= maximum_samples
        self.cursor += samples
        self.batch_sequence += 1
        return ProviderBatch(self.payload, samples)

    def close(self):
        self.closed = True


class TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(1, 2)
        self.fc2 = nn.Linear(2, 1)


class TinyAtomicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = TinyNetwork()

    def clip_weights(self):
        pass


def mse_loss(model, payload, lambda_value):
    del lambda_value
    x, y = payload
    return (model(x) - y).square().mean()


def stochastic_mse_loss(model, payload, lambda_value):
    del lambda_value
    scale = (
        1.0
        + 0.01 * float(torch.rand(()))
        + 0.01 * random.random()
        + 0.01 * float(np.random.random())
    )
    return (model(payload[0]) - payload[1]).square().mean() * scale


def _manifest(path, index):
    return RoleManifest(
        chunk_index=index,
        first_record=index * BOOTSTRAP_RECORDS_PER_MANIFEST,
        records=BOOTSTRAP_RECORDS_PER_MANIFEST,
        path=path,
        sha256=SHA_A,
        payload=b"manifest",
    )


def _snapshot(tmp_path):
    train = tuple(
        _manifest((tmp_path / f"train-{index}.json").absolute(), index)
        for index in range(29)
    )
    validation = (_manifest((tmp_path / "validation.json").absolute(), 29),)
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
        train=train,
        validation=validation,
    )


def _binding(tmp_path, config=None):
    return CheckpointBinding(
        config={"test_config": 1} if config is None else config,
        dataset=DatasetBinding.from_bootstrap(_snapshot(tmp_path)),
        commits=CommitBinding("c" * 40),
    )


def _tiny_components():
    model = nn.Linear(1, 1, bias=True)
    optimizer = Ranger(
        model.parameters(),
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1.0e-7,
        alpha=0.5,
        k=6,
        use_gc=False,
        gc_loc=False,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.987)
    return model, optimizer, scheduler


def test_ranger_construction_and_state_restore_are_silent(capsys):
    model = nn.Linear(1, 1, bias=True)
    optimizer = Ranger(
        model.parameters(),
        lr=0.01,
        use_gc=True,
        gc_conv_only=False,
        gc_loc=False,
    )

    # Exercise the exact hook used by Python/Torch object restoration.  An
    # optimizer is a library component, so neither lifecycle event may leak a
    # banner into the launchers' machine-readable streams.
    optimizer.__setstate__(optimizer.__getstate__())

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def _tensor_bytes(model):
    return b"".join(
        tensor.detach().cpu().contiguous().numpy().tobytes()
        for tensor in model.state_dict().values()
    )


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def test_four_run_contract_and_inclusive_lambda_endpoints():
    assert tuple(RUN_CONFIGS) == (
        "lambda-0",
        "lambda-025",
        "lambda-050",
        "lambda-linear-015-050",
    )
    expected = {
        "lambda-0": (0.0, 0.0),
        "lambda-025": (0.25, 0.25),
        "lambda-050": (0.5, 0.5),
        "lambda-linear-015-050": (0.15, 0.5),
    }
    for run_id, endpoints in expected.items():
        config = production_config(run_id)
        assert config.epochs == EPOCHS == 37
        assert config.epoch_size == EPOCH_SIZE == 20_000_000
        assert config.validation_size == VALIDATION_SIZE == 1_000_000
        assert config.effective_batch_size == EFFECTIVE_BATCH_SIZE == 16_384
        assert MICROBATCH_SIZE == 128
        assert TRAINING_STEPS_PER_EPOCH == 1221
        assert TRAINING_SAMPLES_PER_EPOCH == 20_004_864
        assert VALIDATION_BATCHES_PER_EPOCH == 62
        assert VALIDATION_SAMPLES_PER_EPOCH == 1_015_808
        assert config.seed == 42
        assert config.random_skip == 3
        assert config.gpus == config.threads == config.workers == 1
        assert config.precision == "fp32"
        assert config.lambda_schedule.value(0) == endpoints[0]
        assert config.lambda_schedule.value(36) == endpoints[1]
        document = config.to_document()
        assert document["microbatch_size"] == 128
        assert document["accumulation_steps"] == 128
        assert document["progress_interval_steps"] == PROGRESS_INTERVAL_STEPS == 32
        assert document["training_steps_per_epoch"] == 1221
        assert document["training_samples_accepted_per_epoch"] == 20_004_864
        assert document["validation_batches_per_epoch"] == 62
        assert document["validation_samples_accepted_per_epoch"] == 1_015_808
        assert document["optimizer"]["main_learning_rate"] == MAIN_LEARNING_RATE
        assert document["optimizer"]["final_learning_rate"] == FINAL_LEARNING_RATE
        assert document["scheduler"] == {
            "type": "StepLR",
            "step_size_epochs": 1,
            "gamma": 0.987,
        }


def test_microbatches_are_weighted_to_the_exact_effective_batch():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    provider = FakeProvider([(1, 1), (1, 1), (1, 1), (1, 3), (1, 3)])
    metrics = train_epoch(
        model,
        optimizer,
        provider,
        lambda_value=0.0,
        sample_budget=5,
        effective_batch_size=5,
        microbatch_size=3,
        loss_function=mse_loss,
    )
    assert metrics.samples == 5
    assert metrics.steps == 1
    assert metrics.mean_loss == pytest.approx(4.2)
    # -2 * mean([1, 1, 1, 3, 3]) = -3.6, followed by SGD(lr=1).
    assert model.weight.item() == pytest.approx(3.6)
    assert provider.logical_cursor_state()["accepted_samples"] == 5


def test_training_heartbeat_observes_only_committed_step_counters():
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    provider = FakeProvider([(1, 1), (2, 2), (3, 3)], cap=2)
    events = []
    metrics = train_epoch(
        model,
        optimizer,
        provider,
        lambda_value=0.0,
        sample_budget=10,
        effective_batch_size=4,
        microbatch_size=2,
        loss_function=mse_loss,
        progress_callback=lambda steps, samples, cursor: events.append(
            (steps, samples, cursor)
        ),
        progress_interval_steps=2,
    )
    assert metrics.steps == 3
    assert [(steps, samples) for steps, samples, _ in events] == [(2, 8), (3, 10)]
    assert [cursor["accepted_samples"] for _, _, cursor in events] == [8, 10]


class SparseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(8, 1, sparse=True)
        with torch.no_grad():
            self.table.weight.zero_()

    def forward(self, indices):
        return self.table(indices).reshape(-1, 1)


class SparseProvider:
    def __init__(self):
        self.cursor = 0

    def next_batch(self, maximum_samples):
        indices = torch.tensor(
            [(self.cursor + offset) % 4 for offset in range(maximum_samples)],
            dtype=torch.long,
        )
        target = torch.ones(maximum_samples, 1)
        self.cursor += maximum_samples
        return ProviderBatch((indices, target), maximum_samples)

    def logical_cursor_state(self):
        return {"cursor": self.cursor}

    def restore_logical_cursor(self, state):
        self.cursor = int(state["cursor"])

    def commit(self):
        pass


class DenseCheckingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.1)
        self.saw_dense = False

    def step(self, closure=None):
        gradients = [parameter.grad for group in self.param_groups for parameter in group["params"]]
        assert gradients and all(gradient is None or not gradient.is_sparse for gradient in gradients)
        self.saw_dense = True
        return super().step(closure)


def test_sparse_microbatch_gradients_are_coalesced_then_densified_before_step():
    model = SparseModel()
    optimizer = DenseCheckingSGD(model.parameters())
    train_epoch(
        model,
        optimizer,
        SparseProvider(),
        lambda_value=0.0,
        sample_budget=4,
        effective_batch_size=4,
        microbatch_size=2,
        loss_function=lambda model, payload, _: (
            model(payload[0]) - payload[1]
        ).square().mean(),
    )
    assert optimizer.saw_dense is True
    assert model.table.weight.grad is not None
    assert model.table.weight.grad.is_sparse is False
    assert densify_sparse_gradients(model) == 0


def test_validation_provider_resets_to_identical_fixed_cursor_each_epoch():
    created = []

    def factory():
        provider = FakeProvider([(1, 1), (2, 2)], role="validation")
        created.append(provider)
        return provider

    start = factory().logical_cursor_state()
    created.clear()
    first = reset_validation_provider(factory, start, production_config("lambda-0"))
    first.next_batch(3)
    second = reset_validation_provider(factory, start, production_config("lambda-0"))
    assert first is not second
    assert first.logical_cursor_state()["accepted_samples"] == 3
    assert second.logical_cursor_state() == start


def test_checkpoint_resume_is_byte_identical_with_rng_and_logical_cursor(tmp_path):
    binding = _binding(tmp_path)
    records = [(1, 0.25), (2, -0.5), (3, 0.75), (4, 0.0), (5, 1.0)]
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model, optimizer, scheduler = _tiny_components()
    provider = FakeProvider(records, cap=2)

    first = train_epoch(
        model,
        optimizer,
        provider,
        lambda_value=0.25,
        sample_budget=7,
        effective_batch_size=4,
        microbatch_size=3,
        loss_function=stochastic_mse_loss,
    )
    scheduler.step()
    counters = TrainingCounters(
        completed_epochs=1,
        global_steps=first.steps,
        training_samples=first.samples,
        last_epoch_training_samples=first.samples,
        last_train_loss=first.mean_loss,
        last_lambda=0.25,
    )
    save_last_checkpoint(
        tmp_path,
        checkpoint_document(
            model,
            optimizer,
            scheduler,
            provider.logical_cursor_state(),
            counters,
            binding,
        ),
    )

    uninterrupted_second = train_epoch(
        model,
        optimizer,
        provider,
        lambda_value=0.25,
        sample_budget=7,
        effective_batch_size=4,
        microbatch_size=3,
        loss_function=stochastic_mse_loss,
    )
    scheduler.step()
    uninterrupted_bytes = _tensor_bytes(model)
    uninterrupted_optimizer = optimizer.state_dict()
    uninterrupted_scheduler = scheduler.state_dict()
    uninterrupted_cursor = provider.logical_cursor_state()

    # Construction deliberately consumes RNG and starts both model/provider at
    # unrelated state; restore_checkpoint must replace every relevant state.
    torch.rand(17)
    resumed_model, resumed_optimizer, resumed_scheduler = _tiny_components()
    resumed_provider = FakeProvider(records, cap=2)
    resumed_provider.cursor = 999
    document = load_last_checkpoint(tmp_path, binding)
    restored_counters = restore_checkpoint(
        document,
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        resumed_provider,
    )
    assert restored_counters == counters
    resumed_second = train_epoch(
        resumed_model,
        resumed_optimizer,
        resumed_provider,
        lambda_value=0.25,
        sample_budget=7,
        effective_batch_size=4,
        microbatch_size=3,
        loss_function=stochastic_mse_loss,
    )
    resumed_scheduler.step()

    assert resumed_second == uninterrupted_second
    assert _tensor_bytes(resumed_model) == uninterrupted_bytes
    _assert_nested_equal(resumed_optimizer.state_dict(), uninterrupted_optimizer)
    _assert_nested_equal(resumed_scheduler.state_dict(), uninterrupted_scheduler)
    assert resumed_provider.logical_cursor_state() == uninterrupted_cursor


def test_checkpoint_replace_failure_preserves_previous_last(tmp_path, monkeypatch):
    binding = _binding(tmp_path)
    torch.manual_seed(7)
    model, optimizer, scheduler = _tiny_components()
    provider = FakeProvider([(1, 1)])
    original = checkpoint_document(
        model, optimizer, scheduler, provider.logical_cursor_state(), TrainingCounters(), binding
    )
    target = save_last_checkpoint(tmp_path, original)
    before = target.read_bytes()
    replacement = checkpoint_document(
        model,
        optimizer,
        scheduler,
        provider.logical_cursor_state(),
        TrainingCounters(completed_epochs=1),
        binding,
    )

    def fail_replace(source, destination):
        del source, destination
        raise OSError("injected replace crash")

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace crash"):
        save_last_checkpoint(tmp_path, replacement)
    assert target.read_bytes() == before
    assert list(tmp_path.glob(".last.ckpt.*.tmp")) == []


def test_checkpoint_rejects_incompatible_config(tmp_path):
    binding = _binding(tmp_path, {"run": "lambda-0"})
    model, optimizer, scheduler = _tiny_components()
    provider = FakeProvider([(1, 1)])
    save_last_checkpoint(
        tmp_path,
        checkpoint_document(
            model,
            optimizer,
            scheduler,
            provider.logical_cursor_state(),
            TrainingCounters(),
            binding,
        ),
    )
    incompatible = replace(binding, config={"run": "lambda-050"})
    with pytest.raises(CheckpointError, match="config is incompatible"):
        load_last_checkpoint(tmp_path, incompatible)


def test_checkpoint_resume_rejects_different_native_provider_dll(tmp_path, monkeypatch):
    prepared, _ = _prepared_tiny_run(monkeypatch)
    config = prepared.checkpoint_config_document()
    assert config["execution_identity"]["provider_library_sha256"] == SHA_A
    binding = _binding(tmp_path, config)
    save_last_checkpoint(
        tmp_path,
        checkpoint_document(
            prepared.model,
            prepared.optimizer,
            prepared.scheduler,
            prepared.training_provider.logical_cursor_state(),
            TrainingCounters(),
            binding,
        ),
    )
    changed_config = copy.deepcopy(config)
    changed_config["execution_identity"]["provider_library_sha256"] = SHA_B
    incompatible = replace(binding, config=changed_config)
    with pytest.raises(CheckpointError, match="config is incompatible"):
        load_last_checkpoint(tmp_path, incompatible)


def test_dry_run_cli_validates_without_starting_training(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path)
    monkeypatch.setattr(train_atomic_v3, "inspect_bootstrap_roles", lambda *_: snapshot)
    arguments = argparse.Namespace(
        run="lambda-linear-015-050",
        bootstrap_source=[str(snapshot.receipt_path), SHA_A],
        output_dir=str(tmp_path / "checkpoints-on-any-volume"),
        trainer_commit="d" * 40,
        microbatch_size=128,
        dry_run=True,
    )
    document = train_atomic_v3.dry_run_document(arguments)
    assert document["status"] == "validated-dry-run"
    assert document["training_started"] is False
    assert document["microbatch_size"] == 128
    assert document["config"] == production_config(
        "lambda-linear-015-050"
    ).to_document(microbatch_size=128)
    assert document["dataset"]["dataset_publication_ready"] is False
    assert document["dataset"]["release_candidate_eligible"] is False


def test_cli_refuses_non_dry_execution_before_touching_dataset():
    with pytest.raises(SystemExit) as raised:
        train_atomic_v3.main(
            [
                "--run",
                "lambda-0",
                "--bootstrap-source",
                "missing.json",
                SHA_A,
                "--output-dir",
                "out",
                "--trainer-commit",
                "d" * 40,
            ]
        )
    assert raised.value.code == 2


def test_cli_rejects_non_historical_bootstrap_microbatch():
    with pytest.raises(argparse.ArgumentTypeError, match="exactly 128"):
        train_atomic_v3._positive_microbatch("2048")


def test_skip_three_and_full_historical_ranger_are_frozen():
    document = production_config("lambda-025").to_document()
    assert document["random_skip"] == 3
    assert document["validation_random_skip"] == 3
    assert document["dataset"]["training_random_skip"] == 3
    assert document["dataset"]["validation_random_skip"] == 3
    assert document["optimizer"]["n_sma_threshold"] == 5


def test_default_hot_loss_disables_semantic_validation(monkeypatch):
    monkeypatch.setattr(executor_module, "AtomicNNUEV3", TinyAtomicModel)
    calls = []

    def frozen_loss(model, payload, *, lambda_, validate):
        calls.append((model, payload, lambda_, validate))
        return torch.tensor(0.25)

    monkeypatch.setattr(executor_module, "batch_loss", frozen_loss)
    model = TinyAtomicModel()
    payload = load_canonical_fixture().batch("train")
    assert float(executor_module._default_loss(model, payload, 0.5)) == 0.25
    assert calls == [(model, payload, 0.5, False)]
    assert "loss_function" not in inspect.signature(run_production).parameters


def test_seed_training_repeats_python_numpy_torch_and_cuda_if_present():
    def sample():
        values = (random.random(), float(np.random.random()), float(torch.rand(())))
        cuda = float(torch.rand((), device="cuda")) if torch.cuda.is_available() else None
        return values, cuda

    seed_training(42)
    first = sample()
    seed_training(42)
    assert sample() == first


def test_real_gpu_canary_applies_frozen_single_torch_thread():
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(max(2, previous))
        canary_module._set_frozen_threads()
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(previous)


def test_canary_step_wrapper_reapplies_threads_before_every_optimizer_step(
    monkeypatch,
):
    events = []

    class Prepared:
        model = object()
        optimizer = object()
        training_provider = object()

    monkeypatch.setattr(
        canary_module, "_set_frozen_threads", lambda: events.append("threads")
    )

    def train(*args, **kwargs):
        events.append(("train", args, kwargs))
        return object()

    monkeypatch.setattr(canary_module, "train_epoch", train)
    canary_module._run_canary_optimizer_step(Prepared())
    canary_module._run_canary_optimizer_step(Prepared())
    assert [event if isinstance(event, str) else event[0] for event in events] == [
        "threads",
        "train",
        "threads",
        "train",
    ]
    for _, _, kwargs in (event for event in events if not isinstance(event, str)):
        assert kwargs["sample_budget"] == EFFECTIVE_BATCH_SIZE
        assert kwargs["effective_batch_size"] == EFFECTIVE_BATCH_SIZE
        assert kwargs["microbatch_size"] == MICROBATCH_SIZE
        assert kwargs["require_full_microbatches"] is True


def test_production_optimizer_validates_ranger_threshold(monkeypatch):
    monkeypatch.setattr(executor_module, "AtomicNNUEV3", TinyAtomicModel)
    model = TinyAtomicModel()
    optimizer, scheduler = create_production_optimizer(model)
    validate_production_optimizer(
        model, optimizer, scheduler, completed_epochs=0
    )
    assert optimizer.N_sma_threshold == RANGER_N_SMA_THRESHOLD == 5
    optimizer.N_sma_threshold = 4
    with pytest.raises(executor_module.ExecutorError, match="N_sma_threshold"):
        validate_production_optimizer(
            model, optimizer, scheduler, completed_epochs=0
        )


def test_counter_algebra_rejects_epoch_skips():
    config = production_config("lambda-0")
    cursor = CanaryProvider(
        load_canonical_fixture().batch("train"), role="train"
    ).logical_cursor_state()
    validate_production_counters(config, TrainingCounters(), cursor)

    impossible = TrainingCounters(completed_epochs=36, last_lambda=0.0)
    with pytest.raises(executor_module.ExecutorError, match="global_steps"):
        validate_production_counters(config, impossible, cursor)

    one_epoch = TrainingCounters(
        completed_epochs=1,
        global_steps=TRAINING_STEPS_PER_EPOCH,
        training_samples=TRAINING_SAMPLES_PER_EPOCH,
        validation_samples=VALIDATION_SAMPLES_PER_EPOCH,
        validation_batches=VALIDATION_BATCHES_PER_EPOCH,
        last_epoch_training_samples=TRAINING_SAMPLES_PER_EPOCH,
        last_epoch_validation_samples=VALIDATION_SAMPLES_PER_EPOCH,
        last_epoch_validation_batches=VALIDATION_BATCHES_PER_EPOCH,
        last_train_loss=0.2,
        last_validation_loss=0.3,
        last_lambda=0.0,
    )
    cursor = dict(cursor)
    cursor["accepted_samples"] = TRAINING_SAMPLES_PER_EPOCH
    cursor["next_batch_sequence"] = TRAINING_STEPS_PER_EPOCH * ACCUMULATION_STEPS
    validate_production_counters(config, one_epoch, cursor)


def test_restore_rejects_provider_that_ignores_checkpoint_cursor(tmp_path):
    class BrokenRestoreProvider(FakeProvider):
        def restore_logical_cursor(self, state):
            del state

    binding = _binding(tmp_path)
    model, optimizer, scheduler = _tiny_components()
    source = FakeProvider([(1, 1)])
    source.next_batch(1)
    document = checkpoint_document(
        model,
        optimizer,
        scheduler,
        source.logical_cursor_state(),
        TrainingCounters(),
        binding,
    )
    broken = BrokenRestoreProvider([(1, 1)])
    with pytest.raises(CheckpointError, match="restore the checkpoint cursor exactly"):
        restore_checkpoint(document, model, optimizer, scheduler, broken)


def _prepared_tiny_run(monkeypatch, events=None):
    fixture = _repeat_batch(load_canonical_fixture().batch("train"))

    class LoggedTinyAtomicModel(TinyAtomicModel):
        def __init__(self):
            if events is not None:
                events.append("model")
            super().__init__()

    monkeypatch.setattr(executor_module, "AtomicNNUEV3", LoggedTinyAtomicModel)
    real_seed = executor_module.seed_training

    def logged_seed(seed):
        if events is not None:
            events.append("seed")
        real_seed(seed)

    monkeypatch.setattr(executor_module, "seed_training", logged_seed)
    logged_seed(42)
    shared = SharedInitialState.from_model(LoggedTinyAtomicModel())
    if events is not None:
        events.clear()

    def training_factory():
        return CanaryProvider(fixture, role="train", events=events)

    def validation_factory():
        return CanaryProvider(fixture, role="validation", events=events)

    prepared = prepare_production_run(
        production_config("lambda-0"),
        training_factory,
        validation_factory,
        provider_library_sha256=SHA_A,
        shared_initial_state=shared,
        device="cpu",
    )
    return prepared, shared


def test_preparation_seeds_before_model_and_providers_and_loads_shared_init(monkeypatch):
    events = []
    first, shared = _prepared_tiny_run(monkeypatch, events)
    assert events[:4] == ["seed", "model", "train", "validation"]
    assert first.initial_state_sha256 == shared.sha256
    assert first.training_start_cursor["accepted_samples"] == 0
    assert first.validation_start_cursor["accepted_samples"] == 0
    identity = first.checkpoint_config_document()["execution_identity"]
    assert identity["initial_state_sha256"] == shared.sha256

    events.clear()
    second = prepare_production_run(
        production_config("lambda-0"),
        lambda: CanaryProvider(
            _repeat_batch(load_canonical_fixture().batch("train")),
            role="train",
            events=events,
        ),
        lambda: CanaryProvider(
            _repeat_batch(load_canonical_fixture().batch("train")),
            role="validation",
            events=events,
        ),
        provider_library_sha256=SHA_A,
        shared_initial_state=shared,
        device="cpu",
    )
    assert events[:4] == ["seed", "model", "train", "validation"]
    assert second.initial_state_sha256 == first.initial_state_sha256


def test_run_production_end_to_end_validates_complete_resume_without_gpu_full_run(
    tmp_path, monkeypatch
):
    prepared, _ = _prepared_tiny_run(monkeypatch)
    monkeypatch.setattr(executor_module, "require_production_model", lambda model: None)
    source_model = type(prepared.model)()
    source_model.load_state_dict(prepared.model.state_dict(), strict=True)
    source_optimizer, source_scheduler = create_production_optimizer(source_model)
    for _ in range(EPOCHS):
        source_optimizer.step()
        source_scheduler.step()
    counters = TrainingCounters(
        completed_epochs=EPOCHS,
        global_steps=EPOCHS * TRAINING_STEPS_PER_EPOCH,
        training_samples=EPOCHS * TRAINING_SAMPLES_PER_EPOCH,
        validation_samples=EPOCHS * VALIDATION_SAMPLES_PER_EPOCH,
        validation_batches=EPOCHS * VALIDATION_BATCHES_PER_EPOCH,
        last_epoch_training_samples=TRAINING_SAMPLES_PER_EPOCH,
        last_epoch_validation_samples=VALIDATION_SAMPLES_PER_EPOCH,
        last_epoch_validation_batches=VALIDATION_BATCHES_PER_EPOCH,
        last_train_loss=0.2,
        last_validation_loss=0.3,
        last_lambda=0.0,
    )
    cursor = dict(prepared.training_start_cursor)
    cursor["accepted_samples"] = counters.training_samples
    cursor["record_index"] = counters.training_samples
    cursor["next_batch_sequence"] = (
        EPOCHS * TRAINING_STEPS_PER_EPOCH * ACCUMULATION_STEPS
    )
    binding = _binding(tmp_path, prepared.checkpoint_config_document())
    save_last_checkpoint(
        tmp_path,
        checkpoint_document(
            source_model,
            source_optimizer,
            source_scheduler,
            cursor,
            counters,
            binding,
        ),
    )
    progress = []
    restored = run_production(
        prepared,
        binding,
        str(tmp_path),
        resume=True,
        progress_callback=progress.append,
    )
    assert restored == counters
    assert prepared.training_provider.logical_cursor_state() == cursor
    assert [event["event"] for event in progress] == ["run-ready", "run-complete"]
    assert all(event["global_steps"] == counters.global_steps for event in progress)
