import argparse
from dataclasses import replace
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn

import atomic_v3.checkpoint as checkpoint_module
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
from atomic_v3.dataset import RoleManifest
from atomic_v3.executor import (
    EFFECTIVE_BATCH_SIZE,
    EPOCHS,
    EPOCH_SIZE,
    FINAL_LEARNING_RATE,
    MAIN_LEARNING_RATE,
    MICROBATCH_SIZE,
    ProviderBatch,
    RUN_CONFIGS,
    TRAINING_SAMPLES_PER_EPOCH,
    TRAINING_STEPS_PER_EPOCH,
    VALIDATION_SIZE,
    VALIDATION_BATCHES_PER_EPOCH,
    VALIDATION_SAMPLES_PER_EPOCH,
    densify_sparse_gradients,
    production_config,
    reset_validation_provider,
    train_epoch,
)
from ranger import Ranger
import train_atomic_v3


SHA_A = "a" * 64
SHA_B = "b" * 64


class FakeProvider:
    def __init__(self, records, *, cap=10_000):
        self.records = tuple((float(x), float(y)) for x, y in records)
        self.cap = cap
        self.cursor = 0

    def next_batch(self, maximum_samples):
        count = min(maximum_samples, self.cap)
        selected = [
            self.records[(self.cursor + offset) % len(self.records)]
            for offset in range(count)
        ]
        self.cursor += count
        x = torch.tensor([[item[0]] for item in selected], dtype=torch.float32)
        y = torch.tensor([[item[1]] for item in selected], dtype=torch.float32)
        return ProviderBatch((x, y), count)

    def logical_cursor_state(self):
        return {"cursor": self.cursor, "seed": 42, "cyclic": True}

    def restore_logical_cursor(self, state):
        if set(state) != {"cursor", "seed", "cyclic"}:
            raise ValueError("bad fake cursor")
        if state["seed"] != 42 or state["cyclic"] is not True:
            raise ValueError("bad fake cursor identity")
        self.cursor = int(state["cursor"])


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
    assert provider.logical_cursor_state()["cursor"] == 5


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
        provider = FakeProvider([(1, 1), (2, 2)])
        provider.cursor = 99 + len(created)
        created.append(provider)
        return provider

    start = {"cursor": 0, "seed": 42, "cyclic": True}
    first = reset_validation_provider(factory, start)
    first.next_batch(3)
    second = reset_validation_provider(factory, start)
    assert first is not second
    assert first.logical_cursor_state()["cursor"] == 3
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
