import json

import pytest
import torch

from atomic_v2.dataset import (
    ATOMIC_BIN_V2_MANIFEST_SCHEMA_SHA256,
    ATOMIC_BIN_V2_SCHEMA_SHA256,
    DatasetContractError,
    create_provider,
    validate_batch,
    validate_loader_capabilities,
    validate_train_validation_manifests,
    validate_v2_manifest_entrypoint,
)


def loader_capabilities():
    return {
        "capability_version": 2,
        "formats": {
            "legacy-atomic-v1": {"read": True},
            "atomic-bin-v2": {
                "read": True,
                "write": False,
                "entrypoint": "manifest",
                "header_size": 96,
                "record_size": 64,
                "schema_sha256": ATOMIC_BIN_V2_SCHEMA_SHA256,
                "manifest_schema_sha256": ATOMIC_BIN_V2_MANIFEST_SCHEMA_SHA256,
            },
        },
    }


def batch():
    return (
        torch.ones((2, 1)),
        torch.zeros((2, 1)),
        torch.tensor([[0, -1], [1, -1]], dtype=torch.int32),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
        torch.tensor([[2, -1], [3, -1]], dtype=torch.int32),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
        torch.tensor([[1.0], [0.0]]),
        torch.tensor([[12.0], [-8.0]]),
        torch.tensor([7, 7], dtype=torch.long),
        torch.tensor([7, 7], dtype=torch.long),
    )


def write_v2_manifest(path):
    path.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "manifest_schema_sha256": ATOMIC_BIN_V2_MANIFEST_SCHEMA_SHA256,
                "data_schema_sha256": ATOMIC_BIN_V2_SCHEMA_SHA256,
                "format": "atomic-bin-v2",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_plural_native_handshake_is_required_for_atomic_bin_v2():
    validate_loader_capabilities(loader_capabilities())

    wrong = loader_capabilities()
    wrong["formats"]["atomic-bin-v2"]["schema_sha256"] = "0" * 64
    with pytest.raises(DatasetContractError, match="capability mismatch"):
        validate_loader_capabilities(wrong)


def test_existing_ten_tensor_loader_abi_is_validated():
    values = batch()
    assert validate_batch(values) == values

    with pytest.raises(DatasetContractError, match="exactly 10"):
        validate_batch(values[:-1])
    wrong = list(values)
    wrong[2] = wrong[2].to(torch.int64)
    with pytest.raises(DatasetContractError, match="int32"):
        validate_batch(tuple(wrong))
    wrong = list(values)
    wrong[7] = wrong[7].to(torch.float64)
    with pytest.raises(DatasetContractError, match="float32"):
        validate_batch(tuple(wrong))


def test_v2_entrypoint_rejects_legacy_mixed_and_spoofed_manifests(tmp_path, monkeypatch):
    training = write_v2_manifest(tmp_path / "train.atbin.manifest.json")
    validation = write_v2_manifest(tmp_path / "validation.atbin.manifest.json")
    legacy = tmp_path / "legacy.bin"
    legacy.write_bytes(b"legacy")
    spoofed = tmp_path / "spoofed.atbin.manifest.json"
    spoofed.write_text(
        '{"manifest_version":1,"manifest_schema_sha256":"'
        + ATOMIC_BIN_V2_MANIFEST_SCHEMA_SHA256
        + '","data_schema_sha256":"'
        + ATOMIC_BIN_V2_SCHEMA_SHA256
        + '","format":"legacy-atomic-v1"}',
        encoding="utf-8",
    )

    assert validate_v2_manifest_entrypoint(training) == training
    with pytest.raises(DatasetContractError, match="entrypoint"):
        validate_v2_manifest_entrypoint(legacy)
    with pytest.raises(DatasetContractError, match="format mismatch"):
        validate_v2_manifest_entrypoint(spoofed)

    calls = []
    fake_loader = type(
        "FakeLoader",
        (),
        {
            "atomic_training_data_schemas": staticmethod(loader_capabilities),
            "validate_training_validation_data_paths": staticmethod(
                lambda left, right: calls.append((left, right))
            ),
        },
    )
    monkeypatch.setitem(__import__("sys").modules, "nnue_dataset", fake_loader)
    validate_train_validation_manifests(training, validation)
    assert calls == [(str(training), str(validation))]

    calls.clear()
    with pytest.raises(DatasetContractError, match="entrypoint"):
        validate_train_validation_manifests(training, legacy)
    assert calls == []


def test_provider_rejects_legacy_before_dual_format_native_loader(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy.bin"
    legacy.write_bytes(b"legacy")
    imported = []
    monkeypatch.delitem(__import__("sys").modules, "nnue_dataset", raising=False)

    original_import = __import__("builtins").__import__

    def guarded_import(name, *args, **kwargs):
        if name == "nnue_dataset":
            imported.append(name)
            raise AssertionError("native loader must not be imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    with pytest.raises(DatasetContractError, match="entrypoint"):
        create_provider(legacy, batch_size=1)
    assert imported == []


def test_provider_reads_an_authenticated_v2_manifest(atomic_v2_manifest):
    provider = create_provider(
        atomic_v2_manifest,
        batch_size=1,
        num_workers=1,
        cyclic=False,
    )

    assert validate_batch(next(provider))[0].shape == (1, 1)
    with pytest.raises(StopIteration):
        next(provider)
