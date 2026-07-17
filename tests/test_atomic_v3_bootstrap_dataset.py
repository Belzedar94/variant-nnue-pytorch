import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

import atomic_v3.bootstrap_dataset as bootstrap_dataset
import atomic_v3.dataset as publication_dataset
import atomic_v3.dataset_source as dataset_source
from atomic_v3.contract import (
    CAMPAIGN_SCHEMA_SHA256,
    FEATURE_SCHEMA_SHA256,
    PUBLICATION_CONTRACT_COMMIT,
    PUBLICATION_SCHEMA_SHA256,
    PUBLICATION_VALIDATOR_CONTRACT,
)
from atomic_v3.dataset import DatasetContractError, PUBLICATION_RECEIPT_FORMAT


def _canonical_bytes(document):
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path, document):
    path.write_bytes(_canonical_bytes(document))


def _write_manifest_json(path, document):
    path.write_bytes(
        (json.dumps(document, allow_nan=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path):
    payload = path.read_bytes()
    return {
        "file": os.fspath(path.absolute()),
        "bytes": str(len(payload)),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _static_artifact(path, byte_count, digest):
    return {
        "file": os.fspath(path.absolute()),
        "bytes": str(byte_count),
        "sha256": digest,
    }


def _threads(index):
    return 6 if index == 2 else 30


def _semantic_result(counters):
    return {
        "type": "atomic-data-tools-validation",
        "contract_version": 1,
        "status": "ok",
        "format": "atomic-bin-v2",
        "entrypoint": "manifest",
        "shards": 1,
        "records": str(bootstrap_dataset.BOOTSTRAP_RECORDS_PER_MANIFEST),
        **counters,
    }


def _write_semantic_evidence(state):
    payload = b"".join(_canonical_bytes(item) for item in state["semantic_results"])
    state["semantic_path"].write_bytes(payload)
    state["receipt_document"]["semantic_validation_jsonl"] = {
        "file": os.fspath(state["semantic_path"].absolute()),
        "bytes": str(len(payload)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "domain_sha256": hashlib.sha256(
            bootstrap_dataset.SEMANTIC_VALIDATION_JSONL_DOMAIN + payload
        ).hexdigest(),
        "chunk_results": bootstrap_dataset.BOOTSTRAP_MANIFESTS,
    }


def _write_receipt(state):
    _write_json(state["receipt_path"], state["receipt_document"])
    return _sha256(state["receipt_path"])


def _selection_manifest(receipt, index):
    if index < bootstrap_dataset.BOOTSTRAP_TRAIN_MANIFESTS:
        return receipt["selection"]["train"]["manifests"][index]
    return receipt["selection"]["validation"]["manifests"][0]


def _refresh_manifest_and_evidence(state, index):
    path = state["manifest_paths"][index]
    _write_manifest_json(path, state["manifest_documents"][index])
    descriptor = _artifact(path)
    slot = _selection_manifest(state["receipt_document"], index)
    slot.clear()
    slot.update(descriptor)
    semantic = state["semantic_results"][index]
    semantic["manifest"] = copy.deepcopy(descriptor)
    shard = state["manifest_documents"][index]["shards"][0]
    semantic["shard"] = {
        "file": os.fspath((path.parent / shard["file"]).absolute()),
        "bytes": shard["bytes"],
        "sha256": shard["sha256"],
    }
    semantic_payload = _canonical_bytes(semantic["semantic_result"])
    semantic["stdout"] = semantic_payload.decode("utf-8")
    semantic["stdout_sha256"] = hashlib.sha256(semantic_payload).hexdigest()
    _write_semantic_evidence(state)
    return _write_receipt(state)


def _bootstrap_fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest_paths = []
    manifest_documents = []
    manifest_descriptors = []
    semantic_results = []
    per_chunk_counters = {
        "side_to_move_wins": "4000000",
        "draws": "4000000",
        "side_to_move_losses": "4500000",
        "atomic960_records": "0",
    }
    for index in range(bootstrap_dataset.BOOTSTRAP_MANIFESTS):
        shard_name = f"test_68_chunk_{index}.bin.atbin"
        shard_sha256 = hashlib.sha256(f"shard-{index}".encode("ascii")).hexdigest()
        document = {
            "manifest_version": 1,
            "manifest_schema_sha256": (
                bootstrap_dataset.ATOMIC_BIN_V2_MANIFEST_SCHEMA_SHA256
            ),
            "data_schema_sha256": bootstrap_dataset.ATOMIC_BIN_V2_DATA_SCHEMA_SHA256,
            "format": "atomic-bin-v2",
            "engine": {"commit": bootstrap_dataset.BOOTSTRAP_ENGINE_COMMIT},
            "generation": {
                "resolved_seed": str(bootstrap_dataset.BOOTSTRAP_BASE_SEED + index),
                "threads": _threads(index),
            },
            "statistics": {
                "records": str(bootstrap_dataset.BOOTSTRAP_RECORDS_PER_MANIFEST)
            },
            "shards": [
                {
                    "index": 0,
                    "file": shard_name,
                    "records": str(bootstrap_dataset.BOOTSTRAP_RECORDS_PER_MANIFEST),
                    "bytes": "800000096",
                    "sha256": shard_sha256,
                }
            ],
        }
        path = tmp_path / f"{shard_name}.manifest.json"
        _write_manifest_json(path, document)
        descriptor = _artifact(path)
        semantic = _semantic_result(per_chunk_counters)
        semantic_payload = _canonical_bytes(semantic)
        role = "train" if index < bootstrap_dataset.BOOTSTRAP_TRAIN_MANIFESTS else "validation"
        semantic_results.append(
            {
                "type": "atomic-bootstrap-semantic-validation-v1",
                "index": index,
                "role": role,
                "generation_threads": _threads(index),
                "status": "ok",
                "exit_code": 0,
                "elapsed_nanoseconds": str(1000 + index),
                "manifest": copy.deepcopy(descriptor),
                "shard": {
                    "file": os.fspath((tmp_path / shard_name).absolute()),
                    "bytes": "800000096",
                    "sha256": shard_sha256,
                },
                "stdout": semantic_payload.decode("utf-8"),
                "stdout_sha256": hashlib.sha256(semantic_payload).hexdigest(),
                "stderr": "",
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                "semantic_result": semantic,
                "semantic_counters": {
                    key: int(value) for key, value in per_chunk_counters.items()
                },
                "inputs_unchanged": True,
            }
        )
        manifest_paths.append(path)
        manifest_documents.append(document)
        manifest_descriptors.append(descriptor)

    semantic_totals = {
        key: str(int(value) * bootstrap_dataset.BOOTSTRAP_MANIFESTS)
        for key, value in per_chunk_counters.items()
    }
    capabilities = {
        "type": "atomic-data-tools-capabilities",
        "contract_version": 1,
        "formats": {
            "atomic-bin-v2": {
                "data_schema_sha256": bootstrap_dataset.ATOMIC_BIN_V2_DATA_SCHEMA_SHA256,
                "manifest_schema_sha256": (
                    bootstrap_dataset.ATOMIC_BIN_V2_MANIFEST_SCHEMA_SHA256
                ),
                "decode_schema_sha256": (
                    bootstrap_dataset.ATOMIC_DATA_TOOLS_DECODE_SCHEMA_SHA256
                ),
                "entrypoint": "manifest",
                "read": True,
                "write": False,
                "operations": ["validate", "decode"],
            }
        },
    }
    receipt_document = {
        "receipt_type": bootstrap_dataset.BOOTSTRAP_RECEIPT_TYPE,
        "status": "ok",
        "purpose": bootstrap_dataset.BOOTSTRAP_PURPOSE,
        "provenance_class": bootstrap_dataset.BOOTSTRAP_PROVENANCE_CLASS,
        "source": {
            "openbench_database": os.fspath((tmp_path / "openbench.sqlite3").absolute()),
            "test": {
                "id": 68,
                "author": "belzedar",
                "test_mode": "DATAGEN",
                "passed": 1,
                "finished": 1,
                "error": 0,
                "creation": "2026-07-16 01:10:08.311658",
                "updated": "2026-07-16 15:39:56.397935",
                "datagen_base_seed": bootstrap_dataset.BOOTSTRAP_BASE_SEED,
                "datagen_positions_per_chunk": (
                    bootstrap_dataset.BOOTSTRAP_RECORDS_PER_MANIFEST
                ),
                "datagen_total_count": bootstrap_dataset.BOOTSTRAP_TOTAL_RECORDS,
                "datagen_completed_chunks": bootstrap_dataset.BOOTSTRAP_MANIFESTS,
                "games": bootstrap_dataset.BOOTSTRAP_TOTAL_RECORDS,
                "datagen_command": "openbench_generate_training_data depth 6",
            },
            "test_row_sha256": "1" * 64,
            "chunks_sha256": "2" * 64,
        },
        "inputs": {
            "data_tools": _static_artifact(tmp_path / "data-tools.exe", 1, "3" * 64),
            "data_tools_source_commit": bootstrap_dataset.BOOTSTRAP_ENGINE_COMMIT,
            "data_tools_build_arch": "x86-64-bmi2",
            "data_tools_capabilities": capabilities,
            "teacher_network": _static_artifact(
                tmp_path / "teacher.nnue",
                bootstrap_dataset.BOOTSTRAP_TEACHER_NETWORK_BYTES,
                bootstrap_dataset.BOOTSTRAP_TEACHER_NETWORK_SHA256,
            ),
            "opening_book": _static_artifact(
                tmp_path / "atomic.epd",
                bootstrap_dataset.BOOTSTRAP_OPENING_BOOK_BYTES,
                bootstrap_dataset.BOOTSTRAP_OPENING_BOOK_SHA256,
            ),
        },
        "semantic_validation_jsonl": {},
        "cross_chunk_audit": {
            "chunk_indices_contiguous": True,
            "schemas_identical": True,
            "engine_commit_identical": True,
            "teacher_network_identical": True,
            "opening_book_identical": True,
            "generation_options_and_filters_identical": True,
            "allowed_differences_only_seed_execution_topology_output_and_statistics": True,
            "generation_threads_distribution": {"6": 1, "30": 29},
            "generation_threads_by_chunk": [
                {"index": index, "threads": _threads(index)}
                for index in range(bootstrap_dataset.BOOTSTRAP_MANIFESTS)
            ],
            "seeds_exact_unique": True,
            "all_manifest_hashes_unique": True,
            "all_shard_hashes_unique": True,
            "all_source_bundle_hashes_unique": True,
            "all_paths_and_identities_distinct": True,
            "semantic_validation_complete": True,
            "semantic_totals": semantic_totals,
            "homogeneous_profile_sha256": "4" * 64,
        },
        "selection": {
            "train": {
                "chunk_indices": list(range(bootstrap_dataset.BOOTSTRAP_TRAIN_MANIFESTS)),
                "records": str(bootstrap_dataset.BOOTSTRAP_TRAIN_RECORDS),
                "manifests": manifest_descriptors[:29],
            },
            "validation": {
                "chunk_indices": [29],
                "records": str(bootstrap_dataset.BOOTSTRAP_VALIDATION_RECORDS),
                "manifests": manifest_descriptors[29:],
            },
            "total_records": str(bootstrap_dataset.BOOTSTRAP_TOTAL_RECORDS),
            "selection_sha256": "5" * 64,
        },
        "scope_guards": {
            "aggregate_atomic_bin_v2_manifest_created": False,
            "synthetic_generation_seed_created": False,
            "atomic_v3_campaign_claimed": False,
            "atomic_v3_publication_ready": False,
            "dataset_publication_ready": False,
            "release_candidate_eligible": False,
            "reason": bootstrap_dataset._SCOPE_REASON,
        },
    }
    state = {
        "manifest_paths": manifest_paths,
        "manifest_documents": manifest_documents,
        "semantic_results": semantic_results,
        "semantic_path": tmp_path / "semantic-validation.jsonl",
        "receipt_path": tmp_path / "bootstrap-receipt.json",
        "receipt_document": receipt_document,
    }
    _write_semantic_evidence(state)
    state["receipt_sha256"] = _write_receipt(state)
    return state


def test_bootstrap_receipt_authenticates_fixed_split_and_non_release_provider(tmp_path):
    state = _bootstrap_fixture(tmp_path)
    snapshot = bootstrap_dataset.inspect_bootstrap_roles(
        state["receipt_path"], state["receipt_sha256"]
    )

    assert [item.chunk_index for item in snapshot.train] == list(range(29))
    assert [item.first_record for item in snapshot.train] == [
        index * bootstrap_dataset.BOOTSTRAP_RECORDS_PER_MANIFEST for index in range(29)
    ]
    assert [item.chunk_index for item in snapshot.validation] == [29]
    assert snapshot.provenance_class == "non-publication-bootstrap"
    assert snapshot.dataset_publication_ready is False
    assert snapshot.release_candidate_eligible is False

    calls = []
    result = bootstrap_dataset.create_bootstrap_role_provider(
        state["receipt_path"],
        state["receipt_sha256"],
        "validation",
        provider_factory=lambda **kwargs: calls.append(kwargs) or "provider",
        batch_size=16384,
    )
    assert result == "provider"
    assert calls[0]["dataset_source"] == "bootstrap"
    assert calls[0]["provenance_class"] == "non-publication-bootstrap"
    assert calls[0]["dataset_publication_ready"] is False
    assert calls[0]["release_candidate_eligible"] is False
    assert calls[0]["manifest_records"] == (12_500_000,)
    assert calls[0]["manifest_payloads"] == (state["manifest_paths"][29].read_bytes(),)
    assert calls[0]["batch_size"] == 16384
    assert not any("attestation" in key or "campaign" in key for key in calls[0])


def test_non_release_scope_is_rejected_before_opening_selected_manifests(tmp_path):
    state = _bootstrap_fixture(tmp_path)
    state["manifest_paths"][0].unlink()
    state["receipt_document"]["scope_guards"]["release_candidate_eligible"] = True
    receipt_sha256 = _write_receipt(state)
    with pytest.raises(DatasetContractError, match="release_candidate_eligible.*false"):
        bootstrap_dataset.inspect_bootstrap_roles(state["receipt_path"], receipt_sha256)


@pytest.mark.parametrize("artifact", ["receipt", "semantic", "manifest"])
def test_bootstrap_provider_reauthenticates_every_authoritative_file(tmp_path, artifact):
    state = _bootstrap_fixture(tmp_path)
    bootstrap_dataset.inspect_bootstrap_roles(state["receipt_path"], state["receipt_sha256"])
    if artifact == "receipt":
        state["receipt_path"].write_bytes(state["receipt_path"].read_bytes() + b"\n")
    elif artifact == "semantic":
        state["semantic_path"].write_bytes(state["semantic_path"].read_bytes() + b"\n")
    else:
        state["manifest_paths"][0].write_bytes(
            state["manifest_paths"][0].read_bytes() + b"\n"
        )
    calls = []
    with pytest.raises(DatasetContractError, match="byte count mismatch|SHA-256 mismatch"):
        bootstrap_dataset.create_bootstrap_role_provider(
            state["receipt_path"],
            state["receipt_sha256"],
            "train",
            provider_factory=lambda **kwargs: calls.append(kwargs),
        )
    assert calls == []


def test_bootstrap_receipt_rejects_duplicate_and_noncanonical_json(tmp_path):
    state = _bootstrap_fixture(tmp_path)
    raw = state["receipt_path"].read_bytes()
    duplicate = raw[:-2] + b',"receipt_type":"duplicate"}\n'
    state["receipt_path"].write_bytes(duplicate)
    with pytest.raises(DatasetContractError, match="duplicate JSON property"):
        bootstrap_dataset.inspect_bootstrap_roles(
            state["receipt_path"], _sha256(state["receipt_path"])
        )

    state = _bootstrap_fixture(tmp_path / "noncanonical")
    pretty = json.dumps(state["receipt_document"], indent=2).encode("utf-8") + b"\n"
    state["receipt_path"].write_bytes(pretty)
    with pytest.raises(DatasetContractError, match="not canonical JSON"):
        bootstrap_dataset.inspect_bootstrap_roles(
            state["receipt_path"], _sha256(state["receipt_path"])
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda document: document["selection"]["validation"].update(chunk_indices=[28]), r"fixed 29\+1 split"),
        (lambda document: document["selection"]["train"].update(records="362499999"), "train record total"),
        (lambda document: document["selection"].update(total_records="374999999"), "aggregate record total"),
        (lambda document: document.update(provenance_class="publication"), "provenance class"),
        (lambda document: document["scope_guards"].update(dataset_publication_ready=True), "must be exactly false"),
        (lambda document: document["scope_guards"].update(release_candidate_eligible=True), "must be exactly false"),
        (
            lambda document: document["cross_chunk_audit"].update(
                generation_threads_distribution={"30": 30}
            ),
            "thread distribution",
        ),
    ],
)
def test_bootstrap_receipt_rejects_wrong_role_total_scope_and_topology(
    tmp_path, mutation, match
):
    state = _bootstrap_fixture(tmp_path)
    mutation(state["receipt_document"])
    receipt_sha256 = _write_receipt(state)
    with pytest.raises(DatasetContractError, match=match):
        bootstrap_dataset.inspect_bootstrap_roles(state["receipt_path"], receipt_sha256)


def test_bootstrap_selection_rejects_duplicate_and_cross_role_overlap(tmp_path):
    duplicate_state = _bootstrap_fixture(tmp_path / "duplicate")
    duplicate_state["receipt_document"]["selection"]["train"]["manifests"][1] = copy.deepcopy(
        duplicate_state["receipt_document"]["selection"]["train"]["manifests"][0]
    )
    duplicate_sha256 = _write_receipt(duplicate_state)
    with pytest.raises(DatasetContractError, match="reuses a manifest"):
        bootstrap_dataset.inspect_bootstrap_roles(
            duplicate_state["receipt_path"], duplicate_sha256
        )

    overlap_state = _bootstrap_fixture(tmp_path / "overlap")
    overlap_state["manifest_documents"][29]["shards"][0]["sha256"] = (
        overlap_state["manifest_documents"][0]["shards"][0]["sha256"]
    )
    overlap_sha256 = _refresh_manifest_and_evidence(overlap_state, 29)
    with pytest.raises(DatasetContractError, match="overlap in shard identity"):
        bootstrap_dataset.inspect_bootstrap_roles(overlap_state["receipt_path"], overlap_sha256)


def test_bootstrap_manifest_schema_and_thread_values_fail_closed(tmp_path):
    schema_state = _bootstrap_fixture(tmp_path / "schema")
    schema_state["manifest_documents"][0]["manifest_schema_sha256"] = "0" * 64
    schema_sha256 = _refresh_manifest_and_evidence(schema_state, 0)
    with pytest.raises(DatasetContractError, match="manifest 0 schema differs"):
        bootstrap_dataset.inspect_bootstrap_roles(schema_state["receipt_path"], schema_sha256)

    thread_state = _bootstrap_fixture(tmp_path / "threads")
    thread_state["manifest_documents"][2]["generation"]["threads"] = 31
    thread_state["semantic_results"][2]["generation_threads"] = 31
    thread_state["receipt_document"]["cross_chunk_audit"]["generation_threads_by_chunk"][2][
        "threads"
    ] = 31
    thread_sha256 = _refresh_manifest_and_evidence(thread_state, 2)
    with pytest.raises(DatasetContractError, match="integer in 1..30"):
        bootstrap_dataset.inspect_bootstrap_roles(thread_state["receipt_path"], thread_sha256)


def test_semantic_evidence_must_match_selected_manifest_and_thread_mapping(tmp_path):
    state = _bootstrap_fixture(tmp_path)
    state["semantic_results"][0]["generation_threads"] = 6
    _write_semantic_evidence(state)
    receipt_sha256 = _write_receipt(state)
    with pytest.raises(DatasetContractError, match="result 0 is not successful"):
        bootstrap_dataset.inspect_bootstrap_roles(state["receipt_path"], receipt_sha256)


@pytest.mark.parametrize("invalid", ["4000000", -1, True])
def test_semantic_evidence_counters_are_non_negative_json_integers(tmp_path, invalid):
    state = _bootstrap_fixture(tmp_path)
    state["semantic_results"][0]["semantic_counters"]["side_to_move_wins"] = invalid
    _write_semantic_evidence(state)
    receipt_sha256 = _write_receipt(state)
    with pytest.raises(DatasetContractError, match="must be a non-negative integer"):
        bootstrap_dataset.inspect_bootstrap_roles(state["receipt_path"], receipt_sha256)


def test_symlink_or_reparse_receipt_and_manifest_paths_are_rejected(tmp_path):
    state = _bootstrap_fixture(tmp_path)
    receipt_link = tmp_path / "receipt-link.json"
    try:
        receipt_link.symlink_to(state["receipt_path"])
    except OSError:
        receipt_link = None
    if receipt_link is not None:
        with pytest.raises(DatasetContractError, match="symbolic links and reparse points"):
            bootstrap_dataset.inspect_bootstrap_roles(receipt_link, state["receipt_sha256"])

    manifest = state["manifest_paths"][0]
    backup = tmp_path / "manifest-backup.json"
    backup.write_bytes(manifest.read_bytes())
    try:
        manifest.unlink()
        manifest.symlink_to(backup)
    except OSError:
        pytest.skip("platform cannot create the manifest symlink fixture")
    with pytest.raises(DatasetContractError, match="symbolic links and reparse points"):
        bootstrap_dataset.inspect_bootstrap_roles(state["receipt_path"], state["receipt_sha256"])


def test_bootstrap_and_publication_receipts_are_cross_rejected(tmp_path):
    state = _bootstrap_fixture(tmp_path)
    missing_campaign = tmp_path / "missing-campaign.json"
    with pytest.raises(DatasetContractError, match="publication receipt fields differ"):
        publication_dataset.inspect_campaign_roles(
            missing_campaign, state["receipt_path"], state["receipt_sha256"]
        )

    publication_receipt = tmp_path / "publication-receipt.json"
    publication_document = {
        "receipt_format": PUBLICATION_RECEIPT_FORMAT,
        "validator_contract": PUBLICATION_VALIDATOR_CONTRACT,
        "publication_contract_commit": PUBLICATION_CONTRACT_COMMIT,
        "publication_schema_sha256": dict(PUBLICATION_SCHEMA_SHA256),
        "campaign_schema_sha256": CAMPAIGN_SCHEMA_SHA256,
        "campaign_sha256": "1" * 64,
        "collection_sha256": "2" * 64,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "producer_attestation_sha256": "3" * 64,
        "semantic_audit_sha256": "4" * 64,
        "reachability_attestation_sha256": "5" * 64,
        "dataset_publication_ready": True,
    }
    _write_json(publication_receipt, publication_document)
    with pytest.raises(DatasetContractError, match="bootstrap receipt.*fields differ"):
        bootstrap_dataset.inspect_bootstrap_roles(
            publication_receipt, _sha256(publication_receipt)
        )


def test_dataset_source_api_and_cli_require_exactly_one_explicit_contract(
    tmp_path, monkeypatch
):
    receipt = tmp_path / "receipt.json"
    campaign = tmp_path / "campaign.json"
    digest = "a" * 64
    publication = dataset_source.publication_source(campaign, receipt, digest)
    bootstrap = dataset_source.bootstrap_source(receipt, digest)
    with pytest.raises(DatasetContractError, match="exactly one"):
        dataset_source.create_selected_role_provider(
            "train", provider_factory=lambda **kwargs: kwargs
        )
    with pytest.raises(DatasetContractError, match="exactly one"):
        dataset_source.create_selected_role_provider(
            "train",
            publication=publication,
            bootstrap=bootstrap,
            provider_factory=lambda **kwargs: kwargs,
        )

    calls = []
    monkeypatch.setattr(
        dataset_source.dataset,
        "create_role_provider",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "publication",
    )
    assert (
        dataset_source.create_selected_role_provider(
            "validation", publication=publication, provider_factory=lambda **kwargs: kwargs
        )
        == "publication"
    )
    assert len(calls) == 1

    with pytest.raises(SystemExit):
        dataset_source.parse_dataset_source_args([])
    with pytest.raises(SystemExit):
        dataset_source.parse_dataset_source_args(
            [
                "--publication-source",
                os.fspath(campaign),
                os.fspath(receipt),
                digest,
                "--bootstrap-source",
                os.fspath(receipt),
                digest,
            ]
        )
    selected = dataset_source.parse_dataset_source_args(
        ["--bootstrap-source", os.fspath(receipt), digest]
    )
    assert selected == bootstrap
