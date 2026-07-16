"""Strict non-publication bootstrap receipt seam for AtomicNNUEV3.

This module is intentionally separate from :mod:`atomic_v3.dataset`.  A
bootstrap receipt authenticates the fixed OpenBench #68 29+1 pilot selection,
but it can never satisfy, emulate, or fall back to the publication campaign
contract.  The receipt SHA-256 is always supplied out of process.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Union

from .dataset import (
    ATOMIC_BIN_V2_MANIFEST_SCHEMA_SHA256,
    DatasetContractError,
    RoleManifest,
    _is_plain_int,
    _read_regular_authenticated,
    _require_path,
    _require_sha256,
    _require_uint_string,
    _strict_json_document,
)


BOOTSTRAP_RECEIPT_TYPE = "atomic-v3-bootstrap-training-receipt-v1"
BOOTSTRAP_PURPOSE = "atomic-nnue-v3-bootstrap-training-pilot"
BOOTSTRAP_PROVENANCE_CLASS = "non-publication-bootstrap"
BOOTSTRAP_ENGINE_COMMIT = "1adc5239500a684824b178b19be4266f6e249cc8"
ATOMIC_BIN_V2_DATA_SCHEMA_SHA256 = (
    "0352b036f2a140c609e3eb9c9d635dc553e8d77253d8faa92437390f5cf93cb6"
)
ATOMIC_DATA_TOOLS_DECODE_SCHEMA_SHA256 = (
    "5e3f8d7c6db6ee955b71747ee063859e15609adb557a3754228a606f3df2caad"
)
BOOTSTRAP_TEACHER_NETWORK_SHA256 = (
    "99dc67eabf26a64faeeca3a88b4c38597a840b8d4a874b9f2cf658c6f92a04a6"
)
BOOTSTRAP_TEACHER_NETWORK_BYTES = 47_721_376
BOOTSTRAP_OPENING_BOOK_SHA256 = (
    "28ed51c2f42e723d5e127d2d3f21c0bfa4a9b318615afdb299b93ea62dea2b1e"
)
BOOTSTRAP_OPENING_BOOK_BYTES = 394_785
BOOTSTRAP_OPENBENCH_TEST_ID = 68
BOOTSTRAP_BASE_SEED = 202_607_150_500_000
BOOTSTRAP_RECORDS_PER_MANIFEST = 12_500_000
BOOTSTRAP_TRAIN_MANIFESTS = 29
BOOTSTRAP_VALIDATION_MANIFESTS = 1
BOOTSTRAP_MANIFESTS = BOOTSTRAP_TRAIN_MANIFESTS + BOOTSTRAP_VALIDATION_MANIFESTS
BOOTSTRAP_TRAIN_RECORDS = BOOTSTRAP_TRAIN_MANIFESTS * BOOTSTRAP_RECORDS_PER_MANIFEST
BOOTSTRAP_VALIDATION_RECORDS = (
    BOOTSTRAP_VALIDATION_MANIFESTS * BOOTSTRAP_RECORDS_PER_MANIFEST
)
BOOTSTRAP_TOTAL_RECORDS = BOOTSTRAP_TRAIN_RECORDS + BOOTSTRAP_VALIDATION_RECORDS
MAX_BOOTSTRAP_RECEIPT_BYTES = 256 * 1024
MAX_BOOTSTRAP_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_SEMANTIC_VALIDATION_JSONL_BYTES = 64 * 1024 * 1024
SEMANTIC_VALIDATION_JSONL_DOMAIN = (
    b"atomic-bootstrap-semantic-validation-jsonl-v1\0"
)

_SAFE_BASENAME_RE = re.compile(r'^(?!\.{1,2}$)[^/\\:\x00<>"|?*]+$')
_RECEIPT_KEYS = {
    "receipt_type",
    "status",
    "purpose",
    "provenance_class",
    "source",
    "inputs",
    "semantic_validation_jsonl",
    "cross_chunk_audit",
    "selection",
    "scope_guards",
}
_SOURCE_KEYS = {
    "openbench_database",
    "test",
    "test_row_sha256",
    "chunks_sha256",
}
_OPENBENCH_TEST_KEYS = {
    "id",
    "author",
    "test_mode",
    "passed",
    "finished",
    "error",
    "creation",
    "updated",
    "datagen_base_seed",
    "datagen_positions_per_chunk",
    "datagen_total_count",
    "datagen_completed_chunks",
    "games",
    "datagen_command",
}
_INPUT_KEYS = {
    "data_tools",
    "data_tools_source_commit",
    "data_tools_build_arch",
    "data_tools_capabilities",
    "teacher_network",
    "opening_book",
}
_ARTIFACT_KEYS = {"file", "bytes", "sha256"}
_SEMANTIC_ARTIFACT_KEYS = {
    "file",
    "bytes",
    "sha256",
    "domain_sha256",
    "chunk_results",
}
_AUDIT_BOOLEAN_KEYS = {
    "chunk_indices_contiguous",
    "schemas_identical",
    "engine_commit_identical",
    "teacher_network_identical",
    "opening_book_identical",
    "generation_options_and_filters_identical",
    "allowed_differences_only_seed_execution_topology_output_and_statistics",
    "seeds_exact_unique",
    "all_manifest_hashes_unique",
    "all_shard_hashes_unique",
    "all_source_bundle_hashes_unique",
    "all_paths_and_identities_distinct",
    "semantic_validation_complete",
}
_AUDIT_KEYS = _AUDIT_BOOLEAN_KEYS | {
    "semantic_totals",
    "homogeneous_profile_sha256",
    "generation_threads_distribution",
    "generation_threads_by_chunk",
}
_SEMANTIC_COUNTER_KEYS = {
    "side_to_move_wins",
    "draws",
    "side_to_move_losses",
    "atomic960_records",
}
_SEMANTIC_RESULT_KEYS = {
    "type",
    "index",
    "role",
    "generation_threads",
    "status",
    "exit_code",
    "elapsed_nanoseconds",
    "manifest",
    "shard",
    "stdout",
    "stdout_sha256",
    "stderr",
    "stderr_sha256",
    "semantic_result",
    "semantic_counters",
    "inputs_unchanged",
}
_SELECTION_KEYS = {"train", "validation", "total_records", "selection_sha256"}
_ROLE_KEYS = {"chunk_indices", "records", "manifests"}
_SCOPE_KEYS = {
    "aggregate_atomic_bin_v2_manifest_created",
    "synthetic_generation_seed_created",
    "atomic_v3_campaign_claimed",
    "atomic_v3_publication_ready",
    "dataset_publication_ready",
    "release_candidate_eligible",
    "reason",
}
_SCOPE_BOOLEAN_KEYS = _SCOPE_KEYS - {"reason"}
_SCOPE_REASON = (
    "bootstrap bundles do not contain the per-role trajectory ledgers, "
    "coverage, split audits, and attestations required by the Atomic V3 "
    "publication contract"
)


@dataclass(frozen=True)
class _ArtifactDescriptor:
    path: Path
    bytes: int
    sha256: str


@dataclass(frozen=True)
class _ShardBinding:
    path: Path
    bytes: int
    sha256: str
    records: int
    generation_threads: int


@dataclass(frozen=True)
class BootstrapReceiptSnapshot:
    """Authenticated bootstrap evidence and explicit non-release scope."""

    receipt_path: Path
    receipt_sha256: str
    receipt_payload: bytes
    semantic_validation_jsonl_path: Path
    semantic_validation_jsonl_sha256: str
    semantic_validation_jsonl_domain_sha256: str
    semantic_validation_jsonl_payload: bytes
    selection_sha256: str
    provenance_class: str
    dataset_publication_ready: bool
    release_candidate_eligible: bool
    train: tuple[RoleManifest, ...]
    validation: tuple[RoleManifest, ...]

    def manifests(self, role: Literal["train", "validation"]) -> tuple[RoleManifest, ...]:
        if role == "train":
            return self.train
        if role == "validation":
            return self.validation
        raise DatasetContractError("role must be exactly 'train' or 'validation'")


def _canonical_json_bytes(document: object) -> bytes:
    try:
        wire = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise DatasetContractError(f"bootstrap receipt is not canonical JSON: {error}") from error
    return (wire + "\n").encode("utf-8")


def _require_keys(label: str, value: object, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise DatasetContractError(f"{label} fields differ from the frozen bootstrap contract")
    return value


def _require_absolute_path(label: str, value: object) -> Path:
    path = _require_path(label, value)
    if not path.is_absolute():
        raise DatasetContractError(f"{label} must be an absolute path")
    return Path(os.path.abspath(str(path)))


def _artifact_descriptor(
    label: str,
    value: object,
    *,
    expected_bytes: Optional[int] = None,
    expected_sha256: Optional[str] = None,
) -> _ArtifactDescriptor:
    document = _require_keys(label, value, _ARTIFACT_KEYS)
    path = _require_absolute_path(f"{label}.file", document["file"])
    byte_count = _require_uint_string(f"{label}.bytes", document["bytes"], positive=True)
    digest = _require_sha256(f"{label}.sha256", document["sha256"])
    if expected_bytes is not None and byte_count != expected_bytes:
        raise DatasetContractError(f"{label} byte count differs from the bootstrap contract")
    if expected_sha256 is not None and digest != expected_sha256:
        raise DatasetContractError(f"{label} SHA-256 differs from the bootstrap contract")
    return _ArtifactDescriptor(path, byte_count, digest)


def _validate_source(value: object) -> None:
    source = _require_keys("source", value, _SOURCE_KEYS)
    _require_absolute_path("source.openbench_database", source["openbench_database"])
    _require_sha256("source.test_row_sha256", source["test_row_sha256"])
    _require_sha256("source.chunks_sha256", source["chunks_sha256"])
    test = _require_keys("source.test", source["test"], _OPENBENCH_TEST_KEYS)
    expected_values = {
        "id": BOOTSTRAP_OPENBENCH_TEST_ID,
        "passed": 1,
        "finished": 1,
        "error": 0,
        "datagen_base_seed": BOOTSTRAP_BASE_SEED,
        "datagen_positions_per_chunk": BOOTSTRAP_RECORDS_PER_MANIFEST,
        "datagen_total_count": BOOTSTRAP_TOTAL_RECORDS,
        "datagen_completed_chunks": BOOTSTRAP_MANIFESTS,
        "games": BOOTSTRAP_TOTAL_RECORDS,
    }
    for field, expected in expected_values.items():
        if not _is_plain_int(test[field]) or test[field] != expected:
            raise DatasetContractError(f"source.test.{field} differs from OpenBench #68")
    if test["author"] != "belzedar" or test["test_mode"] != "DATAGEN":
        raise DatasetContractError("source.test identity differs from OpenBench #68")
    for field in ("creation", "updated", "datagen_command"):
        if not isinstance(test[field], str) or not test[field]:
            raise DatasetContractError(f"source.test.{field} must be a non-empty string")


def _expected_capabilities() -> dict[str, Any]:
    return {
        "type": "atomic-data-tools-capabilities",
        "contract_version": 1,
        "formats": {
            "atomic-bin-v2": {
                "data_schema_sha256": ATOMIC_BIN_V2_DATA_SCHEMA_SHA256,
                "manifest_schema_sha256": ATOMIC_BIN_V2_MANIFEST_SCHEMA_SHA256,
                "decode_schema_sha256": ATOMIC_DATA_TOOLS_DECODE_SCHEMA_SHA256,
                "entrypoint": "manifest",
                "read": True,
                "write": False,
                "operations": ["validate", "decode"],
            }
        },
    }


def _validate_inputs(value: object) -> None:
    inputs = _require_keys("inputs", value, _INPUT_KEYS)
    _artifact_descriptor("inputs.data_tools", inputs["data_tools"])
    _artifact_descriptor(
        "inputs.teacher_network",
        inputs["teacher_network"],
        expected_bytes=BOOTSTRAP_TEACHER_NETWORK_BYTES,
        expected_sha256=BOOTSTRAP_TEACHER_NETWORK_SHA256,
    )
    _artifact_descriptor(
        "inputs.opening_book",
        inputs["opening_book"],
        expected_bytes=BOOTSTRAP_OPENING_BOOK_BYTES,
        expected_sha256=BOOTSTRAP_OPENING_BOOK_SHA256,
    )
    if inputs["data_tools_source_commit"] != BOOTSTRAP_ENGINE_COMMIT:
        raise DatasetContractError("data-tools source commit differs from the dataset producer")
    if inputs["data_tools_build_arch"] != "x86-64-bmi2":
        raise DatasetContractError("data-tools build architecture differs from bootstrap contract")
    if inputs["data_tools_capabilities"] != _expected_capabilities():
        raise DatasetContractError("data-tools capabilities differ from contract version 1")


def _validate_manifest_payload(
    payload: bytes, path: Path, *, chunk_index: int, role: str
) -> _ShardBinding:
    document = _strict_json_document(payload, f"bootstrap manifest {chunk_index}")
    if (
        payload.startswith(b"\xef\xbb\xbf")
        or b"\r" in payload
        or not payload.endswith(b"\n")
        or payload.endswith(b"\n\n")
    ):
        raise DatasetContractError(
            f"bootstrap manifest {chunk_index} is not canonical UTF-8/LF"
        )
    if (
        not _is_plain_int(document.get("manifest_version"))
        or document.get("manifest_version") != 1
        or document.get("format") != "atomic-bin-v2"
    ):
        raise DatasetContractError(f"bootstrap manifest {chunk_index} is not Atomic BIN V2")
    if document.get("manifest_schema_sha256") != ATOMIC_BIN_V2_MANIFEST_SCHEMA_SHA256:
        raise DatasetContractError(f"bootstrap manifest {chunk_index} schema differs")
    if document.get("data_schema_sha256") != ATOMIC_BIN_V2_DATA_SCHEMA_SHA256:
        raise DatasetContractError(f"bootstrap manifest {chunk_index} data schema differs")
    engine = document.get("engine")
    generation = document.get("generation")
    statistics = document.get("statistics")
    shards = document.get("shards")
    if not isinstance(engine, Mapping) or engine.get("commit") != BOOTSTRAP_ENGINE_COMMIT:
        raise DatasetContractError(f"bootstrap manifest {chunk_index} engine commit differs")
    if not isinstance(generation, Mapping):
        raise DatasetContractError(f"bootstrap manifest {chunk_index} generation is missing")
    seed = _require_uint_string(
        f"bootstrap manifest {chunk_index} generation.resolved_seed",
        generation.get("resolved_seed"),
    )
    if seed != BOOTSTRAP_BASE_SEED + chunk_index:
        raise DatasetContractError(f"bootstrap manifest {chunk_index} seed differs")
    if not isinstance(statistics, Mapping):
        raise DatasetContractError(f"bootstrap manifest {chunk_index} statistics are missing")
    records = _require_uint_string(
        f"bootstrap manifest {chunk_index} statistics.records",
        statistics.get("records"),
        positive=True,
    )
    if records != BOOTSTRAP_RECORDS_PER_MANIFEST:
        raise DatasetContractError(f"bootstrap manifest {chunk_index} record count differs")
    if not isinstance(shards, list) or len(shards) != 1:
        raise DatasetContractError(f"bootstrap manifest {chunk_index} must bind one shard")
    shard = _require_keys(
        f"bootstrap manifest {chunk_index} shard",
        shards[0],
        {"index", "file", "records", "bytes", "sha256"},
    )
    if not _is_plain_int(shard["index"]) or shard["index"] != 0:
        raise DatasetContractError(f"bootstrap manifest {chunk_index} shard index differs")
    filename = shard["file"]
    expected_filename = f"test_68_chunk_{chunk_index}.bin.atbin"
    if (
        not isinstance(filename, str)
        or _SAFE_BASENAME_RE.fullmatch(filename) is None
        or filename != expected_filename
        or path.name != f"{expected_filename}.manifest.json"
    ):
        raise DatasetContractError(f"bootstrap manifest {chunk_index} path/role binding differs")
    shard_records = _require_uint_string(
        f"bootstrap manifest {chunk_index} shard.records", shard["records"], positive=True
    )
    shard_bytes = _require_uint_string(
        f"bootstrap manifest {chunk_index} shard.bytes", shard["bytes"], positive=True
    )
    if shard_records != BOOTSTRAP_RECORDS_PER_MANIFEST:
        raise DatasetContractError(f"bootstrap manifest {chunk_index} shard records differ")
    shard_sha256 = _require_sha256(
        f"bootstrap manifest {chunk_index} shard.sha256", shard["sha256"]
    )
    expected_role = "train" if chunk_index < BOOTSTRAP_TRAIN_MANIFESTS else "validation"
    if role != expected_role:
        raise DatasetContractError(f"bootstrap manifest {chunk_index} is assigned to the wrong role")
    generation_threads = generation.get("threads")
    if not _is_plain_int(generation_threads) or not 1 <= generation_threads <= 30:
        raise DatasetContractError(
            f"bootstrap manifest {chunk_index} generation threads must be an integer in 1..30"
        )
    return _ShardBinding(
        path.parent / filename,
        shard_bytes,
        shard_sha256,
        shard_records,
        generation_threads,
    )


def _load_role_manifests(
    selection: Mapping[str, Any],
    receipt_identity: tuple[int, int, int, int],
) -> tuple[
    tuple[RoleManifest, ...],
    tuple[RoleManifest, ...],
    dict[int, _ArtifactDescriptor],
    dict[int, _ShardBinding],
    dict[int, int],
    frozenset[tuple[int, int, int, int]],
]:
    roles: dict[str, list[RoleManifest]] = {"train": [], "validation": []}
    descriptors_by_chunk: dict[int, _ArtifactDescriptor] = {}
    shards_by_chunk: dict[int, _ShardBinding] = {}
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    seen_identities = {receipt_identity}
    seen_shard_paths: set[str] = set()
    seen_shard_hashes: set[str] = set()
    generation_threads: dict[int, int] = {}
    role_specs = (
        ("train", list(range(BOOTSTRAP_TRAIN_MANIFESTS)), BOOTSTRAP_TRAIN_RECORDS),
        ("validation", [BOOTSTRAP_TRAIN_MANIFESTS], BOOTSTRAP_VALIDATION_RECORDS),
    )
    for role, expected_indices, expected_records in role_specs:
        partition = _require_keys(f"selection.{role}", selection[role], _ROLE_KEYS)
        indices = partition["chunk_indices"]
        if (
            not isinstance(indices, list)
            or any(not _is_plain_int(index) for index in indices)
            or indices != expected_indices
        ):
            raise DatasetContractError(f"selection.{role} chunk indices differ from fixed 29+1 split")
        declared_records = _require_uint_string(
            f"selection.{role}.records", partition["records"], positive=True
        )
        if declared_records != expected_records:
            raise DatasetContractError(f"selection.{role} record total differs")
        descriptors = partition["manifests"]
        if not isinstance(descriptors, list) or len(descriptors) != len(expected_indices):
            raise DatasetContractError(f"selection.{role} manifest count differs")
        for role_offset, (chunk_index, value) in enumerate(zip(expected_indices, descriptors)):
            descriptor = _artifact_descriptor(
                f"selection.{role}.manifests[{role_offset}]", value
            )
            path_key = os.path.normcase(os.path.abspath(str(descriptor.path)))
            if path_key in seen_paths or descriptor.sha256 in seen_hashes:
                raise DatasetContractError("bootstrap selection reuses a manifest path or SHA-256")
            artifact = _read_regular_authenticated(
                descriptor.path,
                descriptor.sha256,
                expected_bytes=descriptor.bytes,
                maximum=MAX_BOOTSTRAP_MANIFEST_BYTES,
            )
            if artifact.identity in seen_identities:
                raise DatasetContractError("bootstrap selection reuses a filesystem artifact identity")
            shard = _validate_manifest_payload(
                artifact.payload, artifact.path, chunk_index=chunk_index, role=role
            )
            shard_path_key = os.path.normcase(os.path.abspath(str(shard.path)))
            if shard_path_key in seen_shard_paths or shard.sha256 in seen_shard_hashes:
                raise DatasetContractError("bootstrap train/validation selections overlap in shard identity")
            seen_paths.add(path_key)
            seen_hashes.add(descriptor.sha256)
            seen_identities.add(artifact.identity)
            seen_shard_paths.add(shard_path_key)
            seen_shard_hashes.add(shard.sha256)
            first_record = role_offset * BOOTSTRAP_RECORDS_PER_MANIFEST
            roles[role].append(
                RoleManifest(
                    chunk_index,
                    first_record,
                    BOOTSTRAP_RECORDS_PER_MANIFEST,
                    artifact.path,
                    artifact.sha256,
                    artifact.payload,
                )
            )
            descriptors_by_chunk[chunk_index] = descriptor
            shards_by_chunk[chunk_index] = shard
            generation_threads[chunk_index] = shard.generation_threads
    thread_distribution: dict[int, int] = {}
    for threads in generation_threads.values():
        thread_distribution[threads] = thread_distribution.get(threads, 0) + 1
    if thread_distribution != {6: 1, 30: 29}:
        raise DatasetContractError(
            "bootstrap generation thread topology differs from the authenticated 6x1/30x29 run"
        )
    return (
        tuple(roles["train"]),
        tuple(roles["validation"]),
        descriptors_by_chunk,
        shards_by_chunk,
        generation_threads,
        frozenset(seen_identities),
    )


def _validate_semantic_jsonl(
    descriptor_value: object,
    manifest_descriptors: Mapping[int, _ArtifactDescriptor],
    shard_bindings: Mapping[int, _ShardBinding],
    generation_threads: Mapping[int, int],
    occupied_identities: set[tuple[int, int, int, int]],
) -> tuple[Path, str, str, bytes, dict[str, int], tuple[int, int, int, int]]:
    descriptor = _require_keys(
        "semantic_validation_jsonl", descriptor_value, _SEMANTIC_ARTIFACT_KEYS
    )
    path = _require_absolute_path("semantic_validation_jsonl.file", descriptor["file"])
    byte_count = _require_uint_string(
        "semantic_validation_jsonl.bytes", descriptor["bytes"], positive=True
    )
    digest = _require_sha256("semantic_validation_jsonl.sha256", descriptor["sha256"])
    domain_digest = _require_sha256(
        "semantic_validation_jsonl.domain_sha256", descriptor["domain_sha256"]
    )
    if not _is_plain_int(descriptor["chunk_results"]) or descriptor["chunk_results"] != BOOTSTRAP_MANIFESTS:
        raise DatasetContractError("semantic validation result count differs")
    artifact = _read_regular_authenticated(
        path,
        digest,
        expected_bytes=byte_count,
        maximum=MAX_SEMANTIC_VALIDATION_JSONL_BYTES,
    )
    if artifact.identity in occupied_identities:
        raise DatasetContractError("semantic validation JSONL overlaps an authenticated receipt/manifest")
    actual_domain = hashlib.sha256(SEMANTIC_VALIDATION_JSONL_DOMAIN + artifact.payload).hexdigest()
    if actual_domain != domain_digest:
        raise DatasetContractError("semantic validation JSONL domain hash differs")
    lines = artifact.payload.splitlines(keepends=True)
    if len(lines) != BOOTSTRAP_MANIFESTS or any(not line.endswith(b"\n") for line in lines):
        raise DatasetContractError("semantic validation JSONL is not exactly 30 LF records")
    totals = {key: 0 for key in _SEMANTIC_COUNTER_KEYS}
    for chunk_index, line in enumerate(lines):
        document = _strict_json_document(line, f"semantic validation result {chunk_index}")
        if _canonical_json_bytes(document) != line:
            raise DatasetContractError(f"semantic validation result {chunk_index} is not canonical JSON")
        _require_keys(
            f"semantic validation result {chunk_index}", document, _SEMANTIC_RESULT_KEYS
        )
        expected_role = "train" if chunk_index < BOOTSTRAP_TRAIN_MANIFESTS else "validation"
        if (
            document.get("type") != "atomic-bootstrap-semantic-validation-v1"
            or not _is_plain_int(document.get("index"))
            or document.get("index") != chunk_index
            or document.get("role") != expected_role
            or not _is_plain_int(document.get("generation_threads"))
            or document.get("generation_threads") != generation_threads[chunk_index]
            or document.get("status") != "ok"
            or not _is_plain_int(document.get("exit_code"))
            or document.get("exit_code") != 0
            or document.get("inputs_unchanged") is not True
        ):
            raise DatasetContractError(f"semantic validation result {chunk_index} is not successful")
        _require_uint_string(
            f"semantic validation result {chunk_index}.elapsed_nanoseconds",
            document["elapsed_nanoseconds"],
        )
        manifest = _artifact_descriptor(
            f"semantic validation result {chunk_index}.manifest", document.get("manifest")
        )
        selected = manifest_descriptors[chunk_index]
        if manifest != selected:
            raise DatasetContractError(
                f"semantic validation result {chunk_index} does not bind the selected manifest"
            )
        shard = _artifact_descriptor(
            f"semantic validation result {chunk_index}.shard", document.get("shard")
        )
        selected_shard = shard_bindings[chunk_index]
        if shard != _ArtifactDescriptor(
            Path(os.path.abspath(str(selected_shard.path))),
            selected_shard.bytes,
            selected_shard.sha256,
        ):
            raise DatasetContractError(
                f"semantic validation result {chunk_index} does not bind the selected shard"
            )
        stdout = document["stdout"]
        stderr = document["stderr"]
        semantic_result = document["semantic_result"]
        if not isinstance(stdout, str) or not isinstance(stderr, str) or stderr != "":
            raise DatasetContractError(f"semantic validation result {chunk_index} output differs")
        if not isinstance(semantic_result, Mapping):
            raise DatasetContractError(f"semantic validation result {chunk_index} payload is missing")
        stdout_document = _strict_json_document(
            stdout.encode("utf-8"), f"semantic validation result {chunk_index} stdout"
        )
        if stdout_document != semantic_result:
            raise DatasetContractError(f"semantic validation result {chunk_index} stdout differs")
        if _require_sha256(
            f"semantic validation result {chunk_index}.stdout_sha256",
            document["stdout_sha256"],
        ) != hashlib.sha256(stdout.encode("utf-8")).hexdigest():
            raise DatasetContractError(f"semantic validation result {chunk_index} stdout hash differs")
        if _require_sha256(
            f"semantic validation result {chunk_index}.stderr_sha256",
            document["stderr_sha256"],
        ) != hashlib.sha256(b"").hexdigest():
            raise DatasetContractError(f"semantic validation result {chunk_index} stderr hash differs")
        counters = _require_keys(
            f"semantic validation result {chunk_index}.semantic_counters",
            document.get("semantic_counters"),
            _SEMANTIC_COUNTER_KEYS,
        )
        # The verifier keeps data-tools' stdout counters as decimal strings in
        # ``semantic_result``, then emits its independently parsed counters as
        # JSON integers.  Preserve that type boundary so the receipt consumer
        # validates the verifier's real output rather than a lookalike fixture.
        parsed: dict[str, int] = {}
        for key in _SEMANTIC_COUNTER_KEYS:
            value = counters[key]
            if not _is_plain_int(value) or value < 0:
                raise DatasetContractError(
                    f"semantic validation result {chunk_index}.semantic_counters.{key} "
                    "must be a non-negative integer"
                )
            parsed[key] = value
        if (
            parsed["side_to_move_wins"] + parsed["draws"] + parsed["side_to_move_losses"]
            != BOOTSTRAP_RECORDS_PER_MANIFEST
            or parsed["atomic960_records"] != 0
        ):
            raise DatasetContractError(f"semantic validation result {chunk_index} counters differ")
        for key, value in parsed.items():
            totals[key] += value
    return artifact.path, digest, domain_digest, artifact.payload, totals, artifact.identity


def _validate_audit(
    value: object,
    semantic_totals: Mapping[str, int],
    generation_threads: Mapping[int, int],
) -> None:
    audit = _require_keys("cross_chunk_audit", value, _AUDIT_KEYS)
    for field in _AUDIT_BOOLEAN_KEYS:
        if audit[field] is not True:
            raise DatasetContractError(f"cross_chunk_audit.{field} must be exactly true")
    _require_sha256(
        "cross_chunk_audit.homogeneous_profile_sha256",
        audit["homogeneous_profile_sha256"],
    )
    if audit["generation_threads_distribution"] != {"6": 1, "30": 29}:
        raise DatasetContractError("cross-chunk generation thread distribution differs")
    expected_by_chunk = [
        {"index": index, "threads": generation_threads[index]}
        for index in range(BOOTSTRAP_MANIFESTS)
    ]
    if audit["generation_threads_by_chunk"] != expected_by_chunk:
        raise DatasetContractError("cross-chunk generation thread mapping differs from manifests")
    declared = _require_keys(
        "cross_chunk_audit.semantic_totals",
        audit["semantic_totals"],
        _SEMANTIC_COUNTER_KEYS,
    )
    for key in _SEMANTIC_COUNTER_KEYS:
        if _require_uint_string(f"cross_chunk_audit.semantic_totals.{key}", declared[key]) != semantic_totals[key]:
            raise DatasetContractError("cross-chunk semantic totals do not match the JSONL evidence")


def _validate_scope(value: object) -> None:
    scope = _require_keys("scope_guards", value, _SCOPE_KEYS)
    for field in _SCOPE_BOOLEAN_KEYS:
        if scope[field] is not False:
            raise DatasetContractError(f"scope_guards.{field} must be exactly false")
    if scope["reason"] != _SCOPE_REASON:
        raise DatasetContractError("bootstrap non-publication reason differs")


def _load_bootstrap_receipt(
    receipt_path: Union[str, Path], expected_receipt_sha256: str
) -> BootstrapReceiptSnapshot:
    path = _require_path("bootstrap_receipt_path", receipt_path)
    expected = _require_sha256("expected_bootstrap_receipt_sha256", expected_receipt_sha256)
    receipt_artifact = _read_regular_authenticated(
        path, expected, maximum=MAX_BOOTSTRAP_RECEIPT_BYTES
    )
    document = _strict_json_document(receipt_artifact.payload, "bootstrap receipt")
    if _canonical_json_bytes(document) != receipt_artifact.payload:
        raise DatasetContractError("bootstrap receipt is not canonical JSON")
    receipt = _require_keys("bootstrap receipt", document, _RECEIPT_KEYS)
    if receipt["receipt_type"] != BOOTSTRAP_RECEIPT_TYPE:
        raise DatasetContractError("unrecognized bootstrap receipt type")
    if receipt["status"] != "ok":
        raise DatasetContractError("bootstrap receipt status is not ok")
    if receipt["purpose"] != BOOTSTRAP_PURPOSE:
        raise DatasetContractError("bootstrap receipt purpose differs")
    if receipt["provenance_class"] != BOOTSTRAP_PROVENANCE_CLASS:
        raise DatasetContractError("bootstrap receipt provenance class differs")
    # Reject any attempted publication/release claim before opening evidence or
    # selected manifests.  Bootstrap scope is a fail-closed identity, not a
    # property that later provider options may upgrade.
    _validate_scope(receipt["scope_guards"])
    _validate_source(receipt["source"])
    _validate_inputs(receipt["inputs"])

    selection = _require_keys("selection", receipt["selection"], _SELECTION_KEYS)
    total_records = _require_uint_string(
        "selection.total_records", selection["total_records"], positive=True
    )
    if total_records != BOOTSTRAP_TOTAL_RECORDS:
        raise DatasetContractError("selection aggregate record total differs")
    selection_sha256 = _require_sha256("selection.selection_sha256", selection["selection_sha256"])
    (
        train,
        validation,
        manifest_descriptors,
        shard_bindings,
        generation_threads,
        occupied_identity_snapshot,
    ) = _load_role_manifests(selection, receipt_artifact.identity)
    occupied_identities = set(occupied_identity_snapshot)
    (
        semantic_path,
        semantic_sha256,
        semantic_domain_sha256,
        semantic_payload,
        semantic_totals,
        _,
    ) = _validate_semantic_jsonl(
        receipt["semantic_validation_jsonl"],
        manifest_descriptors,
        shard_bindings,
        generation_threads,
        occupied_identities,
    )
    _validate_audit(receipt["cross_chunk_audit"], semantic_totals, generation_threads)

    return BootstrapReceiptSnapshot(
        receipt_path=receipt_artifact.path,
        receipt_sha256=expected,
        receipt_payload=receipt_artifact.payload,
        semantic_validation_jsonl_path=semantic_path,
        semantic_validation_jsonl_sha256=semantic_sha256,
        semantic_validation_jsonl_domain_sha256=semantic_domain_sha256,
        semantic_validation_jsonl_payload=semantic_payload,
        selection_sha256=selection_sha256,
        provenance_class=BOOTSTRAP_PROVENANCE_CLASS,
        dataset_publication_ready=False,
        release_candidate_eligible=False,
        train=train,
        validation=validation,
    )


def inspect_bootstrap_roles(
    receipt_path: Union[str, Path], expected_receipt_sha256: str
) -> BootstrapReceiptSnapshot:
    """Authenticate the fixed bootstrap receipt and return diagnostic bytes."""

    return _load_bootstrap_receipt(receipt_path, expected_receipt_sha256)


def create_bootstrap_role_provider(
    receipt_path: Union[str, Path],
    expected_receipt_sha256: str,
    role: Literal["train", "validation"],
    *,
    provider_factory: Callable[..., Any],
    **provider_options: Any,
) -> Any:
    """Reauthenticate a non-publication bootstrap role, with no fallback."""

    if not callable(provider_factory):
        raise TypeError("provider_factory must be callable")
    snapshot = _load_bootstrap_receipt(receipt_path, expected_receipt_sha256)
    manifests = snapshot.manifests(role)
    return provider_factory(
        backend="atomic-nnue-v3",
        dataset_source="bootstrap",
        provenance_class=snapshot.provenance_class,
        dataset_publication_ready=False,
        release_candidate_eligible=False,
        role=role,
        receipt_path=str(snapshot.receipt_path),
        receipt_sha256=snapshot.receipt_sha256,
        selection_sha256=snapshot.selection_sha256,
        semantic_validation_jsonl_path=str(snapshot.semantic_validation_jsonl_path),
        semantic_validation_jsonl_sha256=snapshot.semantic_validation_jsonl_sha256,
        semantic_validation_jsonl_domain_sha256=(
            snapshot.semantic_validation_jsonl_domain_sha256
        ),
        manifests=tuple(str(item.path) for item in manifests),
        manifest_sha256=tuple(item.sha256 for item in manifests),
        manifest_records=tuple(item.records for item in manifests),
        manifest_payloads=tuple(item.payload for item in manifests),
        **provider_options,
    )


__all__ = [
    "BOOTSTRAP_PROVENANCE_CLASS",
    "BOOTSTRAP_RECEIPT_TYPE",
    "BOOTSTRAP_RECORDS_PER_MANIFEST",
    "BOOTSTRAP_TOTAL_RECORDS",
    "BootstrapReceiptSnapshot",
    "create_bootstrap_role_provider",
    "inspect_bootstrap_roles",
]
