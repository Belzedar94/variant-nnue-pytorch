"""Authenticated campaign-file seam and strict AtomicNNUEV3 tensor batches.

H9.3l-a owns publication validation.  The trust anchor here is the exact
SHA-256 of a strict receipt file, delivered out-of-process by the authenticated
controller/CAS.  Python objects are not authentication capabilities: arbitrary
code in this process is inside the trust boundary.  Every provider creation
re-reads the receipt, campaign and ordered manifests from stable file
descriptors before the factory sees an immutable byte snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Union

import torch

from .contract import (
    BLAST_RING_MAX_ACTIVE,
    CAMPAIGN_SCHEMA_SHA256,
    CAPTURE_PAIR_MAX_ACTIVE,
    FEATURE_SCHEMA_SHA256,
    HM_MAX_ACTIVE,
    HM_ROWS_PER_BUCKET,
    KING_BLAST_EP_MAX_ACTIVE,
    PUBLICATION_CONTRACT_COMMIT,
    PUBLICATION_SCHEMA_SHA256,
    PUBLICATION_VALIDATOR_CONTRACT,
    SLICES,
)
from .indices import Perspective, make_joint_orientation, network_bucket


ATOMIC_BIN_V2_MANIFEST_SCHEMA_SHA256 = (
    "83d63922df3ac4a0c81a21ec9d9fd9e180efe50f26efee62fe01710e09da5b42"
)
MAX_CAMPAIGN_BYTES = 16 * 1024 * 1024
MAX_RECEIPT_BYTES = 64 * 1024
PUBLICATION_RECEIPT_FORMAT = "atomic-v3-publication-validation-receipt-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BASENAME_RE = re.compile(r'^(?!\.{1,2}$)[^/\\:\x00<>"|?*]+$')
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class DatasetContractError(ValueError):
    pass


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DatasetContractError(f"{name} must be lowercase SHA-256 hex")
    return value


def _require_uint_string(name: str, value: object, *, positive: bool = False) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise DatasetContractError(f"{name} must be a canonical decimal string")
    if value != "0" and value.startswith("0"):
        raise DatasetContractError(f"{name} has a leading zero")
    number = int(value)
    if number > (1 << 64) - 1 or (positive and number == 0):
        raise DatasetContractError(f"{name} is outside its uint64 domain")
    return number


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise DatasetContractError(f"duplicate JSON property: {key}")
        document[key] = value
    return document


def _reject_json_constant(value: str) -> None:
    raise DatasetContractError(f"nonstandard JSON constant is forbidden: {value}")


def _require_path(name: str, value: object) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{name} must be a filesystem path")
    return Path(value)


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _is_link_or_reparse(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    return stat.S_ISLNK(value.st_mode) or bool(attributes & _REPARSE_POINT)


def _reject_linked_path_components(path: Path) -> None:
    """Reject symlinks and Windows reparse points in the complete path."""

    absolute = Path(os.path.abspath(str(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise DatasetContractError(
                f"cannot stat authenticated path component {current}: {error}"
            ) from error
        if _is_link_or_reparse(metadata):
            raise DatasetContractError(
                f"symbolic links and reparse points are forbidden: {current}"
            )


@dataclass(frozen=True)
class _AuthenticatedFileSnapshot:
    """Immutable bytes and lexical label captured by one authenticated read.

    ``path`` is an absolute lexical provenance label, not later path authority.
    It is fixed before opening and never replaced by a post-authentication
    ``resolve()`` whose result could point through a raced symlink or parent.
    """

    path: Path
    identity: tuple[int, int, int, int]
    sha256: str
    payload: bytes


def _read_regular_authenticated(
    path: Path,
    expected_sha256: str,
    *,
    expected_bytes: Optional[int] = None,
    maximum: Optional[int] = None,
) -> _AuthenticatedFileSnapshot:
    expected_sha256 = _require_sha256("expected_sha256", expected_sha256)
    # Freeze the non-authoritative provenance label before any filesystem I/O.
    # Never call resolve() after authentication: an attacker could swap the
    # just-read path to a symlink between those two operations.
    path = Path(os.path.abspath(str(path)))
    _reject_linked_path_components(path)
    try:
        path_before = os.lstat(path)
    except OSError as error:
        raise DatasetContractError(f"cannot stat authenticated artifact {path}: {error}") from error
    if _is_link_or_reparse(path_before):
        raise DatasetContractError(f"symbolic links and reparse points are forbidden: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise DatasetContractError(f"cannot open authenticated artifact {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DatasetContractError(f"artifact is not a regular file: {path}")
        if _file_identity(path_before) != _file_identity(before):
            raise DatasetContractError(f"artifact path changed before authentication: {path}")
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise DatasetContractError(
                f"artifact byte count mismatch for {path.name}: {before.st_size} != {expected_bytes}"
            )
        if maximum is not None and before.st_size > maximum:
            raise DatasetContractError(f"artifact exceeds byte limit: {path}")
        chunks: list[bytes] = []
        bytes_read = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            bytes_read += len(block)
            if maximum is not None and bytes_read > maximum:
                raise DatasetContractError(f"artifact exceeds byte limit while reading: {path}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _file_identity(before) != _file_identity(after):
        raise DatasetContractError(f"artifact changed while authenticating: {path}")
    try:
        path_after = os.lstat(path)
    except OSError as error:
        raise DatasetContractError(f"artifact path vanished while authenticating: {path}") from error
    _reject_linked_path_components(path)
    if _is_link_or_reparse(path_after) or _file_identity(path_after) != _file_identity(after):
        raise DatasetContractError(f"artifact path changed while authenticating: {path}")
    payload = b"".join(chunks)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise DatasetContractError(
            f"artifact SHA-256 mismatch for {path.name}: expected {expected_sha256}, got {actual}"
        )
    return _AuthenticatedFileSnapshot(
        path=path,
        identity=_file_identity(after),
        sha256=actual,
        payload=payload,
    )


@dataclass(frozen=True)
class PublicationReceiptSnapshot:
    """Strict receipt values authenticated by an external expected SHA-256."""

    receipt_path: Path
    receipt_sha256: str
    validator_contract: str
    publication_contract_commit: str
    publication_schema_sha256: tuple[tuple[str, str], ...]
    campaign_schema_sha256: str
    campaign_sha256: str
    collection_sha256: str
    feature_schema_sha256: str
    producer_attestation_sha256: str
    semantic_audit_sha256: str
    reachability_attestation_sha256: str
    dataset_publication_ready: bool


_RECEIPT_KEYS = {
    "receipt_format",
    "validator_contract",
    "publication_contract_commit",
    "publication_schema_sha256",
    "campaign_schema_sha256",
    "campaign_sha256",
    "collection_sha256",
    "feature_schema_sha256",
    "producer_attestation_sha256",
    "semantic_audit_sha256",
    "reachability_attestation_sha256",
    "dataset_publication_ready",
}


def _strict_json_document(payload: bytes, label: str) -> dict[str, Any]:
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except DatasetContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetContractError(f"{label} is not strict UTF-8 JSON: {error}") from error
    if not isinstance(document, dict):
        raise DatasetContractError(f"{label} root must be an object")
    return document


def _load_publication_receipt(
    receipt_path: Union[str, Path], expected_receipt_sha256: str
) -> PublicationReceiptSnapshot:
    path = _require_path("receipt_path", receipt_path)
    expected = _require_sha256("expected_receipt_sha256", expected_receipt_sha256)
    artifact = _read_regular_authenticated(path, expected, maximum=MAX_RECEIPT_BYTES)
    raw = artifact.payload
    document = _strict_json_document(raw, "publication receipt")
    if set(document) != _RECEIPT_KEYS:
        raise DatasetContractError("publication receipt fields differ from the frozen contract")
    if document["receipt_format"] != PUBLICATION_RECEIPT_FORMAT:
        raise DatasetContractError("unrecognized publication receipt format")
    if document["validator_contract"] != PUBLICATION_VALIDATOR_CONTRACT:
        raise DatasetContractError("unrecognized publication validator contract")
    if document["publication_contract_commit"] != PUBLICATION_CONTRACT_COMMIT:
        raise DatasetContractError("publication receipt commit is not the merged H9.3l-a contract")
    schemas = document["publication_schema_sha256"]
    if not isinstance(schemas, dict) or schemas != dict(PUBLICATION_SCHEMA_SHA256):
        raise DatasetContractError("publication receipt schema set differs from merged H9.3l-a")
    if document["campaign_schema_sha256"] != CAMPAIGN_SCHEMA_SHA256:
        raise DatasetContractError("campaign schema is not the release-pinned H9.3l-a schema")
    if document["feature_schema_sha256"] != FEATURE_SCHEMA_SHA256:
        raise DatasetContractError("receipt feature schema does not match AtomicNNUEV3")
    digest_fields = (
        "campaign_sha256",
        "collection_sha256",
        "producer_attestation_sha256",
        "semantic_audit_sha256",
        "reachability_attestation_sha256",
    )
    for field_name in digest_fields:
        _require_sha256(field_name, document[field_name])
    if document["dataset_publication_ready"] is not True:
        raise DatasetContractError("campaign is not dataset-publication-ready")
    return PublicationReceiptSnapshot(
        receipt_path=artifact.path,
        receipt_sha256=expected,
        validator_contract=document["validator_contract"],
        publication_contract_commit=document["publication_contract_commit"],
        publication_schema_sha256=tuple(sorted(schemas.items())),
        campaign_schema_sha256=document["campaign_schema_sha256"],
        campaign_sha256=document["campaign_sha256"],
        collection_sha256=document["collection_sha256"],
        feature_schema_sha256=document["feature_schema_sha256"],
        producer_attestation_sha256=document["producer_attestation_sha256"],
        semantic_audit_sha256=document["semantic_audit_sha256"],
        reachability_attestation_sha256=document["reachability_attestation_sha256"],
        dataset_publication_ready=True,
    )


@dataclass(frozen=True)
class RoleManifest:
    chunk_index: int
    first_record: int
    records: int
    path: Path
    sha256: str
    payload: bytes


@dataclass(frozen=True)
class CampaignInspectionSnapshot:
    """Fresh authenticated bytes for inspection, never an authority token."""

    receipt: PublicationReceiptSnapshot
    campaign_path: Path
    campaign_payload: bytes
    campaign_id: str
    train: tuple[RoleManifest, ...]
    validation: tuple[RoleManifest, ...]

    def manifests(self, role: Literal["train", "validation"]) -> tuple[RoleManifest, ...]:
        if role == "train":
            return self.train
        if role == "validation":
            return self.validation
        raise DatasetContractError("role must be exactly 'train' or 'validation'")


def _artifact_manifest(
    root: Path, descriptor: object, *, chunk_index: int, first_record: int, records: int
) -> RoleManifest:
    if not isinstance(descriptor, Mapping):
        raise DatasetContractError("campaign role manifest descriptor must be an object")
    required = {"file", "bytes", "sha256", "schema_sha256"}
    if set(descriptor) != required:
        raise DatasetContractError("campaign role manifest descriptor fields differ")
    filename = descriptor["file"]
    if not isinstance(filename, str) or _BASENAME_RE.fullmatch(filename) is None:
        raise DatasetContractError("campaign role manifest file is not a safe basename")
    byte_count = _require_uint_string("manifest.bytes", descriptor["bytes"], positive=True)
    digest = _require_sha256("manifest.sha256", descriptor["sha256"])
    if descriptor["schema_sha256"] != ATOMIC_BIN_V2_MANIFEST_SCHEMA_SHA256:
        raise DatasetContractError("campaign role does not reference the pinned Atomic BIN V2 manifest schema")
    path = root / filename
    artifact = _read_regular_authenticated(
        path, digest, expected_bytes=byte_count, maximum=MAX_CAMPAIGN_BYTES
    )
    return RoleManifest(
        chunk_index,
        first_record,
        records,
        artifact.path,
        digest,
        artifact.payload,
    )


def _authenticate_campaign_roles(
    campaign_path: Union[str, Path],
    receipt_path: Union[str, Path],
    expected_receipt_sha256: str,
) -> CampaignInspectionSnapshot:
    receipt = _load_publication_receipt(receipt_path, expected_receipt_sha256)
    supplied_path = _require_path("campaign_path", campaign_path)
    campaign_artifact = _read_regular_authenticated(
        supplied_path, receipt.campaign_sha256, maximum=MAX_CAMPAIGN_BYTES
    )
    raw = campaign_artifact.payload
    path = campaign_artifact.path
    document = _strict_json_document(raw, "campaign")
    if document.get("schema_version") != 1 or document.get("status") != "completed":
        raise DatasetContractError("campaign is not a completed schema_version 1 document")
    if document.get("collection_sha256") != receipt.collection_sha256:
        raise DatasetContractError("campaign collection hash does not match validation receipt")
    schemas = document.get("schemas")
    if not isinstance(schemas, Mapping) or schemas.get("feature") != FEATURE_SCHEMA_SHA256:
        raise DatasetContractError("campaign does not bind the frozen AtomicNNUEV3 feature schema")
    campaign_id = document.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise DatasetContractError("campaign_id is missing")
    chunks = document.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise DatasetContractError("campaign chunks must be a non-empty array")

    roles: dict[str, list[RoleManifest]] = {"train": [], "validation": []}
    expected_chunk = None
    expected_offsets = {"train": 0, "validation": 0}
    manifest_names: set[str] = set()
    manifest_hashes: set[str] = set()
    for offset, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping):
            raise DatasetContractError("campaign chunk must be an object")
        chunk_index = _require_uint_string(f"chunks[{offset}].index", chunk.get("index"))
        if expected_chunk is None:
            expected_chunk = chunk_index
        if chunk_index != expected_chunk:
            raise DatasetContractError("campaign chunk indices are not contiguous")
        expected_chunk += 1
        for role in ("train", "validation"):
            partition = chunk.get(role)
            if not isinstance(partition, Mapping):
                raise DatasetContractError(f"chunk {chunk_index} is missing {role} partition")
            first_record = _require_uint_string(
                f"chunks[{offset}].{role}.first_record", partition.get("first_record")
            )
            records = _require_uint_string(
                f"chunks[{offset}].{role}.records", partition.get("records"), positive=True
            )
            if first_record != expected_offsets[role]:
                raise DatasetContractError(f"campaign {role} record offsets are not contiguous")
            expected_offsets[role] += records
            manifest = _artifact_manifest(
                path.parent,
                partition.get("manifest"),
                chunk_index=chunk_index,
                first_record=first_record,
                records=records,
            )
            if manifest.path.name in manifest_names or manifest.sha256 in manifest_hashes:
                raise DatasetContractError("campaign reuses a role manifest artifact")
            manifest_names.add(manifest.path.name)
            manifest_hashes.add(manifest.sha256)
            roles[role].append(manifest)

    totals = document.get("totals")
    if not isinstance(totals, Mapping):
        raise DatasetContractError("campaign totals are missing")
    for role in ("train", "validation"):
        declared = _require_uint_string(
            f"totals.{role}_records", totals.get(f"{role}_records"), positive=True
        )
        if declared != expected_offsets[role]:
            raise DatasetContractError(f"campaign {role} total does not match ordered manifests")
    declared_total = _require_uint_string("totals.records", totals.get("records"), positive=True)
    if declared_total != expected_offsets["train"] + expected_offsets["validation"]:
        raise DatasetContractError("campaign aggregate record total is inconsistent")

    return CampaignInspectionSnapshot(
        receipt=receipt,
        campaign_path=path,
        campaign_payload=raw,
        campaign_id=campaign_id,
        train=tuple(roles["train"]),
        validation=tuple(roles["validation"]),
    )


def inspect_campaign_roles(
    campaign_path: Union[str, Path],
    receipt_path: Union[str, Path],
    expected_receipt_sha256: str,
) -> CampaignInspectionSnapshot:
    """Return a non-authoritative snapshot using a controller/CAS digest."""

    return _authenticate_campaign_roles(
        campaign_path, receipt_path, expected_receipt_sha256
    )


def create_role_provider(
    campaign_path: Union[str, Path],
    receipt_path: Union[str, Path],
    expected_receipt_sha256: str,
    role: Literal["train", "validation"],
    *,
    provider_factory: Callable[..., Any],
    **provider_options: Any,
) -> Any:
    """Reauthenticate from an externally pinned receipt, then invoke V3.

    ``expected_receipt_sha256`` must be supplied by the authenticated external
    controller/CAS.  This function never derives or substitutes that trust
    anchor and has no legacy or loader-autodetection fallback.
    """

    if not callable(provider_factory):
        raise TypeError("provider_factory must be callable")
    snapshot = _authenticate_campaign_roles(
        campaign_path, receipt_path, expected_receipt_sha256
    )
    manifests = snapshot.manifests(role)
    receipt = snapshot.receipt
    return provider_factory(
        backend="atomic-nnue-v3",
        role=role,
        receipt_path=str(receipt.receipt_path),
        receipt_sha256=receipt.receipt_sha256,
        campaign_path=str(snapshot.campaign_path),
        campaign_sha256=receipt.campaign_sha256,
        collection_sha256=receipt.collection_sha256,
        producer_attestation_sha256=receipt.producer_attestation_sha256,
        semantic_audit_sha256=receipt.semantic_audit_sha256,
        reachability_attestation_sha256=receipt.reachability_attestation_sha256,
        manifests=tuple(str(item.path) for item in manifests),
        manifest_sha256=tuple(item.sha256 for item in manifests),
        manifest_records=tuple(item.records for item in manifests),
        manifest_payloads=tuple(item.payload for item in manifests),
        **provider_options,
    )


@dataclass(frozen=True)
class SparseSliceBatch:
    indices: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class PerspectiveBatch:
    own_king_squares: torch.Tensor
    hm: SparseSliceBatch
    capture_pair: SparseSliceBatch
    king_blast_ep: SparseSliceBatch
    blast_ring: SparseSliceBatch


@dataclass(frozen=True)
class AtomicV3Batch:
    side_to_move_white: torch.Tensor
    piece_counts: torch.Tensor
    white: PerspectiveBatch
    black: PerspectiveBatch
    outcome: torch.Tensor
    score: torch.Tensor
    bucket_indices: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.side_to_move_white.shape[0])


def _require_batch_tensor(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise DatasetContractError(f"{name} must be a torch.Tensor")
    return value


def _validate_sparse_slice_layout(
    name: str,
    value: object,
    *,
    batch_size: int,
    max_active: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(value, SparseSliceBatch):
        raise DatasetContractError(f"{name} must be SparseSliceBatch")
    indices = _require_batch_tensor(f"{name}.indices", value.indices)
    values = _require_batch_tensor(f"{name}.values", value.values)
    if indices.ndim != 2 or indices.shape[0] != batch_size or values.shape != indices.shape:
        raise DatasetContractError(
            f"{name} indices/values must have equal [batch, width] shape"
        )
    if indices.shape[1] < 1 or indices.shape[1] > max_active:
        raise DatasetContractError(f"{name} width exceeds frozen active bound")
    if indices.dtype != torch.int32 or values.dtype != torch.float32:
        raise DatasetContractError(f"{name} requires int32 indices and float32 values")
    return indices, values


def validate_batch_layout(batch: AtomicV3Batch) -> AtomicV3Batch:
    """Validate the complete tensor ABI using metadata only.

    This performs no tensor reduction, scalar transfer, or value read, so it is
    safe as the full 16,384-row provider-canary boundary before the bounded
    semantic reconstruction.
    """

    if not isinstance(batch, AtomicV3Batch):
        raise DatasetContractError("AtomicNNUEV3 requires an AtomicV3Batch")
    tensors: list[tuple[str, torch.Tensor]] = []

    def owned(name: str, value: object) -> torch.Tensor:
        tensor = _require_batch_tensor(name, value)
        tensors.append((name, tensor))
        return tensor

    stm = owned("side_to_move_white", batch.side_to_move_white)
    if stm.ndim != 2 or stm.shape[1] != 1 or stm.dtype != torch.float32:
        raise DatasetContractError("side_to_move_white must be float32 [batch, 1]")
    batch_size = int(stm.shape[0])
    if batch_size == 0:
        raise DatasetContractError("AtomicNNUEV3 batch must contain at least one sample")

    piece_counts = owned("piece_counts", batch.piece_counts)
    if piece_counts.shape != (batch_size,) or piece_counts.dtype != torch.long:
        raise DatasetContractError("piece_counts must be torch.long [batch]")
    bucket_indices = owned("bucket_indices", batch.bucket_indices)
    if bucket_indices.shape != (batch_size,) or bucket_indices.dtype != torch.long:
        raise DatasetContractError("bucket_indices must be torch.long [batch]")
    outcome = owned("outcome", batch.outcome)
    score = owned("score", batch.score)
    if outcome.shape != (batch_size, 1) or score.shape != (batch_size, 1):
        raise DatasetContractError("outcome and score must have shape [batch, 1]")
    if outcome.dtype != torch.float32 or score.dtype != torch.float32:
        raise DatasetContractError("outcome and score must use float32")

    slices = (
        ("hm", 0, HM_MAX_ACTIVE),
        ("capture_pair", 1, CAPTURE_PAIR_MAX_ACTIVE),
        ("king_blast_ep", 2, KING_BLAST_EP_MAX_ACTIVE),
        ("blast_ring", 3, BLAST_RING_MAX_ACTIVE),
    )
    for perspective, perspective_batch in (
        (Perspective.WHITE, batch.white),
        (Perspective.BLACK, batch.black),
    ):
        if not isinstance(perspective_batch, PerspectiveBatch):
            raise DatasetContractError("perspective payload must be PerspectiveBatch")
        prefix = perspective.name
        kings = owned(f"{prefix}.own_king_squares", perspective_batch.own_king_squares)
        if kings.shape != (batch_size,) or kings.dtype != torch.long:
            raise DatasetContractError("own_king_squares must be torch.long [batch]")
        for field, _, max_active in slices:
            indices, values = _validate_sparse_slice_layout(
                f"{prefix}.{field}",
                getattr(perspective_batch, field),
                batch_size=batch_size,
                max_active=max_active,
            )
            tensors.extend(
                ((f"{prefix}.{field}.indices", indices), (f"{prefix}.{field}.values", values))
            )

    devices = {tensor.device for _, tensor in tensors}
    if len(devices) != 1:
        raise DatasetContractError("AtomicNNUEV3 batch tensors must share one device")
    return batch


def _validate_sparse_slice(
    name: str,
    value: SparseSliceBatch,
    *,
    batch_size: int,
    dimensions: int,
    max_active: int,
    require_active: bool,
    sorted_unique: bool,
) -> None:
    indices, values = value.indices, value.values
    if not torch.all(torch.isfinite(values)):
        raise DatasetContractError(f"{name} values are not finite")
    for row_index in range(batch_size):
        row = indices[row_index].tolist()
        active: list[int] = []
        empty_seen = False
        for column, index in enumerate(row):
            if index == -1:
                empty_seen = True
                if float(values[row_index, column]) != 0.0:
                    raise DatasetContractError(f"{name} empty sentinel has nonzero value")
                continue
            if empty_seen:
                raise DatasetContractError(f"{name} uses a non-suffix empty sentinel")
            if index < 0 or index >= dimensions:
                raise DatasetContractError(f"{name} index {index} escapes [0, {dimensions - 1}]")
            if float(values[row_index, column]) != 1.0:
                raise DatasetContractError(f"{name} active features must be boolean one")
            active.append(index)
        if require_active and not active:
            raise DatasetContractError(f"{name} must contain an active feature")
        if len(active) != len(set(active)):
            raise DatasetContractError(f"{name} contains duplicate active indices")
        if sorted_unique and active != sorted(active):
            raise DatasetContractError(f"{name} must use canonical ascending relation order")


def _decode_hm_board(
    perspective: Perspective, own_king_square: int, active_hm: Sequence[int]
) -> dict[int, tuple[int, int]]:
    """Reconstruct one absolute-color board from one perspective's HM rows.

    Values are ``(absolute_color, piece_kind)`` where colors use WHITE=0 /
    BLACK=1 and piece kinds use pawn..queen=0..4, king=5.  The two provider
    perspectives must independently reconstruct the identical map.
    """

    orientation = make_joint_orientation(perspective, own_king_square)
    board: dict[int, tuple[int, int]] = {}
    color_counts = [0, 0]
    relative_king_counts = [0, 0]
    own_king_feature_square: Optional[int] = None
    for index in active_hm:
        if index // HM_ROWS_PER_BUCKET != orientation.king_bucket:
            raise DatasetContractError(
                f"{perspective.name} HM feature uses a bucket inconsistent with its own king"
            )
        local = index % HM_ROWS_PER_BUCKET
        plane, oriented_square = divmod(local, 64)
        own = plane == 10 or (plane < 10 and plane % 2 == 0)
        opponent = plane == 11 or (plane < 10 and plane % 2 == 1)
        if not own and not opponent:
            raise DatasetContractError(f"{perspective.name} HM feature has an invalid piece plane")
        is_king = plane >= 10
        piece_kind = 5 if is_king else plane // 2
        absolute_color = int(perspective) if own else 1 - int(perspective)
        square = orientation.orient(oriented_square)
        if square in board:
            raise DatasetContractError(
                f"{perspective.name} HM reconstructs more than one piece on square {square}"
            )
        board[square] = (absolute_color, piece_kind)
        color_counts[absolute_color] += 1
        if color_counts[absolute_color] > 16:
            raise DatasetContractError(
                f"{perspective.name} HM reconstructs more than 16 pieces for one color"
            )
        if is_king:
            relative_king_counts[0 if own else 1] += 1
            if own:
                own_king_feature_square = square

    if relative_king_counts != [1, 1]:
        raise DatasetContractError(
            f"{perspective.name} HM must contain exactly one own and one opponent king"
        )
    if own_king_feature_square != own_king_square:
        raise DatasetContractError(
            f"{perspective.name} own king square differs from its HM king feature"
        )
    return board


def validate_batch(batch: AtomicV3Batch) -> AtomicV3Batch:
    validate_batch_layout(batch)
    stm = batch.side_to_move_white
    batch_size = int(stm.shape[0])
    if not torch.all((stm == 0.0) | (stm == 1.0)):
        raise DatasetContractError("side_to_move_white must contain exact zero/one values")
    if not torch.all(torch.isfinite(batch.outcome)) or not torch.all(torch.isfinite(batch.score)):
        raise DatasetContractError("labels must be finite")
    if not torch.all(
        (batch.outcome == 0.0) | (batch.outcome == 0.5) | (batch.outcome == 1.0)
    ):
        raise DatasetContractError("outcome must be exactly one of 0, 0.5, or 1")
    score64 = batch.score.to(torch.float64)
    if not torch.all(batch.score == torch.trunc(batch.score)) or not torch.all(
        (score64 >= -(1 << 31)) & (score64 <= (1 << 31) - 1)
    ):
        raise DatasetContractError("score must contain integer values in the signed int32 domain")

    for row, count in enumerate(batch.piece_counts.tolist()):
        try:
            expected = network_bucket(count)
        except (TypeError, ValueError) as error:
            raise DatasetContractError(f"piece_counts[{row}] is outside [2, 32]") from error
        if int(batch.bucket_indices[row]) != expected:
            raise DatasetContractError("shared PSQT/dense bucket does not match piece count")

    reconstructed: dict[Perspective, list[dict[int, tuple[int, int]]]] = {
        Perspective.WHITE: [],
        Perspective.BLACK: [],
    }
    for perspective, perspective_batch in (
        (Perspective.WHITE, batch.white),
        (Perspective.BLACK, batch.black),
    ):
        kings = perspective_batch.own_king_squares
        _validate_sparse_slice(
            f"{perspective.name}.hm",
            perspective_batch.hm,
            batch_size=batch_size,
            dimensions=SLICES[0].dimensions,
            max_active=HM_MAX_ACTIVE,
            require_active=True,
            sorted_unique=False,
        )
        _validate_sparse_slice(
            f"{perspective.name}.capture_pair",
            perspective_batch.capture_pair,
            batch_size=batch_size,
            dimensions=SLICES[1].dimensions,
            max_active=CAPTURE_PAIR_MAX_ACTIVE,
            require_active=False,
            sorted_unique=True,
        )
        _validate_sparse_slice(
            f"{perspective.name}.king_blast_ep",
            perspective_batch.king_blast_ep,
            batch_size=batch_size,
            dimensions=SLICES[2].dimensions,
            max_active=KING_BLAST_EP_MAX_ACTIVE,
            require_active=False,
            sorted_unique=True,
        )
        _validate_sparse_slice(
            f"{perspective.name}.blast_ring",
            perspective_batch.blast_ring,
            batch_size=batch_size,
            dimensions=SLICES[3].dimensions,
            max_active=BLAST_RING_MAX_ACTIVE,
            require_active=False,
            sorted_unique=True,
        )
        for row, king in enumerate(kings.tolist()):
            try:
                orientation = make_joint_orientation(perspective, king)
            except (TypeError, ValueError) as error:
                raise DatasetContractError(f"invalid {perspective.name} own king square") from error
            active_hm = [index for index in perspective_batch.hm.indices[row].tolist() if index != -1]
            if len(active_hm) != int(batch.piece_counts[row]):
                raise DatasetContractError(
                    f"{perspective.name} active HM count does not match piece_count"
                )
            reconstructed[perspective].append(
                _decode_hm_board(perspective, king, active_hm)
            )
    for row in range(batch_size):
        if reconstructed[Perspective.WHITE][row] != reconstructed[Perspective.BLACK][row]:
            raise DatasetContractError(
                "WHITE and BLACK HM perspectives do not reconstruct the same board"
            )
    return batch


def _padded_slice(rows: Sequence[Sequence[int]]) -> SparseSliceBatch:
    width = max(1, max((len(row) for row in rows), default=0))
    indices = torch.full((len(rows), width), -1, dtype=torch.int32)
    values = torch.zeros((len(rows), width), dtype=torch.float32)
    for row_index, row in enumerate(rows):
        if row:
            indices[row_index, : len(row)] = torch.tensor(row, dtype=torch.int32)
            values[row_index, : len(row)] = 1.0
    return SparseSliceBatch(indices, values)


def _perspective_from_fixture(samples: Sequence[Mapping[str, Any]], key: str) -> PerspectiveBatch:
    payloads = [sample[key] for sample in samples]
    return PerspectiveBatch(
        own_king_squares=torch.tensor(
            [payload["own_king_square"] for payload in payloads], dtype=torch.long
        ),
        hm=_padded_slice([payload["hm"] for payload in payloads]),
        capture_pair=_padded_slice([payload["capture_pair"] for payload in payloads]),
        king_blast_ep=_padded_slice([payload["king_blast_ep"] for payload in payloads]),
        blast_ring=_padded_slice([payload["blast_ring"] for payload in payloads]),
    )


@dataclass(frozen=True)
class CanonicalFixture:
    campaign_sha256: str
    collection_sha256: str
    roles: Mapping[str, tuple[Mapping[str, Any], ...]]

    def batch(self, role: Literal["train", "validation"]) -> AtomicV3Batch:
        if role not in ("train", "validation"):
            raise DatasetContractError("fixture role must be train or validation")
        samples = self.roles[role]
        for sample in samples:
            if sample.get("side_to_move") not in ("WHITE", "BLACK"):
                raise DatasetContractError("fixture side_to_move must be WHITE or BLACK")
            if not _is_plain_int(sample.get("piece_count")):
                raise DatasetContractError("fixture piece_count must be an integer")
            outcome = sample.get("outcome")
            if isinstance(outcome, bool) or not isinstance(outcome, (int, float)):
                raise DatasetContractError("fixture outcome must be a numeric result enum")
            if float(outcome) not in (0.0, 0.5, 1.0):
                raise DatasetContractError("fixture outcome is outside the contractual domain")
            score = sample.get("score")
            if not _is_plain_int(score) or score < -(1 << 31) or score > (1 << 31) - 1:
                raise DatasetContractError("fixture score must be a signed int32 integer")
        batch = AtomicV3Batch(
            side_to_move_white=torch.tensor(
                [[1.0 if sample["side_to_move"] == "WHITE" else 0.0] for sample in samples],
                dtype=torch.float32,
            ),
            piece_counts=torch.tensor([sample["piece_count"] for sample in samples], dtype=torch.long),
            white=_perspective_from_fixture(samples, "white"),
            black=_perspective_from_fixture(samples, "black"),
            outcome=torch.tensor([[sample["outcome"]] for sample in samples], dtype=torch.float32),
            score=torch.tensor([[sample["score"]] for sample in samples], dtype=torch.float32),
            bucket_indices=torch.tensor(
                [network_bucket(sample["piece_count"]) for sample in samples], dtype=torch.long
            ),
        )
        return validate_batch(batch)


# Updated only when the reviewed fixture bytes intentionally change.
CANONICAL_FIXTURE_SHA256 = "d6a715d838f5dd029a793f5e62552afb34275f0310e720ff61514138eee47276"


def load_canonical_fixture(
    path: Optional[Union[str, Path]] = None,
) -> CanonicalFixture:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "atomic_v3" / "trainer-core-v1.json"
    fixture_artifact = _read_regular_authenticated(
        Path(path), CANONICAL_FIXTURE_SHA256, maximum=1024 * 1024
    )
    raw = fixture_artifact.payload
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except DatasetContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetContractError(f"canonical trainer fixture is malformed: {error}") from error
    if not isinstance(document, dict) or document.get("fixture_schema") != "atomic-v3-trainer-core-v1":
        raise DatasetContractError("canonical trainer fixture schema mismatch")
    if document.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256:
        raise DatasetContractError("canonical trainer fixture feature schema mismatch")
    roles = document.get("roles")
    if not isinstance(roles, dict) or set(roles) != {"train", "validation"}:
        raise DatasetContractError("canonical trainer fixture roles differ")
    normalized: dict[str, tuple[Mapping[str, Any], ...]] = {}
    ids: dict[str, set[str]] = {}
    for role in ("train", "validation"):
        samples = roles[role]
        if not isinstance(samples, list) or not samples:
            raise DatasetContractError(f"canonical {role} fixture is empty")
        if not all(isinstance(sample, dict) for sample in samples):
            raise DatasetContractError(f"canonical {role} sample is not an object")
        role_ids = {sample.get("sample_id") for sample in samples}
        if None in role_ids or len(role_ids) != len(samples):
            raise DatasetContractError(f"canonical {role} sample IDs are not unique")
        ids[role] = role_ids
        normalized[role] = tuple(samples)
    if ids["train"] & ids["validation"]:
        raise DatasetContractError("canonical train and validation IDs overlap")
    fixture = CanonicalFixture(
        campaign_sha256=_require_sha256("fixture.campaign_sha256", document.get("campaign_sha256")),
        collection_sha256=_require_sha256(
            "fixture.collection_sha256", document.get("collection_sha256")
        ),
        roles=normalized,
    )
    fixture.batch("train")
    fixture.batch("validation")
    return fixture
