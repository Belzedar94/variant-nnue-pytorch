"""Minimal authenticated sequential provider for AtomicNNUEV3 bootstrap data.

The native stream is intentionally conservative: one synchronous C++ worker,
ordered manifests, no shuffle and no loader auto-detection.  Training keeps a
single cyclic provider alive across trainer epochs.  Validation creates or
resets a non-cyclic provider with the same seed/cursor so each validation epoch
observes the same deterministic prefix.
"""

from __future__ import annotations

import ctypes
import os
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from .contract import BACKEND_KEY
from .dataset import (
    AtomicV3Batch,
    PerspectiveBatch,
    SparseSliceBatch,
)


ABI_VERSION = 1
STATUS_OK = 0
STATUS_EOF = 1
STATUS_ERROR = 2
DEFAULT_SEED = 20_260_716
DEFAULT_RANDOM_FEN_SKIPPING = 3


class NativeProviderError(RuntimeError):
    pass


class ManifestInputV1(ctypes.Structure):
    _fields_ = [
        ("pathUtf8", ctypes.c_char_p),
        ("pathBytes", ctypes.c_uint64),
        ("payload", ctypes.POINTER(ctypes.c_uint8)),
        ("payloadBytes", ctypes.c_uint64),
        ("sha256Hex", ctypes.c_char_p),
        ("sha256Bytes", ctypes.c_uint64),
        ("expectedRecords", ctypes.c_uint64),
    ]


class ProviderCursorV1(ctypes.Structure):
    _fields_ = [
        ("abiVersion", ctypes.c_uint32),
        ("structSize", ctypes.c_uint32),
        ("bindingSha256", ctypes.c_uint8 * 32),
        ("epoch", ctypes.c_uint64),
        ("manifestIndex", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("recordIndex", ctypes.c_uint64),
        ("acceptedSamples", ctypes.c_uint64),
        ("nextBatchSequence", ctypes.c_uint64),
        ("eof", ctypes.c_uint8),
        ("reserved1", ctypes.c_uint8 * 7),
    ]


class ProviderConfigV1(ctypes.Structure):
    _fields_ = [
        ("abiVersion", ctypes.c_uint32),
        ("structSize", ctypes.c_uint32),
        ("manifests", ctypes.POINTER(ManifestInputV1)),
        ("manifestCount", ctypes.c_uint32),
        ("batchSize", ctypes.c_uint32),
        ("randomFenSkipping", ctypes.c_uint32),
        ("nativeWorkers", ctypes.c_uint32),
        ("seed", ctypes.c_uint64),
        ("cyclic", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 7),
        ("resumeCursor", ctypes.POINTER(ProviderCursorV1)),
    ]


class SparseSliceViewV1(ctypes.Structure):
    _fields_ = [
        ("indices", ctypes.POINTER(ctypes.c_int32)),
        ("values", ctypes.POINTER(ctypes.c_float)),
        ("width", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class PerspectiveViewV1(ctypes.Structure):
    _fields_ = [
        ("ownKingSquares", ctypes.POINTER(ctypes.c_int64)),
        ("hm", SparseSliceViewV1),
        ("capturePair", SparseSliceViewV1),
        ("kingBlastEp", SparseSliceViewV1),
        ("blastRing", SparseSliceViewV1),
    ]


class BatchViewV1(ctypes.Structure):
    _fields_ = [
        ("abiVersion", ctypes.c_uint32),
        ("structSize", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("sideToMoveWhite", ctypes.POINTER(ctypes.c_float)),
        ("pieceCounts", ctypes.POINTER(ctypes.c_int64)),
        ("white", PerspectiveViewV1),
        ("black", PerspectiveViewV1),
        ("outcome", ctypes.POINTER(ctypes.c_float)),
        ("score", ctypes.POINTER(ctypes.c_float)),
        ("bucketIndices", ctypes.POINTER(ctypes.c_int64)),
        ("cursorAfter", ProviderCursorV1),
    ]


if ctypes.sizeof(ProviderCursorV1) != 88:
    raise RuntimeError("Atomic V3 provider cursor ABI has unexpected size")


_BOUND_LIBRARY = None


def _decode_error(value: Optional[bytes], fallback: str) -> str:
    if not value:
        return fallback
    return value.decode("utf-8", errors="replace")


def _bind_library():
    global _BOUND_LIBRARY
    if _BOUND_LIBRARY is not None:
        return _BOUND_LIBRARY
    # Reuse the loader selected by the legacy binding so one process cannot
    # silently load two different native builds from the repository root.
    import nnue_dataset

    library = nnue_dataset.dll
    library.atomic_v3_provider_abi_version.argtypes = []
    library.atomic_v3_provider_abi_version.restype = ctypes.c_uint32
    library.atomic_v3_provider_creation_error.argtypes = []
    library.atomic_v3_provider_creation_error.restype = ctypes.c_char_p
    library.atomic_v3_provider_create.argtypes = [
        ctypes.POINTER(ProviderConfigV1),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.atomic_v3_provider_create.restype = ctypes.c_int32
    library.atomic_v3_provider_destroy.argtypes = [ctypes.c_void_p]
    library.atomic_v3_provider_destroy.restype = None
    library.atomic_v3_provider_fetch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.atomic_v3_provider_fetch.restype = ctypes.c_int32
    library.atomic_v3_provider_error.argtypes = [ctypes.c_void_p]
    library.atomic_v3_provider_error.restype = ctypes.c_char_p
    library.atomic_v3_provider_batch_view.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(BatchViewV1),
    ]
    library.atomic_v3_provider_batch_view.restype = ctypes.c_int32
    library.atomic_v3_provider_destroy_batch.argtypes = [ctypes.c_void_p]
    library.atomic_v3_provider_destroy_batch.restype = None
    library.atomic_v3_provider_commit.argtypes = [ctypes.c_void_p]
    library.atomic_v3_provider_commit.restype = ctypes.c_int32
    library.atomic_v3_provider_committed_cursor.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProviderCursorV1),
    ]
    library.atomic_v3_provider_committed_cursor.restype = ctypes.c_int32
    if library.atomic_v3_provider_abi_version() != ABI_VERSION:
        raise RuntimeError("native Atomic V3 provider ABI version differs")
    _BOUND_LIBRARY = library
    return library


def _plain_int(name: str, value: object, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _cursor_from_state(value: Optional[Mapping[str, Any]]) -> Optional[ProviderCursorV1]:
    if value is None:
        return None
    if not isinstance(value, Mapping) or value.get("provider") != "atomic-v3-sequential-v1":
        raise ValueError("resume_state is not an Atomic V3 sequential provider cursor")
    binding = value.get("binding_sha256")
    if not isinstance(binding, str) or len(binding) != 64:
        raise ValueError("resume_state binding_sha256 is invalid")
    try:
        binding_bytes = bytes.fromhex(binding)
    except ValueError as error:
        raise ValueError("resume_state binding_sha256 is invalid") from error
    cursor = ProviderCursorV1()
    cursor.abiVersion = ABI_VERSION
    cursor.structSize = ctypes.sizeof(ProviderCursorV1)
    cursor.bindingSha256[:] = binding_bytes
    cursor.epoch = _plain_int("resume_state.epoch", value.get("epoch"), 0, 2**64 - 1)
    cursor.manifestIndex = _plain_int(
        "resume_state.manifest_index", value.get("manifest_index"), 0, 2**32 - 1
    )
    cursor.recordIndex = _plain_int(
        "resume_state.record_index", value.get("record_index"), 0, 2**64 - 1
    )
    cursor.acceptedSamples = _plain_int(
        "resume_state.accepted_samples", value.get("accepted_samples"), 0, 2**64 - 1
    )
    cursor.nextBatchSequence = _plain_int(
        "resume_state.next_batch_sequence",
        value.get("next_batch_sequence"),
        0,
        2**64 - 1,
    )
    eof = value.get("eof")
    if not isinstance(eof, bool):
        raise TypeError("resume_state.eof must be bool")
    cursor.eof = int(eof)
    return cursor


def _cursor_state(cursor: ProviderCursorV1) -> dict[str, Any]:
    return {
        "provider": "atomic-v3-sequential-v1",
        "binding_sha256": bytes(cursor.bindingSha256).hex(),
        "epoch": int(cursor.epoch),
        "manifest_index": int(cursor.manifestIndex),
        "record_index": int(cursor.recordIndex),
        "accepted_samples": int(cursor.acceptedSamples),
        "next_batch_sequence": int(cursor.nextBatchSequence),
        "eof": bool(cursor.eof),
    }


def _owned_tensor(pointer, shape, dtype, device: torch.device) -> torch.Tensor:
    array = np.ctypeslib.as_array(pointer, shape=shape).copy()
    tensor = torch.from_numpy(array).to(dtype=dtype)
    if device.type == "cuda":
        return tensor.pin_memory().to(device=device, non_blocking=True)
    return tensor.to(device=device)


def _sparse_batch(view: SparseSliceViewV1, size: int, device: torch.device) -> SparseSliceBatch:
    shape = (size, int(view.width))
    return SparseSliceBatch(
        indices=_owned_tensor(view.indices, shape, torch.int32, device),
        values=_owned_tensor(view.values, shape, torch.float32, device),
    )


def _perspective_batch(
    view: PerspectiveViewV1, size: int, device: torch.device
) -> PerspectiveBatch:
    return PerspectiveBatch(
        own_king_squares=_owned_tensor(
            view.ownKingSquares, (size,), torch.long, device
        ),
        hm=_sparse_batch(view.hm, size, device),
        capture_pair=_sparse_batch(view.capturePair, size, device),
        king_blast_ep=_sparse_batch(view.kingBlastEp, size, device),
        blast_ring=_sparse_batch(view.blastRing, size, device),
    )


def _batch_from_view(view: BatchViewV1, device: torch.device) -> AtomicV3Batch:
    if view.abiVersion != ABI_VERSION or view.structSize != ctypes.sizeof(BatchViewV1):
        raise NativeProviderError("native Atomic V3 batch view ABI differs")
    size = int(view.size)
    if size <= 0:
        raise NativeProviderError("native Atomic V3 provider returned an empty batch")
    return AtomicV3Batch(
        side_to_move_white=_owned_tensor(
            view.sideToMoveWhite, (size, 1), torch.float32, device
        ),
        piece_counts=_owned_tensor(view.pieceCounts, (size,), torch.long, device),
        white=_perspective_batch(view.white, size, device),
        black=_perspective_batch(view.black, size, device),
        outcome=_owned_tensor(view.outcome, (size, 1), torch.float32, device),
        score=_owned_tensor(view.score, (size, 1), torch.float32, device),
        bucket_indices=_owned_tensor(view.bucketIndices, (size,), torch.long, device),
    )


class NativeAtomicV3Provider:
    """One ordered role stream with explicit optimizer-step commit."""

    def __init__(
        self,
        *,
        backend: str,
        role: str,
        manifests: Sequence[str],
        manifest_sha256: Sequence[str],
        manifest_records: Sequence[int],
        manifest_payloads: Sequence[bytes],
        batch_size: int = 128,
        random_fen_skipping: int = DEFAULT_RANDOM_FEN_SKIPPING,
        seed: int = DEFAULT_SEED,
        native_workers: int = 1,
        cyclic: Optional[bool] = None,
        device: object = "cpu",
        resume_state: Optional[Mapping[str, Any]] = None,
        **provenance: Any,
    ) -> None:
        if backend != BACKEND_KEY:
            raise ValueError("native provider accepts only atomic-nnue-v3")
        if role not in ("train", "validation"):
            raise ValueError("role must be exactly train or validation")
        expected_cyclic = role == "train"
        if cyclic is None:
            cyclic = expected_cyclic
        if not isinstance(cyclic, bool) or cyclic != expected_cyclic:
            raise ValueError("train must be cyclic and validation must be non-cyclic")
        self.batch_size = _plain_int("batch_size", batch_size, 1, 1 << 20)
        self.random_fen_skipping = _plain_int(
            "random_fen_skipping", random_fen_skipping, 0, 2**32 - 1
        )
        self.seed = _plain_int("seed", seed, 0, 2**64 - 1)
        self.native_workers = _plain_int("native_workers", native_workers, 1, 1)
        self.role = role
        self.cyclic = cyclic
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Atomic V3 provider device must be cpu or cuda")
        if not (
            len(manifests)
            == len(manifest_sha256)
            == len(manifest_records)
            == len(manifest_payloads)
            > 0
        ):
            raise ValueError("ordered Atomic V3 manifest arrays have different lengths")
        self._manifests = tuple(os.path.abspath(os.fspath(path)) for path in manifests)
        self._manifest_sha256 = tuple(manifest_sha256)
        self._manifest_records = tuple(manifest_records)
        self._manifest_payloads = tuple(manifest_payloads)
        self.provenance = dict(provenance)
        self._library = _bind_library()
        self._stream = ctypes.c_void_p()
        self._create_stream(resume_state)

    def _create_stream(self, resume_state: Optional[Mapping[str, Any]]) -> None:
        path_bytes = [os.fsencode(path) for path in self._manifests]
        hash_bytes = [value.encode("ascii") for value in self._manifest_sha256]
        payload_buffers = [
            (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
            for payload in self._manifest_payloads
        ]
        descriptors = (ManifestInputV1 * len(self._manifests))()
        for index in range(len(self._manifests)):
            descriptors[index] = ManifestInputV1(
                path_bytes[index],
                len(path_bytes[index]),
                ctypes.cast(payload_buffers[index], ctypes.POINTER(ctypes.c_uint8)),
                len(self._manifest_payloads[index]),
                hash_bytes[index],
                len(hash_bytes[index]),
                _plain_int(
                    f"manifest_records[{index}]",
                    self._manifest_records[index],
                    1,
                    2**64 - 1,
                ),
            )
        resume_cursor = _cursor_from_state(resume_state)
        resume_pointer = (
            ctypes.pointer(resume_cursor)
            if resume_cursor is not None
            else ctypes.POINTER(ProviderCursorV1)()
        )
        config = ProviderConfigV1(
            ABI_VERSION,
            ctypes.sizeof(ProviderConfigV1),
            descriptors,
            len(descriptors),
            self.batch_size,
            self.random_fen_skipping,
            self.native_workers,
            self.seed,
            int(self.cyclic),
            (ctypes.c_uint8 * 7)(),
            resume_pointer,
        )
        output = ctypes.c_void_p()
        status = self._library.atomic_v3_provider_create(
            ctypes.byref(config), ctypes.byref(output)
        )
        if status != STATUS_OK or not output.value:
            message = _decode_error(
                self._library.atomic_v3_provider_creation_error(),
                "native Atomic V3 provider creation failed",
            )
            raise NativeProviderError(message)
        self._stream = output

    def __iter__(self):
        return self

    def __next__(self) -> AtomicV3Batch:
        if not self._stream.value:
            raise RuntimeError("Atomic V3 provider is closed")
        batch = ctypes.c_void_p()
        status = self._library.atomic_v3_provider_fetch(
            self._stream, ctypes.byref(batch)
        )
        if status == STATUS_EOF:
            raise StopIteration
        if status != STATUS_OK or not batch.value:
            message = _decode_error(
                self._library.atomic_v3_provider_error(self._stream),
                "native Atomic V3 provider fetch failed",
            )
            self.close()
            raise NativeProviderError(message)
        try:
            view = BatchViewV1()
            status = self._library.atomic_v3_provider_batch_view(
                batch, ctypes.byref(view)
            )
            if status != STATUS_OK:
                raise NativeProviderError("native Atomic V3 batch view failed")
            return _batch_from_view(view, self.device)
        except Exception:
            # A delivered native cursor must never remain usable when Python
            # failed to take ownership of its tensors: continuing could skip
            # an unconsumed microbatch and later commit the gap.
            self.close()
            raise
        finally:
            self._library.atomic_v3_provider_destroy_batch(batch)

    def commit(self) -> None:
        """Commit all delivered microbatches after one successful optimizer step."""

        if not self._stream.value:
            raise RuntimeError("Atomic V3 provider is closed")
        if self._library.atomic_v3_provider_commit(self._stream) != STATUS_OK:
            message = _decode_error(
                self._library.atomic_v3_provider_error(self._stream),
                "native Atomic V3 provider commit failed",
            )
            self.close()
            raise NativeProviderError(message)

    def state_dict(self) -> dict[str, Any]:
        if not self._stream.value:
            raise RuntimeError("Atomic V3 provider is closed")
        cursor = ProviderCursorV1()
        status = self._library.atomic_v3_provider_committed_cursor(
            self._stream, ctypes.byref(cursor)
        )
        if status != STATUS_OK:
            raise NativeProviderError("cannot read Atomic V3 committed cursor")
        return _cursor_state(cursor)

    def reset_validation(self) -> None:
        """Reset validation to its fixed seed and initial non-cyclic cursor."""

        if self.role != "validation":
            raise RuntimeError("training provider must continue across epochs and cannot reset")
        self.close()
        self._create_stream(None)

    def close(self) -> None:
        if getattr(self, "_stream", None) is not None and self._stream.value:
            self._library.atomic_v3_provider_destroy(self._stream)
            self._stream = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def create_native_provider(**kwargs: Any) -> NativeAtomicV3Provider:
    """Factory passed explicitly to ``create_selected_role_provider``."""

    return NativeAtomicV3Provider(**kwargs)


__all__ = [
    "DEFAULT_RANDOM_FEN_SKIPPING",
    "DEFAULT_SEED",
    "NativeAtomicV3Provider",
    "NativeProviderError",
    "create_native_provider",
]
