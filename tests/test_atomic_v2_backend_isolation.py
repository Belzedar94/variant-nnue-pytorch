import io
import struct

import pytest

import serialize as legacy_serialize
from atomic_v2.checkpoint import CheckpointError, validate_checkpoint_document
from atomic_v2.contract import BACKEND_KEY, FILE_VERSION, NETWORK_HASH
from atomic_v2.serialization import AtomicV2FormatError, read_header, write_header, write_nnue


def test_v1_and_v2_readers_reject_the_other_backend_before_parameters():
    legacy_header = struct.pack("<III", legacy_serialize.VERSION, 0, 0)
    with pytest.raises(AtomicV2FormatError, match="version"):
        read_header(io.BytesIO(legacy_header))

    v2_header = io.BytesIO()
    write_header(v2_header, "isolation")
    with pytest.raises(legacy_serialize.NNUEFormatError, match="Unsupported NNUE version"):
        legacy_serialize.NNUEReader(io.BytesIO(v2_header.getvalue()))


def test_v2_writer_rejects_legacy_or_untyped_objects():
    with pytest.raises(TypeError, match="only AtomicNNUEV2"):
        write_nnue(io.BytesIO(), object())


def test_checkpoint_backend_identity_is_mandatory_and_fail_closed():
    valid = {
        "backend": BACKEND_KEY,
        "file_version": FILE_VERSION,
        "network_hash": NETWORK_HASH,
        "step": 1,
        "model_state": {},
    }
    assert validate_checkpoint_document(valid) is valid

    for field, value in (
        ("backend", "legacy-atomic-v1"),
        ("file_version", 0x7AF32F20),
        ("network_hash", 0),
    ):
        invalid = dict(valid)
        invalid[field] = value
        with pytest.raises(CheckpointError, match=field):
            validate_checkpoint_document(invalid)
