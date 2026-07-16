import hashlib
import json
import struct

import pytest
import torch

from atomic_v3.contract import BACKEND_KEY
from atomic_v3.dataset import validate_batch
from atomic_v3.executor import ProviderBatch, ResumableBatchProvider
from atomic_v3.native_provider import NativeAtomicV3Provider, NativeProviderError


def _arguments(manifest, *, role="validation", batch_size=8, skipping=0, resume=None):
    payload = manifest.read_bytes()
    return {
        "backend": BACKEND_KEY,
        "role": role,
        "manifests": [str(manifest)],
        "manifest_sha256": [hashlib.sha256(payload).hexdigest()],
        "manifest_records": [1],
        "manifest_payloads": [payload],
        "batch_size": batch_size,
        "random_fen_skipping": skipping,
        "seed": 20260716,
        "native_workers": 1,
        "device": "cpu",
        "resume_state": resume,
    }


def _batch_tensors(batch):
    return (
        batch.side_to_move_white,
        batch.piece_counts,
        batch.white.own_king_squares,
        batch.white.hm.indices,
        batch.white.capture_pair.indices,
        batch.white.king_blast_ep.indices,
        batch.white.blast_ring.indices,
        batch.black.own_king_squares,
        batch.black.hm.indices,
        batch.black.capture_pair.indices,
        batch.black.king_blast_ep.indices,
        batch.black.blast_ring.indices,
        batch.outcome,
        batch.score,
        batch.bucket_indices,
    )


def _assert_same_batch(left, right):
    for left_tensor, right_tensor in zip(_batch_tensors(left), _batch_tensors(right)):
        assert torch.equal(left_tensor, right_tensor)


def _repeat_fixture_records(manifest, count):
    document = json.loads(manifest.read_text(encoding="utf-8"))
    shard = manifest.parent / document["shards"][0]["file"]
    original = shard.read_bytes()
    assert len(original) == 96 + 64
    header = bytearray(original[:96])
    struct.pack_into("<Q", header, 56, count)
    payload = bytes(header) + original[96:] * count
    shard.write_bytes(payload)
    records = str(count)
    options = document["generation"]["options"]
    options["requested_records"] = records
    options["records_per_shard"] = records
    document["statistics"]["records"] = records
    document["statistics"]["draws"] = "0"
    document["shards"][0].update(
        records=records,
        bytes=str(len(payload)),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    manifest.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def test_validation_partial_eof_commit_resume_and_fixed_reset(atomic_v2_manifest):
    provider = NativeAtomicV3Provider(**_arguments(atomic_v2_manifest))
    initial = provider.state_dict()
    assert initial["manifest_index"] == 0
    assert initial["record_index"] == 0
    assert initial["accepted_samples"] == 0

    first = next(provider)
    assert validate_batch(first) is first
    assert first.batch_size == 1
    assert float(first.score[0, 0]) == -123.0
    assert provider.state_dict() == initial  # Fetch is not an optimizer commit.
    provider.commit()
    committed = provider.state_dict()
    assert committed["eof"] is True
    assert committed["manifest_index"] == 1
    assert committed["accepted_samples"] == 1
    assert committed["next_batch_sequence"] == 1
    with pytest.raises(StopIteration):
        next(provider)

    provider.reset_validation()
    repeated = next(provider)
    _assert_same_batch(first, repeated)
    provider.close()

    resumed = NativeAtomicV3Provider(
        **_arguments(atomic_v2_manifest, resume=committed)
    )
    with pytest.raises(StopIteration):
        next(resumed)
    resumed.close()


def test_training_stream_is_cyclic_and_resume_is_exact(atomic_v2_manifest):
    arguments = _arguments(
        atomic_v2_manifest, role="train", batch_size=3, skipping=0
    )
    provider = NativeAtomicV3Provider(**arguments)
    first = next(provider)
    assert first.batch_size == 3
    assert first.score.flatten().tolist() == [-123.0, -123.0, -123.0]
    with pytest.raises(RuntimeError, match="cannot reset"):
        provider.reset_validation()
    provider.commit()
    committed = provider.state_dict()
    assert committed["epoch"] == 3
    assert committed["accepted_samples"] == 3

    expected_next = next(provider)
    # This delivered batch was deliberately not committed.
    assert provider.state_dict() == committed
    provider.close()

    resumed = NativeAtomicV3Provider(**{**arguments, "resume_state": committed})
    actual_next = next(resumed)
    _assert_same_batch(expected_next, actual_next)
    resumed.close()


def test_skip_three_training_resume_restores_exact_selector_state(atomic_v2_manifest):
    manifest = _repeat_fixture_records(atomic_v2_manifest, 64)
    arguments = _arguments(manifest, role="train", batch_size=5, skipping=3)
    arguments["manifest_records"] = [64]
    provider = NativeAtomicV3Provider(**arguments)
    next(provider)
    provider.commit()
    committed = provider.state_dict()

    expected = next(provider)
    provider.commit()
    expected_cursor = provider.state_dict()
    provider.close()

    resumed = NativeAtomicV3Provider(**{**arguments, "resume_state": committed})
    actual = next(resumed)
    _assert_same_batch(expected, actual)
    resumed.commit()
    assert resumed.state_dict() == expected_cursor
    resumed.close()


def test_skip_three_is_deterministic_without_smart_filter(atomic_v2_manifest):
    arguments = _arguments(
        atomic_v2_manifest, role="train", batch_size=5, skipping=3
    )
    left = NativeAtomicV3Provider(**arguments)
    right = NativeAtomicV3Provider(**arguments)
    left_batch = next(left)
    right_batch = next(right)
    _assert_same_batch(left_batch, right_batch)
    left.commit()
    right.commit()
    assert left.state_dict() == right.state_dict()
    # With one raw record per epoch, accepted=5 and epoch>5 proves that skip=3
    # is a deterministic selector; no legacy "smart" filter is introduced.
    assert left.state_dict()["epoch"] > 5
    left.close()
    right.close()


def test_validation_skip_three_repeats_same_accepted_prefix(atomic_v2_manifest):
    manifest = _repeat_fixture_records(atomic_v2_manifest, 64)
    arguments = _arguments(manifest, batch_size=5, skipping=3)
    arguments["manifest_records"] = [64]
    provider = NativeAtomicV3Provider(**arguments)
    first = next(provider)
    assert first.batch_size == 5  # target counts accepted records, not raw reads
    provider.reset_validation()
    repeated = next(provider)
    _assert_same_batch(first, repeated)
    provider.close()


def test_manifest_tamper_fails_at_native_creation(atomic_v2_manifest):
    arguments = _arguments(atomic_v2_manifest)
    atomic_v2_manifest.write_bytes(atomic_v2_manifest.read_bytes() + b" ")
    with pytest.raises(NativeProviderError, match="changed after authentication"):
        NativeAtomicV3Provider(**arguments)


def test_native_batch_is_destroyed_if_python_conversion_raises(
    atomic_v2_manifest, monkeypatch
):
    import atomic_v3.native_provider as module

    provider = NativeAtomicV3Provider(**_arguments(atomic_v2_manifest))

    def fail_conversion(view, device):
        raise RuntimeError("synthetic conversion failure")

    monkeypatch.setattr(module, "_batch_from_view", fail_conversion)
    with pytest.raises(RuntimeError, match="synthetic conversion failure"):
        next(provider)
    # Native progress was delivered but never committed. The Python seam closes
    # the stream so a caller cannot continue and later commit across the gap.
    with pytest.raises(RuntimeError, match="closed"):
        provider.state_dict()
    provider.close()


def test_provider_rejects_role_and_worker_contract(atomic_v2_manifest):
    arguments = _arguments(atomic_v2_manifest)
    with pytest.raises(ValueError, match="cyclic"):
        NativeAtomicV3Provider(**{**arguments, "cyclic": True})
    with pytest.raises(ValueError, match="native_workers"):
        NativeAtomicV3Provider(**{**arguments, "native_workers": 2})

    provider = NativeAtomicV3Provider(**arguments)
    with pytest.raises(NativeProviderError, match="no delivered batch"):
        provider.commit()
    with pytest.raises(RuntimeError, match="closed"):
        next(provider)


def test_commit_requires_newly_delivered_microbatch(atomic_v2_manifest):
    provider = NativeAtomicV3Provider(**_arguments(atomic_v2_manifest))
    next(provider)
    provider.commit()
    with pytest.raises(NativeProviderError, match="no delivered batch"):
        provider.commit()
    with pytest.raises(RuntimeError, match="closed"):
        provider.state_dict()


def test_validation_resume_rejects_nonzero_selector_epoch(atomic_v2_manifest):
    arguments = _arguments(atomic_v2_manifest, skipping=3)
    provider = NativeAtomicV3Provider(**arguments)
    forged = provider.state_dict()
    provider.close()
    forged["epoch"] = 1
    with pytest.raises(NativeProviderError, match="non-cyclic resume epoch must be zero"):
        NativeAtomicV3Provider(**{**arguments, "resume_state": forged})


def test_role_defaults_preserve_historical_skip_policy(atomic_v2_manifest):
    base = _arguments(atomic_v2_manifest)
    base.pop("random_fen_skipping")
    validation = NativeAtomicV3Provider(**base)
    assert validation.random_fen_skipping == 3
    validation.close()
    training = NativeAtomicV3Provider(**{**base, "role": "train"})
    assert training.random_fen_skipping == 3
    training.close()


def test_native_provider_satisfies_resumable_executor_protocol(atomic_v2_manifest):
    provider = NativeAtomicV3Provider(**_arguments(atomic_v2_manifest))
    assert isinstance(provider, ResumableBatchProvider)
    initial = provider.logical_cursor_state()
    delivered = provider.next_batch(provider.batch_size)
    assert isinstance(delivered, ProviderBatch)
    assert delivered.samples == delivered.payload.batch_size == 1
    assert provider.logical_cursor_state() == initial
    provider.commit()
    assert provider.logical_cursor_state()["accepted_samples"] == 1
    provider.restore_logical_cursor(initial)
    assert provider.logical_cursor_state() == initial
    provider.close()


def test_restore_is_transactional_when_native_rejects_cursor(atomic_v2_manifest):
    provider = NativeAtomicV3Provider(**_arguments(atomic_v2_manifest))
    initial = provider.logical_cursor_state()
    forged = dict(initial)
    forged["binding_sha256"] = "f" * 64
    with pytest.raises(NativeProviderError, match="binding"):
        provider.restore_logical_cursor(forged)
    # The original stream was neither destroyed nor advanced by the rejected
    # replacement, and therefore remains usable.
    assert provider.logical_cursor_state() == initial
    assert provider.next_batch(provider.batch_size).samples == 1
    provider.close()


def test_executor_batch_request_mismatch_does_not_fetch(atomic_v2_manifest):
    provider = NativeAtomicV3Provider(**_arguments(atomic_v2_manifest))
    initial = provider.logical_cursor_state()
    with pytest.raises(ValueError, match="configured batch_size"):
        provider.next_batch(provider.batch_size - 1)
    assert provider.logical_cursor_state() == initial
    provider.close()
