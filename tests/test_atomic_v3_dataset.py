import dataclasses
import hashlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import atomic_v3.dataset as v3_dataset
from atomic_v3.contract import (
    CAMPAIGN_SCHEMA_SHA256,
    FEATURE_SCHEMA_SHA256,
    PUBLICATION_CONTRACT_COMMIT,
    PUBLICATION_SCHEMA_SHA256,
    PUBLICATION_VALIDATOR_CONTRACT,
)
from atomic_v3.dataset import (
    ATOMIC_BIN_V2_MANIFEST_SCHEMA_SHA256,
    CANONICAL_FIXTURE_SHA256,
    MAX_RECEIPT_BYTES,
    PUBLICATION_RECEIPT_FORMAT,
    AtomicV3Batch,
    DatasetContractError,
    PerspectiveBatch,
    SparseSliceBatch,
    create_role_provider,
    inspect_campaign_roles,
    load_canonical_fixture,
    validate_batch,
)
from atomic_v3.indices import Perspective, hm_training_index, make_joint_orientation


def _artifact(path):
    payload = path.read_bytes()
    return {
        "file": path.name,
        "bytes": str(len(payload)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "schema_sha256": ATOMIC_BIN_V2_MANIFEST_SCHEMA_SHA256,
    }


def _write_json(path, document):
    path.write_bytes(
        (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_receipt(path, document):
    _write_json(path, document)
    return _sha256(path)


def _campaign(tmp_path):
    train = tmp_path / "chunk-0-train.atbin.manifest.json"
    validation = tmp_path / "chunk-0-validation.atbin.manifest.json"
    train.write_bytes(b'{"role":"train"}\n')
    validation.write_bytes(b'{"role":"validation"}\n')
    collection = "2" * 64
    document = {
        "schema_version": 1,
        "campaign_id": "trainer-campaign-fixture",
        "status": "completed",
        "schemas": {"feature": FEATURE_SCHEMA_SHA256},
        "chunks": [
            {
                "index": "0",
                "train": {
                    "first_record": "0",
                    "records": "2",
                    "manifest": _artifact(train),
                },
                "validation": {
                    "first_record": "0",
                    "records": "1",
                    "manifest": _artifact(validation),
                },
            }
        ],
        "totals": {"train_records": "2", "validation_records": "1", "records": "3"},
        "collection_sha256": collection,
    }
    campaign = tmp_path / "campaign.json"
    _write_json(campaign, document)
    receipt_document = {
        "receipt_format": PUBLICATION_RECEIPT_FORMAT,
        "validator_contract": PUBLICATION_VALIDATOR_CONTRACT,
        "publication_contract_commit": PUBLICATION_CONTRACT_COMMIT,
        "publication_schema_sha256": dict(PUBLICATION_SCHEMA_SHA256),
        "campaign_schema_sha256": CAMPAIGN_SCHEMA_SHA256,
        "campaign_sha256": _sha256(campaign),
        "collection_sha256": collection,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "producer_attestation_sha256": "3" * 64,
        "semantic_audit_sha256": "4" * 64,
        "reachability_attestation_sha256": "5" * 64,
        "dataset_publication_ready": True,
    }
    receipt = tmp_path / "publication-receipt.json"
    receipt_sha256 = _write_receipt(receipt, receipt_document)
    return campaign, receipt, receipt_sha256, train, validation, receipt_document


def test_canonical_fixture_preserves_role_and_sample_order_separately():
    fixture = load_canonical_fixture()
    assert [sample["sample_id"] for sample in fixture.roles["train"]] == [
        "train-000",
        "train-001",
    ]
    assert [sample["sample_id"] for sample in fixture.roles["validation"]] == [
        "validation-000"
    ]
    assert fixture.batch("train").batch_size == 2
    assert fixture.batch("validation").batch_size == 1


def test_canonical_fixture_bytes_and_checkout_policy_are_lf_stable():
    fixture_path = Path(__file__).parent / "fixtures" / "atomic_v3" / "trainer-core-v1.json"
    raw = fixture_path.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert hashlib.sha256(raw).hexdigest() == CANONICAL_FIXTURE_SHA256
    attributes = (Path(__file__).parents[1] / ".gitattributes").read_text(encoding="utf-8")
    assert "tests/fixtures/atomic_v3/*.json text eol=lf" in attributes.splitlines()


def test_campaign_requires_receipt_before_opening_any_role_manifest(tmp_path):
    campaign, receipt, _, train, _, receipt_document = _campaign(tmp_path)
    train.unlink()
    rejected = dict(receipt_document)
    rejected["dataset_publication_ready"] = False
    expected_receipt_sha256 = _write_receipt(receipt, rejected)
    with pytest.raises(DatasetContractError, match="not dataset-publication-ready"):
        inspect_campaign_roles(campaign, receipt, expected_receipt_sha256)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("validator_contract", "unknown", "validator contract"),
        ("publication_contract_commit", "0" * 40, "receipt commit"),
        ("campaign_schema_sha256", "0" * 64, "campaign schema"),
        ("feature_schema_sha256", "0" * 64, "feature schema"),
        ("campaign_sha256", "0" * 64, "SHA-256 mismatch"),
        ("collection_sha256", "0" * 64, "collection hash"),
    ],
)
def test_campaign_receipt_mismatches_fail_closed(tmp_path, field, value, match):
    campaign, receipt, _, _, _, receipt_document = _campaign(tmp_path)
    rejected = dict(receipt_document)
    rejected[field] = value
    expected_receipt_sha256 = _write_receipt(receipt, rejected)
    with pytest.raises(DatasetContractError, match=match):
        inspect_campaign_roles(campaign, receipt, expected_receipt_sha256)


def test_publication_receipt_requires_the_complete_merged_schema_set(tmp_path):
    campaign, receipt, _, _, _, receipt_document = _campaign(tmp_path)
    rejected = dict(receipt_document)
    schemas = dict(receipt_document["publication_schema_sha256"])
    schemas.pop("atomic-v3-semantic-audit-v1.json")
    rejected["publication_schema_sha256"] = schemas
    expected_receipt_sha256 = _write_receipt(receipt, rejected)
    with pytest.raises(DatasetContractError, match="schema set differs"):
        inspect_campaign_roles(campaign, receipt, expected_receipt_sha256)


def test_authenticated_roles_are_ordered_and_pass_only_through_explicit_v3_factory(tmp_path):
    campaign_path, receipt, receipt_sha256, train, validation, _ = _campaign(tmp_path)
    snapshot = inspect_campaign_roles(campaign_path, receipt, receipt_sha256)
    assert [item.path for item in snapshot.train] == [train]
    assert [item.path for item in snapshot.validation] == [validation]

    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return "provider"

    assert (
        create_role_provider(
            campaign_path,
            receipt,
            receipt_sha256,
            "train",
            provider_factory=factory,
            batch_size=128,
        )
        == "provider"
    )
    assert calls[0]["backend"] == "atomic-nnue-v3"
    assert calls[0]["role"] == "train"
    assert calls[0]["manifests"] == (str(train),)
    assert calls[0]["manifest_sha256"] == (snapshot.train[0].sha256,)
    assert calls[0]["manifest_records"] == (2,)
    assert calls[0]["manifest_payloads"] == (train.read_bytes(),)
    assert calls[0]["receipt_sha256"] == receipt_sha256
    assert calls[0]["producer_attestation_sha256"] == "3" * 64
    assert calls[0]["semantic_audit_sha256"] == "4" * 64
    assert calls[0]["reachability_attestation_sha256"] == "5" * 64
    assert calls[0]["batch_size"] == 128

    forged_snapshot = dataclasses.replace(snapshot, campaign_id="forged")
    with pytest.raises(TypeError, match="filesystem path"):
        create_role_provider(
            forged_snapshot,
            receipt,
            receipt_sha256,
            "train",
            provider_factory=factory,
        )


def test_no_same_process_capability_or_issuer_is_an_authentication_root():
    exported_names = set(vars(v3_dataset))
    assert not any("capability" in name or "issuer" in name for name in exported_names)
    documentation = " ".join(v3_dataset.__doc__.split())
    assert "arbitrary code in this process is inside the trust boundary" in documentation


def test_role_manifest_is_reauthenticated_after_campaign_validation(tmp_path):
    campaign, receipt, receipt_sha256, train, _, _ = _campaign(tmp_path)
    train.write_bytes(b"changed after publication validation\n")
    with pytest.raises(DatasetContractError, match="byte count mismatch|SHA-256 mismatch"):
        inspect_campaign_roles(campaign, receipt, receipt_sha256)


def test_malformed_or_duplicate_campaign_json_is_rejected_after_exact_hash_authentication(tmp_path):
    campaign, receipt, _, _, _, receipt_document = _campaign(tmp_path)
    campaign.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
    receipt_document = dict(receipt_document)
    receipt_document["campaign_sha256"] = _sha256(campaign)
    receipt_sha256 = _write_receipt(receipt, receipt_document)
    with pytest.raises(DatasetContractError, match="duplicate JSON property"):
        inspect_campaign_roles(campaign, receipt, receipt_sha256)


def test_train_and_validation_cannot_reuse_one_manifest_digest(tmp_path):
    campaign, receipt, _, train, validation, receipt_document = _campaign(tmp_path)
    validation.write_bytes(train.read_bytes())
    document = json.loads(campaign.read_text(encoding="utf-8"))
    document["chunks"][0]["validation"]["manifest"] = _artifact(validation)
    _write_json(campaign, document)
    receipt_document = dict(receipt_document)
    receipt_document["campaign_sha256"] = _sha256(campaign)
    receipt_sha256 = _write_receipt(receipt, receipt_document)
    with pytest.raises(DatasetContractError, match="reuses a role manifest"):
        inspect_campaign_roles(campaign, receipt, receipt_sha256)


@pytest.mark.parametrize("artifact", ["receipt", "campaign", "manifest"])
def test_provider_reauthenticates_every_file_after_an_inspection_snapshot(tmp_path, artifact):
    campaign, receipt, receipt_sha256, train, _, _ = _campaign(tmp_path)
    inspect_campaign_roles(campaign, receipt, receipt_sha256)
    if artifact == "receipt":
        receipt.write_bytes(b"{}\n")
        match = "SHA-256 mismatch"
    elif artifact == "campaign":
        replacement = tmp_path / "replacement-campaign.json"
        replacement.write_bytes(campaign.read_bytes() + b"\n")
        os.replace(replacement, campaign)
        match = "SHA-256 mismatch"
    else:
        train.write_bytes(train.read_bytes() + b"\n")
        match = "byte count mismatch|SHA-256 mismatch"
    calls = []
    with pytest.raises(DatasetContractError, match=match):
        create_role_provider(
            campaign,
            receipt,
            receipt_sha256,
            "train",
            provider_factory=lambda **kwargs: calls.append(kwargs),
        )
    assert calls == []


def test_provider_factory_consumes_the_fresh_manifest_bytes_not_mutable_path_authority(tmp_path):
    campaign, receipt, receipt_sha256, train, _, _ = _campaign(tmp_path)
    authenticated_payload = train.read_bytes()

    def factory(**kwargs):
        train.write_bytes(b"changed after factory entry\n")
        assert kwargs["manifest_payloads"] == (authenticated_payload,)
        assert hashlib.sha256(kwargs["manifest_payloads"][0]).hexdigest() == kwargs[
            "manifest_sha256"
        ][0]
        return "provider"

    assert (
        create_role_provider(
            campaign,
            receipt,
            receipt_sha256,
            "train",
            provider_factory=factory,
        )
        == "provider"
    )


def test_post_authentication_symlink_swap_cannot_retarget_provenance_parent(
    tmp_path, monkeypatch
):
    original = tmp_path / "original"
    alternate = tmp_path / "alternate"
    original.mkdir()
    alternate.mkdir()
    campaign, receipt, receipt_sha256, train, validation, _ = _campaign(original)
    for artifact in (campaign, train, validation):
        (alternate / artifact.name).write_bytes(artifact.read_bytes())

    authenticated_read = v3_dataset._read_regular_authenticated
    swapped = False

    def race_after_authenticated_read(path, *args, **kwargs):
        nonlocal swapped
        snapshot = authenticated_read(path, *args, **kwargs)
        if Path(path) == campaign and not swapped:
            swapped = True
            campaign.unlink()
            try:
                campaign.symlink_to(alternate / campaign.name)
            except OSError as error:
                pytest.skip(f"platform cannot create the symlink race fixture: {error}")
        return snapshot

    monkeypatch.setattr(
        v3_dataset, "_read_regular_authenticated", race_after_authenticated_read
    )
    snapshot = inspect_campaign_roles(campaign, receipt, receipt_sha256)

    # The immutable bytes and lexical path captured by the authenticated read
    # stay paired. A later resolve() must not redirect manifest discovery or
    # provenance labels into the alternate tree.
    assert swapped
    assert snapshot.campaign_path == Path(os.path.abspath(str(campaign)))
    assert snapshot.campaign_path.parent == original
    assert snapshot.train[0].path == train
    assert snapshot.validation[0].path == validation
    assert all(item.path.parent != alternate for item in snapshot.train + snapshot.validation)


def test_publication_receipt_is_strict_sized_duplicate_free_json(tmp_path):
    campaign, receipt, _, _, _, receipt_document = _campaign(tmp_path)
    duplicate = json.dumps(receipt_document)[:-1] + ',"receipt_format":"duplicate"}\n'
    receipt.write_text(duplicate, encoding="utf-8")
    with pytest.raises(DatasetContractError, match="duplicate JSON property"):
        inspect_campaign_roles(campaign, receipt, _sha256(receipt))

    receipt.write_bytes(b'{"receipt_format":NaN}\n')
    with pytest.raises(DatasetContractError, match="nonstandard JSON constant"):
        inspect_campaign_roles(campaign, receipt, _sha256(receipt))

    rejected = dict(receipt_document)
    rejected["extra"] = "not allowed"
    with pytest.raises(DatasetContractError, match="fields differ"):
        inspect_campaign_roles(campaign, receipt, _write_receipt(receipt, rejected))

    rejected = dict(receipt_document)
    rejected["dataset_publication_ready"] = 1
    with pytest.raises(DatasetContractError, match="not dataset-publication-ready"):
        inspect_campaign_roles(campaign, receipt, _write_receipt(receipt, rejected))

    receipt.write_bytes(b" " * (MAX_RECEIPT_BYTES + 1))
    with pytest.raises(DatasetContractError, match="byte limit"):
        inspect_campaign_roles(campaign, receipt, _sha256(receipt))


def test_symlink_and_reparse_receipts_are_rejected(tmp_path):
    campaign, receipt, receipt_sha256, _, _, _ = _campaign(tmp_path)
    link = tmp_path / "receipt-link.json"
    try:
        link.symlink_to(receipt)
    except OSError:
        link = None
    if link is not None:
        with pytest.raises(DatasetContractError, match="symbolic links and reparse points"):
            inspect_campaign_roles(campaign, link, receipt_sha256)

    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        linked_parent = None
    if linked_parent is not None:
        with pytest.raises(DatasetContractError, match="symbolic links and reparse points"):
            inspect_campaign_roles(
                linked_parent / campaign.name,
                linked_parent / receipt.name,
                receipt_sha256,
            )

    synthetic_reparse = SimpleNamespace(
        st_mode=stat.S_IFREG, st_file_attributes=v3_dataset._REPARSE_POINT
    )
    assert v3_dataset._is_link_or_reparse(synthetic_reparse)


def test_orientation_bucket_and_slice_indices_fail_closed_in_batches():
    fixture = load_canonical_fixture()
    batch = fixture.batch("train")

    wrong_hm = batch.white.hm.indices.clone()
    wrong_hm[0, 0] = 0
    wrong_white = dataclasses.replace(
        batch.white, hm=dataclasses.replace(batch.white.hm, indices=wrong_hm)
    )
    with pytest.raises(DatasetContractError, match="HM feature uses a bucket"):
        validate_batch(dataclasses.replace(batch, white=wrong_white))

    wrong_relation = batch.black.capture_pair.indices.clone()
    wrong_relation[0, 0] = 40_012
    wrong_black = dataclasses.replace(
        batch.black,
        capture_pair=dataclasses.replace(batch.black.capture_pair, indices=wrong_relation),
    )
    with pytest.raises(DatasetContractError, match="escapes"):
        validate_batch(dataclasses.replace(batch, black=wrong_black))

    wrong_bucket = batch.bucket_indices.clone()
    wrong_bucket[0] = 7
    with pytest.raises(DatasetContractError, match="bucket does not match"):
        validate_batch(dataclasses.replace(batch, bucket_indices=wrong_bucket))


def test_relation_rows_must_be_sorted_unique_boolean_sets():
    fixture = load_canonical_fixture()
    batch = fixture.batch("train")
    indices = batch.white.capture_pair.indices.clone()
    indices[0, :2] = torch.tensor([5, 0], dtype=torch.int32)
    changed = dataclasses.replace(
        batch.white,
        capture_pair=dataclasses.replace(batch.white.capture_pair, indices=indices),
    )
    with pytest.raises(DatasetContractError, match="canonical ascending"):
        validate_batch(dataclasses.replace(batch, white=changed))


def _hm_rows(perspective, own_king, board):
    orientation = make_joint_orientation(perspective, own_king)
    rows = []
    for square, absolute_color, piece_kind in board:
        own = absolute_color == int(perspective)
        plane = (10 if own else 11) if piece_kind == 5 else piece_kind * 2 + (0 if own else 1)
        rows.append(hm_training_index(orientation.king_bucket, plane, orientation.orient(square)))
    return rows


def _empty_relation(batch_size=1):
    return SparseSliceBatch(
        torch.full((batch_size, 1), -1, dtype=torch.int32),
        torch.zeros((batch_size, 1), dtype=torch.float32),
    )


def _coherent_batch(board, white_king, black_king):
    count = len(board)
    relation = _empty_relation()

    def perspective(which, king):
        indices = torch.tensor([_hm_rows(which, king, board)], dtype=torch.int32)
        return PerspectiveBatch(
            own_king_squares=torch.tensor([king], dtype=torch.long),
            hm=SparseSliceBatch(indices, torch.ones_like(indices, dtype=torch.float32)),
            capture_pair=relation,
            king_blast_ep=relation,
            blast_ring=relation,
        )

    return AtomicV3Batch(
        side_to_move_white=torch.ones((1, 1), dtype=torch.float32),
        piece_counts=torch.tensor([count], dtype=torch.long),
        white=perspective(Perspective.WHITE, white_king),
        black=perspective(Perspective.BLACK, black_king),
        outcome=torch.tensor([[0.5]], dtype=torch.float32),
        score=torch.tensor([[0.0]], dtype=torch.float32),
        bucket_indices=torch.tensor([(count - 1) // 4], dtype=torch.long),
    )


def test_hm_active_count_must_equal_piece_count_in_each_perspective():
    batch = load_canonical_fixture().batch("train")
    indices = batch.white.hm.indices.clone()
    values = batch.white.hm.values.clone()
    indices[0, 7] = -1
    values[0, 7] = 0.0
    white = dataclasses.replace(batch.white, hm=SparseSliceBatch(indices, values))
    with pytest.raises(DatasetContractError, match="active HM count does not match"):
        validate_batch(dataclasses.replace(batch, white=white))


def test_hm_requires_one_king_per_relation_and_own_king_square_identity():
    batch = load_canonical_fixture().batch("train")
    indices = batch.white.hm.indices.clone()
    # Reinterpret WHITE's own king on e1 as an own queen on e1.
    indices[0, 0] -= 2 * 64
    white = dataclasses.replace(
        batch.white, hm=dataclasses.replace(batch.white.hm, indices=indices)
    )
    with pytest.raises(DatasetContractError, match="exactly one own and one opponent king"):
        validate_batch(dataclasses.replace(batch, white=white))


def test_hm_rejects_two_piece_planes_on_one_square():
    batch = load_canonical_fixture().batch("train")
    indices = batch.white.hm.indices.clone()
    # Move the own queen plane onto the own king's oriented square.
    indices[0, 1] += 1
    white = dataclasses.replace(
        batch.white, hm=dataclasses.replace(batch.white.hm, indices=indices)
    )
    with pytest.raises(DatasetContractError, match="more than one piece on square"):
        validate_batch(dataclasses.replace(batch, white=white))


def test_white_and_black_hm_must_reconstruct_the_identical_absolute_board():
    batch = load_canonical_fixture().batch("train")
    indices = batch.black.hm.indices.clone()
    # Move BLACK's view of its own e7 pawn to f7 without changing cardinality.
    indices[0, 7] += 1
    black = dataclasses.replace(
        batch.black, hm=dataclasses.replace(batch.black.hm, indices=indices)
    )
    with pytest.raises(DatasetContractError, match="do not reconstruct the same board"):
        validate_batch(dataclasses.replace(batch, black=black))


def test_hm_rejects_more_than_sixteen_pieces_for_one_color():
    board = [(0, 0, 5)]
    board.extend((square, 0, 0) for square in range(1, 17))
    board.append((63, 1, 5))
    board.extend((square, 1, 0) for square in range(49, 63))
    batch = _coherent_batch(board, white_king=0, black_king=63)
    with pytest.raises(DatasetContractError, match="more than 16 pieces"):
        validate_batch(batch)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("outcome", 0.25, "exactly one of"),
        ("score", 1.5, "integer values"),
        ("score", float(1 << 31), "int32 domain"),
    ],
)
def test_label_domains_are_the_exact_wire_contract(field, value, match):
    batch = load_canonical_fixture().batch("validation")
    changed = torch.tensor([[value]], dtype=torch.float32)
    with pytest.raises(DatasetContractError, match=match):
        validate_batch(dataclasses.replace(batch, **{field: changed}))
