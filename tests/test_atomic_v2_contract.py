import hashlib
import json

import pytest

from atomic_v2.contract import (
    ARCHITECTURE_HASH,
    CONTRACT_SOURCE_BLOB,
    CONTRACT_SOURCE_COMMIT,
    CONTRACT_SOURCE_SHA256,
    FEATURE_DIMENSIONS,
    FEATURE_HASH,
    FEATURE_TRANSFORMER_HASH,
    FILE_VERSION,
    NETWORK_HASH,
    OFFICIAL_TRAINER_COMMIT,
    ContractError,
    contract_bytes,
    load_contract,
    validate_contract,
)


def test_vendored_contract_is_the_authenticated_engine_document():
    raw = contract_bytes()

    assert hashlib.sha256(raw).hexdigest() == CONTRACT_SOURCE_SHA256
    assert CONTRACT_SOURCE_SHA256 == (
        "70ea8da4cdd2f209fafc25f9419d3b3f226711b00701bf2fc02587dba65b82d7"
    )
    assert CONTRACT_SOURCE_COMMIT == "0818886d77328fe25850a3187e6460adaa980316"
    assert CONTRACT_SOURCE_BLOB == "8a8003cbf5f7b97b50470de898ae16074aad7bca"
    assert OFFICIAL_TRAINER_COMMIT == "b8512291deb4cd18afa67003bb6bc53dd522cbf0"
    assert raw.endswith(b"\n")


def test_atomic_v2_contract_freezes_identity_and_physical_topology():
    document = load_contract()
    backend = validate_contract(document)

    assert FILE_VERSION == 0xA70C0002
    assert FEATURE_HASH == 0x5F234CB8
    assert FEATURE_TRANSFORMER_HASH == 0x5F2344B8
    assert ARCHITECTURE_HASH == 0x63337116
    assert NETWORK_HASH == 0x3C1035AE
    assert FEATURE_DIMENSIONS == 45056
    assert backend["feature_set"] == "HalfKAv2Atomic"
    assert backend["accumulator_dimensions_per_perspective"] == 1024
    assert backend["pairwise_multiply"] == {
        "input_dimensions_per_perspective": 1024,
        "half_dimensions": 512,
        "output_dimensions_per_perspective": 512,
        "concatenated_output_dimensions": 1024,
    }
    assert backend["topology"]["fc0"] == {
        "input_dimensions": 1024,
        "output_dimensions": 32,
    }
    assert backend["topology"]["fc1"] == {
        "input_dimensions": 64,
        "output_dimensions": 32,
    }
    assert backend["topology"]["fc2"] == {
        "input_dimensions": 128,
        "output_dimensions": 1,
    }
    assert backend["topology"]["fc0_skip_indices"] == [30, 31]
    assert backend["topology"]["fc0_skip_coefficients"] == [1, -1]


def test_contract_validation_is_fail_closed():
    document = json.loads(contract_bytes())
    document["backends"]["atomic-nnue-v2"]["network_hash"] = "0x00000000"

    with pytest.raises(ContractError, match="network_hash"):
        validate_contract(document)
