"""Authenticated AtomicNNUEV2 contract shared with Atomic-Stockfish."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from typing import Any


CONTRACT_SOURCE_COMMIT = "0818886d77328fe25850a3187e6460adaa980316"
CONTRACT_SOURCE_BLOB = "8a8003cbf5f7b97b50470de898ae16074aad7bca"
CONTRACT_SOURCE_SHA256 = "70ea8da4cdd2f209fafc25f9419d3b3f226711b00701bf2fc02587dba65b82d7"
OFFICIAL_TRAINER_COMMIT = "b8512291deb4cd18afa67003bb6bc53dd522cbf0"

BACKEND_KEY = "atomic-nnue-v2"
BACKEND_NAME = "AtomicNNUEV2"
FEATURE_NAME = "HalfKAv2Atomic"
FILE_VERSION = 0xA70C0002
FEATURE_HASH = 0x5F234CB8
FEATURE_TRANSFORMER_HASH = 0x5F2344B8
ARCHITECTURE_HASH = 0x63337116
NETWORK_HASH = 0x3C1035AE
FEATURE_DIMENSIONS = 45056
ACCUMULATOR_DIMENSIONS = 1024
TRANSFORMED_DIMENSIONS = 1024
PSQT_BUCKETS = 8
LAYER_STACKS = 8
FC0_OUTPUTS = 32
FC1_INPUTS = 64
FC1_OUTPUTS = 32
FC2_INPUTS = 128
FC2_OUTPUTS = 1
MAX_DESCRIPTION_BYTES = 1 << 20


class ContractError(ValueError):
    """The vendored or supplied V2 contract is not the pinned contract."""


_EXPECTED_BACKEND: dict[str, Any] = {
    "file_version": "0xA70C0002",
    "feature_set": FEATURE_NAME,
    "feature_dimensions": FEATURE_DIMENSIONS,
    "accumulator_dimensions_per_perspective": ACCUMULATOR_DIMENSIONS,
    "feature_transformer_output_dimensions": TRANSFORMED_DIMENSIONS,
    "feature_transformer_hash_xor": 2048,
    "feature_set_hash": "0x5F234CB8",
    "feature_transformer_hash": "0x5F2344B8",
    "architecture_hash": "0x63337116",
    "network_hash": "0x3C1035AE",
    "psqt_buckets": PSQT_BUCKETS,
    "layer_stacks": LAYER_STACKS,
    "pairwise_multiply": {
        "input_dimensions_per_perspective": 1024,
        "half_dimensions": 512,
        "output_dimensions_per_perspective": 512,
        "concatenated_output_dimensions": 1024,
    },
    "topology": {
        "fc0": {"input_dimensions": 1024, "output_dimensions": 32},
        "fc1": {"input_dimensions": 64, "output_dimensions": 32},
        "fc2": {"input_dimensions": 128, "output_dimensions": 1},
        "activation_paths": {
            "after_fc0": ["squared-clipped-relu", "clipped-relu"],
            "after_fc1": ["squared-clipped-relu", "clipped-relu"],
        },
        "fc0_skip_indices": [30, 31],
        "fc0_skip_coefficients": [1, -1],
        "output_scaling": {
            "hidden_one": 128,
            "weight_scale_bits": 6,
            "output_scale": 16,
            "network_unit_scale": 600,
            "multiplier": 9600,
            "denominator": 16384,
        },
    },
}


def contract_bytes() -> bytes:
    return files("atomic_v2").joinpath("schema", "atomic-nnue-v2.json").read_bytes()


def load_contract() -> dict[str, Any]:
    raw = contract_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != CONTRACT_SOURCE_SHA256:
        raise ContractError(
            f"contract SHA-256 mismatch: expected {CONTRACT_SOURCE_SHA256}, got {actual_sha}"
        )
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("contract is not valid UTF-8 JSON") from error
    validate_contract(document)
    return document


def _first_difference(expected: Any, actual: Any, path: str) -> str | None:
    if type(expected) is not type(actual):
        return f"{path} has type {type(actual).__name__}, expected {type(expected).__name__}"
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            return f"{path} keys differ (missing={missing}, extra={extra})"
        for key in expected:
            difference = _first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path} length is {len(actual)}, expected {len(expected)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            difference = _first_difference(expected_item, actual_item, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if expected != actual:
        return f"{path} is {actual!r}, expected {expected!r}"
    return None


def validate_contract(document: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ContractError("contract root must be an object")
    expected_root = {
        "schema_version": 1,
        "schema_id": "atomic-nnue-dual-backend-v1",
        "variant": "atomic",
        "byte_order": "little-endian",
    }
    for key, expected in expected_root.items():
        if document.get(key) != expected:
            raise ContractError(f"{key} is {document.get(key)!r}, expected {expected!r}")
    backends = document.get("backends")
    if not isinstance(backends, dict) or BACKEND_KEY not in backends:
        raise ContractError(f"missing backends.{BACKEND_KEY}")
    backend = backends[BACKEND_KEY]
    difference = _first_difference(_EXPECTED_BACKEND, backend, f"backends.{BACKEND_KEY}")
    if difference:
        raise ContractError(difference)

    if FEATURE_TRANSFORMER_HASH != (FEATURE_HASH ^ (ACCUMULATOR_DIMENSIONS * 2)):
        raise ContractError("feature-transformer hash derivation is inconsistent")
    if NETWORK_HASH != (FEATURE_TRANSFORMER_HASH ^ ARCHITECTURE_HASH):
        raise ContractError("network hash derivation is inconsistent")
    return backend


# Fail during import if even the bytes of the package and engine contract drift.
_PINNED_CONTRACT = load_contract()
