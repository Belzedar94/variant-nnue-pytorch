import pytest

from atomic_v3 import BACKEND_KEY, BACKEND_NAME, FEATURE_NAME
from atomic_v3.contract import (
    ACCUMULATOR_DIMENSIONS,
    BLAST_RING_DIMENSIONS,
    CAPTURE_PAIR_DIMENSIONS,
    FEATURE_HASH,
    HM_OUTPUT_DIMENSIONS,
    HM_PHYSICAL_DIMENSIONS,
    HM_TRAINING_DIMENSIONS,
    HM_VIRTUAL_DIMENSIONS,
    KING_BLAST_EP_DIMENSIONS,
    MAX_ACTIVE_FACTORIZED,
    MAX_ACTIVE_PHYSICAL,
    NETWORK_HASH,
    PHYSICAL_DIMENSIONS,
    SLICE_HASHES,
    TRANSFORMER_DESCRIPTOR,
    TRANSFORMER_DESCRIPTOR_HASH,
    TRAINING_PARAMETER_ROWS,
    fnv1a_32,
    fold_slice_hashes,
)
from atomic_v3.indices import (
    ActorRelation,
    CollateralRelation,
    Perspective,
    blast_ring_index,
    capture_pair_ep_index,
    capture_pair_normal_index,
    hm_export_sources,
    hm_physical_index,
    hm_training_index,
    hm_virtual_index,
    king_blast_ep_index,
    make_joint_orientation,
    network_bucket,
    psqt_perspective_difference,
    scale_raw_output,
    trunc_div_toward_zero,
)


def test_v3_identity_and_all_frozen_dimensions_are_exact():
    assert BACKEND_KEY == "atomic-nnue-v3"
    assert BACKEND_NAME == "AtomicNNUEV3"
    assert FEATURE_NAME == "HalfKAv2AtomicHM+CapturePair+KingBlastEP+BlastRing"
    assert (HM_TRAINING_DIMENSIONS, HM_VIRTUAL_DIMENSIONS, HM_PHYSICAL_DIMENSIONS) == (
        24_576,
        768,
        22_528,
    )
    assert HM_OUTPUT_DIMENSIONS == ACCUMULATOR_DIMENSIONS + 8 == 1_032
    assert (CAPTURE_PAIR_DIMENSIONS, KING_BLAST_EP_DIMENSIONS, BLAST_RING_DIMENSIONS) == (
        40_012,
        2_304,
        10_240,
    )
    assert PHYSICAL_DIMENSIONS == 75_084
    assert TRAINING_PARAMETER_ROWS == 77_900
    assert (MAX_ACTIVE_PHYSICAL, MAX_ACTIVE_FACTORIZED) == (547, 579)
    assert SLICE_HASHES == (0xA34A8666, 0x9AEDB186, 0xF5172BC0, 0x38377946)
    assert fold_slice_hashes() == FEATURE_HASH == 0xA3FBDBE8
    assert NETWORK_HASH == 0x0CF9A484


def test_global_transformer_descriptor_is_the_exact_engine_wire_identity():
    encoded = TRANSFORMER_DESCRIPTOR.encode("ascii")
    assert len(encoded) == 799
    assert fnv1a_32(encoded) == TRANSFORMER_DESCRIPTOR_HASH == 0xCC31067A
    assert encoded.startswith(b"AtomicNNUEV3Transformer|v1|")
    assert encoded.endswith(b"|strict_eof=true")


@pytest.mark.parametrize(
    "perspective,king,vertical,horizontal,oriented,bucket",
    [
        (Perspective.WHITE, 0, 0, 7, 7, 28),
        (Perspective.WHITE, 4, 0, 0, 4, 31),
        (Perspective.WHITE, 63, 0, 0, 63, 0),
        (Perspective.BLACK, 56, 56, 7, 7, 28),
        (Perspective.BLACK, 60, 56, 0, 4, 31),
        (Perspective.BLACK, 7, 56, 0, 63, 0),
    ],
)
def test_joint_orientation_is_perspective_local_and_bucket_exact(
    perspective, king, vertical, horizontal, oriented, bucket
):
    value = make_joint_orientation(perspective, king)
    assert (value.vertical_xor, value.horizontal_xor) == (vertical, horizontal)
    assert value.oriented_own_king == oriented
    assert value.king_bucket == bucket
    assert value.orient(king) == oriented


def test_hm_factorization_and_12_to_11_king_merge_boundaries():
    assert hm_training_index(0, 0, 0) == 0
    assert hm_training_index(31, 11, 63) == 24_575
    assert hm_virtual_index(0, 0) == 0
    assert hm_virtual_index(11, 63) == 767
    assert hm_physical_index(31, 10, 63) == 22_527

    # Merged king takes OWN_KING at the oriented own king and OPP_KING elsewhere.
    assert hm_export_sources(31, 10, 4, 4) == (
        hm_training_index(31, 10, 4),
        hm_virtual_index(10, 4),
    )
    assert hm_export_sources(31, 10, 60, 4) == (
        hm_training_index(31, 11, 60),
        hm_virtual_index(11, 60),
    )


def test_relation_index_formulas_hit_every_frozen_endpoint():
    assert capture_pair_normal_index(ActorRelation.OWN, 0, 0) == 0
    assert capture_pair_normal_index(ActorRelation.OPP, 3331, 5) == 39_983
    assert capture_pair_ep_index(ActorRelation.OWN, 0) == 39_984
    assert capture_pair_ep_index(ActorRelation.OPP, 13) == 40_011
    assert king_blast_ep_index(0, ActorRelation.OWN, 0) == 0
    assert king_blast_ep_index(63, ActorRelation.OPP, 17) == 2_303
    assert blast_ring_index(
        0, ActorRelation.OWN, CollateralRelation.OWN, 0, 0
    ) == 0
    assert blast_ring_index(
        63, ActorRelation.OPP, CollateralRelation.OPP, 7, 4
    ) == 10_239


@pytest.mark.parametrize(
    "call",
    [
        lambda: hm_training_index(32, 0, 0),
        lambda: hm_virtual_index(12, 0),
        lambda: capture_pair_normal_index(0, 3332, 0),
        lambda: capture_pair_ep_index(1, 14),
        lambda: king_blast_ep_index(64, 0, 0),
        lambda: blast_ring_index(0, 0, 0, 8, 0),
        lambda: network_bucket(1),
    ],
)
def test_index_oracles_fail_closed_outside_the_domain(call):
    with pytest.raises((TypeError, ValueError)):
        call()


def test_signed_arithmetic_matches_cpp_truncation_not_python_flooring():
    assert [trunc_div_toward_zero(value, 2) for value in (-5, -3, -1, 1, 3, 5)] == [
        -2,
        -1,
        0,
        0,
        1,
        2,
    ]
    assert psqt_perspective_difference(-2, 1) == -1
    assert psqt_perspective_difference(2, -1) == 1
    assert scale_raw_output(-1) == 0
    assert scale_raw_output(-3_665_038_760) == -(1 << 31)
    assert scale_raw_output(3_665_038_759) == (1 << 31) - 1


def test_signed_arithmetic_rejects_bool_and_overflow_inputs():
    with pytest.raises(TypeError):
        trunc_div_toward_zero(True, 2)
    with pytest.raises(ValueError):
        scale_raw_output(3_665_038_760)


@pytest.mark.parametrize(
    "call",
    [
        lambda: make_joint_orientation(True, 4),
        lambda: capture_pair_normal_index(False, 0, 0),
        lambda: capture_pair_ep_index(True, 0),
        lambda: king_blast_ep_index(0, False, 0),
        lambda: blast_ring_index(0, 0, True, 0, 0),
    ],
)
def test_integer_enums_reject_bool_aliases(call):
    with pytest.raises(TypeError, match="integer enum"):
        call()


@pytest.mark.parametrize("count,expected", [(2, 0), (4, 0), (5, 1), (32, 7)])
def test_shared_network_bucket_formula(count, expected):
    assert network_bucket(count) == expected
