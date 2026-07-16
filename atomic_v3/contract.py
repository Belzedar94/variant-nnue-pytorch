"""Frozen AtomicNNUEV3 trainer-side architecture contract.

This module intentionally does not participate in the legacy trainer or in
``atomic_v2`` dispatch.  H9.3l-j needs an importable training graph, not a new
generic feature-set alias, so every public identity is V3-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


ENGINE_CONTRACT_COMMIT = "dde43fc08fb2bd45eec09d3dbe9f6d06845eeb24"
ENGINE_CONTRACT_BLOB = "4ed24d80d07336497947b69a9d9a2ddffef25514"
FEATURE_SCHEMA_SHA256 = "9d3c77a58e5e55ac1bc798dab41977451eb523fce1d6fd3ec3f7c1e574a78750"
OFFICIAL_TRAINER_COMMIT = "b8512291deb4cd18afa67003bb6bc53dd522cbf0"

# H9.3l-a publication boundary.  The trainer accepts an explicit validation
# receipt bearing this schema identity; it never treats a campaign filename as
# authentication by itself.
CAMPAIGN_SCHEMA_SHA256 = "36a86983d63e71e20daa3bcf7a574dfc95abb544974e36c064445e79ad706517"
CAMPAIGN_SCHEMA_BLOB = "41486135e3921f32ebabe0f070af1f2bd3f01b7a"
PUBLICATION_VALIDATOR_CONTRACT = "atomic-v3-distributed-publication-h9.3l-a-v1"
PUBLICATION_CONTRACT_COMMIT = ENGINE_CONTRACT_COMMIT
PUBLICATION_SCHEMA_SHA256 = MappingProxyType({
    "atomic-v3-dataset-campaign-v1.json": CAMPAIGN_SCHEMA_SHA256,
    "atomic-v3-producer-attestation-v1.json": "de55f384fdea56fdb28addd50b78da7e0256b5a8857d5aec856219a3e922193e",
    "atomic-v3-semantic-audit-v1.json": "e1aed04f4291f1ae514ba532b9a4c21fd926e41b7e29b7782a5222a85eda7810",
    "atomic-v3-reachability-attestation-v1.json": "fb1af7130a2fa74be0fadd721db980269e12c89204b627eec63e6074ed3983e8",
    "atomic-v3-training-environment-v1.json": "8e2f9b97183d3deedfbc1d03ac396ace7c069fc369af09db7e1ee693cd59f3d0",
    "atomic-v3-training-run-manifest-v2.json": "7703f038262cd4a69299aeaf3e0bb35c6d3181029fd448f91560807a0507d184",
})

BACKEND_KEY = "atomic-nnue-v3"
BACKEND_NAME = "AtomicNNUEV3"
FEATURE_NAME = "HalfKAv2AtomicHM+CapturePair+KingBlastEP+BlastRing"

FILE_VERSION = 0xA70C0003
FEATURE_HASH = 0xA3FBDBE8
TRANSFORMER_DESCRIPTOR_HASH = 0xCC31067A
FEATURE_TRANSFORMER_HASH = 0x6FCAD592
ARCHITECTURE_HASH = 0x63337116
NETWORK_HASH = 0x0CF9A484

ACCUMULATOR_DIMENSIONS = 1024
PSQT_BUCKETS = 8
LAYER_STACKS = 8

HM_KING_BUCKETS = 32
HM_TRAINING_PLANES = 12
HM_PHYSICAL_PLANES = 11
HM_ROWS_PER_BUCKET = 768
HM_PHYSICAL_ROWS_PER_BUCKET = 704
HM_TRAINING_DIMENSIONS = 24_576
HM_PHYSICAL_DIMENSIONS = 22_528
HM_VIRTUAL_DIMENSIONS = 768
HM_OUTPUT_DIMENSIONS = ACCUMULATOR_DIMENSIONS + PSQT_BUCKETS
HM_MAX_ACTIVE = 32

CAPTURE_PAIR_DIMENSIONS = 40_012
CAPTURE_PAIR_NORMAL_DIMENSIONS = 39_984
CAPTURE_PAIR_GEOMETRY_DIMENSIONS = 3_332
CAPTURE_PAIR_EP_EDGES_PER_RELATION = 14
CAPTURE_PAIR_MAX_ACTIVE = 240

KING_BLAST_EP_DIMENSIONS = 2_304
KING_BLAST_EP_MAX_ACTIVE = 35

BLAST_RING_DIMENSIONS = 10_240
BLAST_RING_MAX_ACTIVE = 240

PHYSICAL_DIMENSIONS = 75_084
TRAINING_DIMENSIONS_EXCLUDING_VIRTUAL = 77_132
TRAINING_PARAMETER_ROWS = 77_900
MAX_ACTIVE_PHYSICAL = 547
MAX_ACTIVE_FACTORIZED = 579

FC0_INPUTS = 1024
FC0_OUTPUTS = 32
FC1_INPUTS = 64
FC1_OUTPUTS = 32
FC2_INPUTS = 128
FC2_OUTPUTS = 1

HM_DESCRIPTOR = (
    "HalfKAv2Atomic_hm|v1|square=A1_0_rank_major|offset=0|"
    "axes=king_bucket_h8_to_e1:32,piece_plane:OWN_P,OPP_P,OWN_N,OPP_N,"
    "OWN_B,OPP_B,OWN_R,OPP_R,OWN_Q,OPP_Q,MERGED_K,square:64|physical=22528|"
    "training_planes=OWN_P,OPP_P,OWN_N,OPP_N,OWN_B,OPP_B,OWN_R,OPP_R,"
    "OWN_Q,OPP_Q,OWN_K,OPP_K|training=24576|virtual=768|"
    "factor=bucket_plus_virtual_all_1032_then_export|"
    "king_merge=opp_then_own_king_square_all_1032|"
    "orientation=per_perspective_black_xor56_mirror_if_pre_h_king_file_lt4_"
    "shared_all_slices|royal=KING|dtype=i16|"
    "wire=i16_sleb_feature_major_output1024_contiguous_canonical_unpermuted_"
    "permute16|psqt=hm_only_i32_sleb_feature_major_bucket8_contiguous_after_relations"
)

CAPTURE_PAIR_DESCRIPTOR = (
    "AtomicCapturePair|v2-compact|square=A1_0_rank_major|offset=22528|"
    "axes=normal:actor_rel_accumulator:OWN,OPP;edge:PAWN84@0,KNIGHT336@84,"
    "BISHOP560@420,ROOK896@980,QUEEN1456@1876;target_enemy_of_actor:PAWN,"
    "KNIGHT,BISHOP,ROOK,QUEEN,KING;normal_local=((actor_rel*3332+edge)*6+target)"
    "@offset0_count39984;ep_tail=(actor_rel*14+ep_ordinal)@offset39984_count28;"
    "ep_edges=OWN_rank5_to6_OPP_rank4_to3_oriented_from_then_center_asc|"
    "physical=40012|physical_index=22528+local|"
    "orientation=per_perspective_black_xor56_mirror_if_pre_h_king_file_lt4_"
    "shared_all_slices|actor_rel=color_only|pawn_edge=OWN_north_OPP_south_no_extra_flip|"
    "occupancy=stop_first_occupied_emit_enemy|pins_checks_self_blast=unfiltered|"
    "promotion=one_pawn_relation_no_choice_expansion_current_piece_types|"
    "ep=validated_stm_geometric_cold_tail_fail_closed_source_for_kbr_ring|"
    "impossible_ep_rows=eliminated_no_holes|order=local_asc_unique|"
    "ownership=caller_owned|thread=pure_reentrant_immutable_position|"
    "king_actor=excluded|pawn_push=excluded|dtype=i8|"
    "wire=i8_raw_signed_twos_feature_major_output1024_contiguous_canonical_"
    "unpermuted_permute8_after_hm|psqt=none"
)

KING_BLAST_EP_DESCRIPTOR = (
    "AtomicKingBlastEP|v1|offset=62540|axes=center:64;actor_rel_accumulator:OWN,OPP;"
    "king_rel_actor:ENEMY_KING_CENTER,ENEMY_KING_N,ENEMY_KING_NE,ENEMY_KING_E,"
    "ENEMY_KING_SE,ENEMY_KING_S,ENEMY_KING_SW,ENEMY_KING_W,ENEMY_KING_NW,"
    "OWN_KING_N,OWN_KING_NE,OWN_KING_E,OWN_KING_SE,OWN_KING_S,OWN_KING_SW,"
    "OWN_KING_W,OWN_KING_NW;class:EN_PASSANT_MARKER|"
    "local=((center*2+actor_rel)*18+class)@0..2303|physical=2304@62540..64843|"
    "orientation=per_perspective_black_xor56_mirror_if_pre_h_king_file_lt4_"
    "shared_all_slices|source=single_exact_unfiltered_cp_emission_including_"
    "validated_geometric_ep|offset=related_king_minus_center_in_joint_frame_exact_dfdr|"
    "activation=boolean_sorted_unique_capture_center_set|ep=landing_center_dedup_"
    "offcenter_pawn_excluded_fail_closed|rectangle=full_no_holes|"
    "error=cp_mapped_empty_no_partial|ownership=caller_owned|"
    "thread=pure_reentrant_immutable_position|max=17x2_plus1_eq35|dtype=i16|"
    "wire=i16_sleb_feature_major_output1024_contiguous_canonical_unpermuted_"
    "permute16_after_capture_pair|psqt=none"
)

BLAST_RING_DESCRIPTOR = (
    "AtomicBlastRing|v1|offset=64844|axes=center:64;actor_rel_accumulator:OWN,OPP;"
    "collateral_rel_accumulator:OWN,OPP;offset:N,NE,E,SE,S,SW,W,NW;class:KNIGHT,"
    "BISHOP,ROOK,QUEEN,ADJACENT_PAWN_SURVIVES|local=((((center*2+actor_rel)*2+"
    "collateral_rel)*8+offset)*5+class)@0..10239|physical=10240@64844..75083|"
    "orientation=per_perspective_black_xor56_mirror_if_pre_h_king_file_lt4_"
    "shared_all_slices|source=single_exact_unfiltered_cp_emission_including_"
    "validated_geometric_ep|group=center_actor_rel_distinct_origins|"
    "offset=collateral_minus_center_in_joint_frame_exact_dfdr|"
    "activation=boolean_sorted_unique_capture_center_union|"
    "origin=exclude_only_single_distinct_origin_group_retain_all_origins_if_multi|"
    "nonpawn=current_NBRQ_explodes|pawn=adjacent_survives_except_single_origin_or_ep_"
    "captured|ep=landing_center_malformed_omitted_normal_preserved|"
    "ep_captured_pawn=oriented_center_minus_own8_or_opp_minus8_always_excluded_even_multi|"
    "kings=separate|rectangle=full_no_holes|error=cp_mapped_empty_no_partial|"
    "ownership=caller_owned|thread=pure_reentrant_immutable_position|max=30x8_eq240|"
    "dtype=i8|wire=i8_raw_signed_twos_feature_major_output1024_contiguous_canonical_"
    "unpermuted_permute8_after_king_blast|psqt=none"
)

# This is the global mixed-wire descriptor frozen by the engine in H9.3g.
# Keep the complete ASCII payload here rather than trusting the precomputed
# hash alone: the trainer serializer introduced in H9.3l-k must emit exactly
# the same contract bytes as C++ and the independent wire oracle.
TRANSFORMER_DESCRIPTOR = (
    "AtomicNNUEV3Transformer|v1|wire=biases:i16_sleb[1024],hm:i16_sleb[22528x1024],"
    "cp:i8_raw[40012x1024],kbr:i16_sleb[2304x1024],ring:i8_raw[10240x1024],hm_psqt:"
    "i32_sleb[22528x8],dense:8x(architecture_hash_u32=0x63337116,sfnnv15)|layout=each_"
    "feature_slice_feature_major_output1024_contiguous;hm_psqt_feature_major_bucket8_"
    "contiguous|sleb=COMPRESSED_LEB128_then_u32_le_byte_count_canonical_signed|file=canonical_"
    "unpermuted|raw_i8=signed_twos_complement|load_permute=biases,hm,kbr:i16_block16;cp,"
    "ring:i8_block8;hm_psqt:none|permute_order=avx512[0,2,4,6,1,3,5,7],avx2_lasx[0,2,1,"
    "3,4,6,5,7],other[0,1,2,3,4,5,6,7]|save=unpermute_copy_inverse_order_no_live_"
    "mutation|psqt=hm_only_same_virtual_factor_coalesce_and_12to11_export|dense_tail=byte_"
    "identical_atomic_v2_sfnnv15_architecture_0x63337116|strict_eof=true"
)


def fnv1a_32(data: bytes) -> int:
    value = 0x811C9DC5
    for byte in data:
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value


def _rotate_left_one(value: int) -> int:
    return ((value << 1) | (value >> 31)) & 0xFFFFFFFF


SLICE_HASHES = (
    fnv1a_32(HM_DESCRIPTOR.encode("ascii")),
    fnv1a_32(CAPTURE_PAIR_DESCRIPTOR.encode("ascii")),
    fnv1a_32(KING_BLAST_EP_DESCRIPTOR.encode("ascii")),
    fnv1a_32(BLAST_RING_DESCRIPTOR.encode("ascii")),
)


def fold_slice_hashes(values: tuple[int, ...] = SLICE_HASHES) -> int:
    folded = 0
    for value in values:
        folded = _rotate_left_one(folded) ^ value
    return folded


@dataclass(frozen=True)
class SliceContract:
    key: str
    dimensions: int
    outputs: int
    integer_bits: int
    max_active: int
    has_psqt: bool = False


SLICES = (
    SliceContract("half_ka_v2_atomic_hm", HM_TRAINING_DIMENSIONS, HM_OUTPUT_DIMENSIONS, 16, HM_MAX_ACTIVE, True),
    SliceContract("atomic_capture_pair", CAPTURE_PAIR_DIMENSIONS, ACCUMULATOR_DIMENSIONS, 8, CAPTURE_PAIR_MAX_ACTIVE),
    SliceContract("atomic_king_blast_ep", KING_BLAST_EP_DIMENSIONS, ACCUMULATOR_DIMENSIONS, 16, KING_BLAST_EP_MAX_ACTIVE),
    SliceContract("atomic_blast_ring", BLAST_RING_DIMENSIONS, ACCUMULATOR_DIMENSIONS, 8, BLAST_RING_MAX_ACTIVE),
)


assert SLICE_HASHES == (0xA34A8666, 0x9AEDB186, 0xF5172BC0, 0x38377946)
assert fold_slice_hashes() == FEATURE_HASH
assert len(TRANSFORMER_DESCRIPTOR.encode("ascii")) == 799
assert fnv1a_32(TRANSFORMER_DESCRIPTOR.encode("ascii")) == TRANSFORMER_DESCRIPTOR_HASH
assert FEATURE_TRANSFORMER_HASH == (FEATURE_HASH ^ 2048 ^ TRANSFORMER_DESCRIPTOR_HASH)
assert NETWORK_HASH == (FEATURE_TRANSFORMER_HASH ^ ARCHITECTURE_HASH)
assert HM_TRAINING_DIMENSIONS + CAPTURE_PAIR_DIMENSIONS + KING_BLAST_EP_DIMENSIONS + BLAST_RING_DIMENSIONS == TRAINING_DIMENSIONS_EXCLUDING_VIRTUAL
assert TRAINING_DIMENSIONS_EXCLUDING_VIRTUAL + HM_VIRTUAL_DIMENSIONS == TRAINING_PARAMETER_ROWS
assert HM_MAX_ACTIVE + CAPTURE_PAIR_MAX_ACTIVE + KING_BLAST_EP_MAX_ACTIVE + BLAST_RING_MAX_ACTIVE == MAX_ACTIVE_PHYSICAL
