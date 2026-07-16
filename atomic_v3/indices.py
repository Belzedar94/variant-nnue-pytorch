"""Pure trainer-side index and signed-arithmetic oracles for AtomicNNUEV3."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Union

from .contract import (
    BLAST_RING_DIMENSIONS,
    CAPTURE_PAIR_DIMENSIONS,
    CAPTURE_PAIR_EP_EDGES_PER_RELATION,
    CAPTURE_PAIR_GEOMETRY_DIMENSIONS,
    CAPTURE_PAIR_NORMAL_DIMENSIONS,
    HM_KING_BUCKETS,
    HM_PHYSICAL_ROWS_PER_BUCKET,
    HM_ROWS_PER_BUCKET,
    KING_BLAST_EP_DIMENSIONS,
)


class Perspective(IntEnum):
    WHITE = 0
    BLACK = 1


class ActorRelation(IntEnum):
    OWN = 0
    OPP = 1


class CollateralRelation(IntEnum):
    OWN = 0
    OPP = 1


def _bounded_integer(name: str, value: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} outside [{minimum}, {maximum}]")
    return value


def _strict_enum(name: str, enum_type: type[IntEnum], value: Union[IntEnum, int]) -> IntEnum:
    # ``bool`` is an ``int`` subclass, so IntEnum(True) silently aliases enum
    # value 1 unless it is rejected before conversion.  Wire/provider enums are
    # integer domains, never truth values.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer enum value")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"{name} is outside the enum domain") from error


@dataclass(frozen=True)
class JointOrientation:
    perspective: Perspective
    own_king: int
    oriented_own_king: int
    vertical_xor: int
    horizontal_xor: int
    king_bucket: int

    def orient(self, square: int) -> int:
        return _bounded_integer("square", square, 0, 63) ^ self.vertical_xor ^ self.horizontal_xor


def make_joint_orientation(
    perspective: Union[Perspective, int], own_king: int
) -> JointOrientation:
    perspective = _strict_enum("perspective", Perspective, perspective)
    own_king = _bounded_integer("own_king", own_king, 0, 63)
    vertical = 56 if perspective is Perspective.BLACK else 0
    pre_horizontal = own_king ^ vertical
    horizontal = 7 if (pre_horizontal & 7) < 4 else 0
    oriented = pre_horizontal ^ horizontal
    rank_index, file_index = divmod(oriented, 8)
    bucket = (7 - rank_index) * 4 + (7 - file_index)
    assert 0 <= bucket < HM_KING_BUCKETS
    return JointOrientation(perspective, own_king, oriented, vertical, horizontal, bucket)


def network_bucket(piece_count: int) -> int:
    piece_count = _bounded_integer("piece_count", piece_count, 2, 32)
    return (piece_count - 1) // 4


def hm_training_index(king_bucket: int, training_plane: int, oriented_square: int) -> int:
    king_bucket = _bounded_integer("king_bucket", king_bucket, 0, 31)
    training_plane = _bounded_integer("training_plane", training_plane, 0, 11)
    oriented_square = _bounded_integer("oriented_square", oriented_square, 0, 63)
    return king_bucket * HM_ROWS_PER_BUCKET + training_plane * 64 + oriented_square


def hm_virtual_index(training_plane: int, oriented_square: int) -> int:
    training_plane = _bounded_integer("training_plane", training_plane, 0, 11)
    oriented_square = _bounded_integer("oriented_square", oriented_square, 0, 63)
    return training_plane * 64 + oriented_square


def hm_physical_index(king_bucket: int, physical_plane: int, oriented_square: int) -> int:
    king_bucket = _bounded_integer("king_bucket", king_bucket, 0, 31)
    physical_plane = _bounded_integer("physical_plane", physical_plane, 0, 10)
    oriented_square = _bounded_integer("oriented_square", oriented_square, 0, 63)
    return king_bucket * HM_PHYSICAL_ROWS_PER_BUCKET + physical_plane * 64 + oriented_square


def hm_export_training_plane(physical_plane: int, oriented_square: int, oriented_own_king: int) -> int:
    """Map the frozen 11-plane wire row back to its factorized 12-plane row."""

    physical_plane = _bounded_integer("physical_plane", physical_plane, 0, 10)
    oriented_square = _bounded_integer("oriented_square", oriented_square, 0, 63)
    oriented_own_king = _bounded_integer("oriented_own_king", oriented_own_king, 0, 63)
    if physical_plane < 10:
        return physical_plane
    return 10 if oriented_square == oriented_own_king else 11


def hm_export_sources(
    king_bucket: int, physical_plane: int, oriented_square: int, oriented_own_king: int
) -> tuple[int, int]:
    plane = hm_export_training_plane(physical_plane, oriented_square, oriented_own_king)
    return (
        hm_training_index(king_bucket, plane, oriented_square),
        hm_virtual_index(plane, oriented_square),
    )


def capture_pair_normal_index(
    actor_relation: Union[ActorRelation, int], edge: int, target: int
) -> int:
    actor_relation = _strict_enum("actor_relation", ActorRelation, actor_relation)
    edge = _bounded_integer("edge", edge, 0, CAPTURE_PAIR_GEOMETRY_DIMENSIONS - 1)
    target = _bounded_integer("target", target, 0, 5)
    result = (int(actor_relation) * CAPTURE_PAIR_GEOMETRY_DIMENSIONS + edge) * 6 + target
    assert 0 <= result < CAPTURE_PAIR_NORMAL_DIMENSIONS
    return result


def capture_pair_ep_index(
    actor_relation: Union[ActorRelation, int], ep_ordinal: int
) -> int:
    actor_relation = _strict_enum("actor_relation", ActorRelation, actor_relation)
    ep_ordinal = _bounded_integer(
        "ep_ordinal", ep_ordinal, 0, CAPTURE_PAIR_EP_EDGES_PER_RELATION - 1
    )
    result = CAPTURE_PAIR_NORMAL_DIMENSIONS + int(actor_relation) * 14 + ep_ordinal
    assert 0 <= result < CAPTURE_PAIR_DIMENSIONS
    return result


def king_blast_ep_index(
    center: int, actor_relation: Union[ActorRelation, int], relation_class: int
) -> int:
    center = _bounded_integer("center", center, 0, 63)
    actor_relation = _strict_enum("actor_relation", ActorRelation, actor_relation)
    relation_class = _bounded_integer("relation_class", relation_class, 0, 17)
    result = (center * 2 + int(actor_relation)) * 18 + relation_class
    assert 0 <= result < KING_BLAST_EP_DIMENSIONS
    return result


def blast_ring_index(
    center: int,
    actor_relation: Union[ActorRelation, int],
    collateral_relation: Union[CollateralRelation, int],
    direction: int,
    collateral_class: int,
) -> int:
    center = _bounded_integer("center", center, 0, 63)
    actor_relation = _strict_enum("actor_relation", ActorRelation, actor_relation)
    collateral_relation = _strict_enum(
        "collateral_relation", CollateralRelation, collateral_relation
    )
    direction = _bounded_integer("direction", direction, 0, 7)
    collateral_class = _bounded_integer("collateral_class", collateral_class, 0, 4)
    result = (
        ((((center * 2 + int(actor_relation)) * 2 + int(collateral_relation)) * 8 + direction) * 5)
        + collateral_class
    )
    assert 0 <= result < BLAST_RING_DIMENSIONS
    return result


def trunc_div_toward_zero(numerator: int, denominator: int) -> int:
    if isinstance(numerator, bool) or not isinstance(numerator, int):
        raise TypeError("numerator must be an integer")
    if isinstance(denominator, bool) or not isinstance(denominator, int):
        raise TypeError("denominator must be an integer")
    if denominator == 0:
        raise ZeroDivisionError("division by zero")
    magnitude = abs(numerator) // abs(denominator)
    return -magnitude if (numerator < 0) != (denominator < 0) else magnitude


def scale_raw_output(raw_output: int) -> int:
    raw_output = _bounded_integer("raw_output", raw_output, -3_665_038_760, 3_665_038_759)
    result = trunc_div_toward_zero(raw_output * 9_600, 16_384)
    if result < -(1 << 31) or result > (1 << 31) - 1:
        raise OverflowError("scaled output does not fit signed i32")
    return result


def psqt_perspective_difference(first: int, second: int) -> int:
    for name, value in (("first", first), ("second", second)):
        _bounded_integer(name, value, -(1 << 31), (1 << 31) - 1)
    result = trunc_div_toward_zero(first - second, 2)
    if result < -(1 << 31) or result > (1 << 31) - 1:
        raise OverflowError("PSQT perspective difference does not fit signed i32")
    return result
