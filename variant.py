RANKS = 8
FILES = 8
SQUARES = RANKS * FILES
KING_SQUARES = 64
PIECE_TYPES = 6
PIECES = 2 * PIECE_TYPES
USE_POCKETS = False
POCKETS = 2 * FILES if USE_POCKETS else 0

PIECE_VALUES = {
  1: 781,
  2: 825,
  3: 1276,
  4: 2538,
  5: 204,
}
