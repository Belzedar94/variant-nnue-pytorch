RANKS = 10
FILES = 10
SQUARES = RANKS * FILES
KING_SQUARES = RANKS * FILES
PIECE_TYPES = 8
PIECES = 2 * PIECE_TYPES
USE_POCKETS = True
POCKETS = 2 * FILES if USE_POCKETS else 0

PIECE_VALUES = {
  1: 126,
  2: 781,
  3: 825,
  4: 1276,
  5: 2538,
  6: 2200,
  7: 2300,
}
