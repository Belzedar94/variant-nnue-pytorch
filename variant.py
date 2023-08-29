RANKS = 6
FILES = 6
SQUARES = RANKS * FILES
KING_SQUARES = 1
PIECE_TYPES = 9
PIECES = 2 * PIECE_TYPES
USE_POCKETS = False
POCKETS = 2 * FILES if USE_POCKETS else 0

PIECE_VALUES = {
  1: 126,
  2: 781,
  3: 825,
  4: 1276,
  5: 2538,
  6: 420,
  7: 2700,
  8: 400,
  9: 700,
}
