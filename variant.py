RANKS = 8
FILES = 8
SQUARES = RANKS * FILES
KING_SQUARES = RANKS * FILES
PIECE_TYPES = 9
PIECES = 2 * PIECE_TYPES
USE_POCKETS = True
POCKETS = 2 * FILES if USE_POCKETS else 0

PIECE_VALUES = {
  1: 825,
  2: 1276,
  3: 660,
  4: 1800,
  5: 90,
  6: 720,
  7: 1550,
  8: 351,
}
