RANKS = 6
FILES = 7
SQUARES = RANKS * FILES
KING_SQUARES = 1
PIECE_TYPES = 1
PIECES = 2 * PIECE_TYPES
USE_POCKETS = False
POCKETS = 2 * FILES if USE_POCKETS else 0

PIECE_VALUES = {
  1: 50,
}
