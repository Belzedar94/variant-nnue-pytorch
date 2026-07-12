RANKS = 8
FILES = 8
SQUARES = RANKS * FILES
KING_SQUARES = RANKS * FILES
PIECE_TYPES = 8  # P N B R Q F J K (spell-chess)
PIECES = 2 * PIECE_TYPES
USE_POCKETS = True  # spells in hand ride the pockets
POCKETS = 2 * FILES if USE_POCKETS else 0

HAS_POTIONS = True  # spell-chess zone/cooldown feature planes

PIECE_VALUES = {
    1 : 126,
    2 : 781,
    3 : 825,
    4 : 1276,
    5 : 2538,
    6 : 0,     # F (spell, never on the board)
    7 : 0,     # J (spell, never on the board)
}