import chess
import torch
import feature_block
from collections import OrderedDict
from feature_block import *

import variant

NUM_SQ = variant.SQUARES
NUM_KSQ = variant.KING_SQUARES
NUM_PT_REAL = variant.PIECES - (NUM_KSQ != 1)
NUM_PT_VIRTUAL = variant.PIECES
HAS_POTIONS = getattr(variant, "HAS_POTIONS", False)
POTION_TYPE_NB = 2
POTION_COOLDOWN_BITS = 16
POTION_ZONE_PLANES = POTION_TYPE_NB * 2 if HAS_POTIONS else 0
POTION_ZONE_FEATURES = POTION_ZONE_PLANES * NUM_SQ
POTION_COOLDOWN_FEATURES = POTION_TYPE_NB * 2 * POTION_COOLDOWN_BITS if HAS_POTIONS else 0
NUM_PLANES_BASE = NUM_SQ * NUM_PT_REAL + (NUM_PT_REAL - (NUM_KSQ != 1)) * variant.POCKETS
POTION_ZONE_OFFSET = NUM_PLANES_BASE
POTION_COOLDOWN_OFFSET = POTION_ZONE_OFFSET + POTION_ZONE_FEATURES
NUM_PLANES_REAL = NUM_PLANES_BASE + POTION_ZONE_FEATURES + POTION_COOLDOWN_FEATURES
NUM_PLANES_VIRTUAL = NUM_SQ * NUM_PT_VIRTUAL + (NUM_PT_REAL - (NUM_KSQ != 1)) * variant.POCKETS + POTION_ZONE_FEATURES + POTION_COOLDOWN_FEATURES
NUM_INPUTS = NUM_PLANES_REAL * NUM_KSQ
FEATURE_HASH = 0x7c2d4f9e if HAS_POTIONS else 0x5f234cb8

def orient(is_white_pov: bool, sq: int):
  return sq % variant.FILES + (variant.RANKS - 1 - (sq // variant.FILES)) * variant.FILES if not is_white_pov else sq

def halfka_idx(is_white_pov: bool, king_sq: int, sq: int, piece_type: int, color: bool):
  p_idx = (piece_type - 1) * 2 + (color != is_white_pov)
  if NUM_PT_REAL % 2 and p_idx == NUM_PT_REAL:
    # merge kings into one plane
    p_idx -= 1
  return orient(is_white_pov, sq) + p_idx * NUM_SQ + king_sq * NUM_PLANES_REAL

def halfka_hand_idx(is_white_pov: bool, king_sq: int, handCount: int, piece_type: int, color: bool):
  p_idx = (piece_type - 1) * 2 + (color != is_white_pov)
  return handCount + p_idx * variant.POCKETS + NUM_SQ * NUM_PT_REAL + king_sq * NUM_PLANES_REAL

def halfka_potion_zone_idx(is_white_pov: bool, king_sq: int, sq: int, potion_type: int, owner_color: bool):
  potion_idx = potion_type + POTION_TYPE_NB * (0 if owner_color == chess.WHITE else 1)
  return orient(is_white_pov, sq) + POTION_ZONE_OFFSET + potion_idx * NUM_SQ + king_sq * NUM_PLANES_REAL

def halfka_potion_cooldown_idx(is_white_pov: bool, king_sq: int, bit: int, potion_type: int, owner_color: bool):
  potion_idx = potion_type + POTION_TYPE_NB * (0 if owner_color == chess.WHITE else 1)
  return bit + POTION_COOLDOWN_OFFSET + potion_idx * POTION_COOLDOWN_BITS + king_sq * NUM_PLANES_REAL

def map_king(sq: int):
  # palace squares for Xiangi/Janggi
  if NUM_KSQ == 9 and NUM_KSQ != NUM_SQ:
    if sq > variant.FILES * ((variant.RANKS + 1) // 2):
      # in order to allow unambiguously detecting opposing kings, just return value out of range
      return sq
    # map accessible king squares skipping the gaps
    return (sq - 6 * (sq // variant.FILES) - 3) % NUM_KSQ
  return sq % NUM_KSQ

def halfka_psqts():
  values = [0] * (NUM_PLANES_REAL * NUM_KSQ)

  for ksq in range(NUM_KSQ):
    for s in range(NUM_SQ):
      for pt, val in variant.PIECE_VALUES.items():
        idxw = halfka_idx(True, ksq, s, pt, chess.WHITE)
        idxb = halfka_idx(True, ksq, s, pt, chess.BLACK)
        values[idxw] = val
        values[idxb] = -val
    for i in range(variant.POCKETS):
      for pt, val in variant.PIECE_VALUES.items():
        idxw = halfka_hand_idx(True, ksq, i, pt, chess.WHITE)
        idxb = halfka_hand_idx(True, ksq, i, pt, chess.BLACK)
        values[idxw] = val
        values[idxb] = -val

  return values

class Features(FeatureBlock):
  def __init__(self):
    super(Features, self).__init__('HalfKAv2', FEATURE_HASH, OrderedDict([('HalfKAv2', NUM_PLANES_REAL * NUM_KSQ)]))

  def get_active_features(self, board: chess.Board):
    def piece_features(turn):
      indices = torch.zeros(NUM_PLANES_REAL * NUM_KSQ)
      ksq = orient(turn, board.king(turn))
      for sq, p in board.piece_map().items():
        indices[halfka_idx(turn, ksq, sq, p)] = 1.0
      if HAS_POTIONS:
        potion_zones = getattr(board, "potion_zones", None)
        if isinstance(potion_zones, dict):
          for (owner_color, potion_type), squares in potion_zones.items():
            for sq in squares:
              indices[halfka_potion_zone_idx(turn, ksq, sq, potion_type, owner_color)] = 1.0
        potion_cooldowns = getattr(board, "potion_cooldowns", None)
        if isinstance(potion_cooldowns, dict):
          for (owner_color, potion_type), cooldown in potion_cooldowns.items():
            for bit in range(POTION_COOLDOWN_BITS):
              if cooldown & (1 << bit):
                indices[halfka_potion_cooldown_idx(turn, ksq, bit, potion_type, owner_color)] = 1.0
      return indices
    return (piece_features(chess.WHITE), piece_features(chess.BLACK))

  def get_initial_psqt_features(self):
    return halfka_psqts()

class FactorizedFeatures(FeatureBlock):
  def __init__(self):
    super(FactorizedFeatures, self).__init__('HalfKAv2^', FEATURE_HASH, OrderedDict([('HalfKAv2', NUM_PLANES_REAL * NUM_KSQ), ('A', NUM_PLANES_VIRTUAL)]))

  def get_active_features(self, board: chess.Board):
    raise Exception('Not supported yet, you must use the c++ data loader for factorizer support during training')

  def get_feature_factors(self, idx):
    if idx >= self.num_real_features:
      raise Exception('Feature must be real')

    a_idx = idx % NUM_PLANES_REAL
    k_idx = idx // NUM_PLANES_REAL

    if NUM_PT_VIRTUAL != NUM_PT_REAL:
      if a_idx // NUM_SQ == NUM_PT_REAL - 1 and k_idx != map_king(a_idx % NUM_SQ):
        # is king piece, but not ours
        a_idx += NUM_SQ
      elif a_idx >= NUM_SQ * NUM_PT_REAL:
        # pockets
        a_idx += NUM_SQ

    return [idx, self.get_factor_base_feature('A') + a_idx]

  def get_initial_psqt_features(self):
    return halfka_psqts() + [0] * NUM_PLANES_VIRTUAL

'''
This is used by the features module for discovery of feature blocks.
'''
def get_feature_block_clss():
  return [Features, FactorizedFeatures]
