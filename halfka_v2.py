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
HAS_POINTS = getattr(variant, "HAS_POINTS", False)
HAS_CHECKS = getattr(variant, "HAS_CHECKS", False)
POINTS_SCORE_BITS = getattr(variant, "POINTS_SCORE_BITS", 0)
CHECKS_BITS = getattr(variant, "CHECKS_BITS", 0)
FEATURE_HASH = 0x8f3f9d5a if (HAS_POINTS or HAS_CHECKS) else 0x5f234cb8
NUM_POINTS_SCORE_PLANES = 2 * POINTS_SCORE_BITS if HAS_POINTS else 0
NUM_CHECK_PLANES = 2 * CHECKS_BITS if HAS_CHECKS else 0
NUM_POINTS_PLANES = NUM_POINTS_SCORE_PLANES + NUM_CHECK_PLANES
NUM_PLANES_BASE = NUM_SQ * NUM_PT_REAL + (NUM_PT_REAL - (NUM_KSQ != 1)) * variant.POCKETS
NUM_PLANES_REAL = NUM_PLANES_BASE + NUM_POINTS_PLANES
NUM_PLANES_VIRTUAL = NUM_SQ * NUM_PT_VIRTUAL + (NUM_PT_REAL - (NUM_KSQ != 1)) * variant.POCKETS + NUM_POINTS_PLANES
NUM_INPUTS = NUM_PLANES_REAL * NUM_KSQ

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

def halfka_points_idx(is_white_pov: bool, king_sq: int, plane: int):
  return plane + NUM_PLANES_BASE + king_sq * NUM_PLANES_REAL

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
      king_sq = orient(turn, board.king(turn))
      for sq, p in board.piece_map().items():
        indices[halfka_idx(turn, king_sq, sq, p)] = 1.0

      def get_color_value(values, color):
        if values is None:
          return 0
        if isinstance(values, dict):
          return int(values.get(color, 0))
        try:
          return int(values[color])
        except Exception:
          return int(values)

      if HAS_POINTS:
        points_score = getattr(board, "points_score", None)
        if points_score is not None:
          clamp = (1 << POINTS_SCORE_BITS) - 1
          us_score = max(0, min(get_color_value(points_score, turn), clamp))
          them_score = max(0, min(get_color_value(points_score, not turn), clamp))
          for bit in range(POINTS_SCORE_BITS):
            mask = 1 << bit
            if us_score & mask:
              indices[halfka_points_idx(turn, king_sq, bit)] = 1.0
            if them_score & mask:
              indices[halfka_points_idx(turn, king_sq, POINTS_SCORE_BITS + bit)] = 1.0

      if HAS_CHECKS:
        checks_remaining = getattr(board, "checks_remaining", None)
        if checks_remaining is not None:
          clamp = (1 << CHECKS_BITS) - 1
          us_checks = max(0, min(get_color_value(checks_remaining, turn), clamp))
          them_checks = max(0, min(get_color_value(checks_remaining, not turn), clamp))
          offset = NUM_POINTS_SCORE_PLANES
          for bit in range(CHECKS_BITS):
            mask = 1 << bit
            if us_checks & mask:
              indices[halfka_points_idx(turn, king_sq, offset + bit)] = 1.0
            if them_checks & mask:
              indices[halfka_points_idx(turn, king_sq, offset + CHECKS_BITS + bit)] = 1.0
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
      if a_idx < NUM_SQ * NUM_PT_REAL:
        if a_idx // NUM_SQ == NUM_PT_REAL - 1 and k_idx != map_king(a_idx % NUM_SQ):
          # is king piece, but not ours
          a_idx += NUM_SQ
      elif a_idx < NUM_PLANES_BASE:
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
