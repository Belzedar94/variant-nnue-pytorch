"""Explicit AtomicNNUEV3 trainer backend.

Import submodules directly (``atomic_v3.model``, ``atomic_v3.dataset`` and
``atomic_v3.training``).  No registration into the generic legacy feature-set
dispatcher occurs in H9.3l-j.
"""

from .contract import BACKEND_KEY, BACKEND_NAME, FEATURE_NAME

__all__ = ["BACKEND_KEY", "BACKEND_NAME", "FEATURE_NAME"]
