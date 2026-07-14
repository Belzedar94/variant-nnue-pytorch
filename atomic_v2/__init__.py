"""Isolated AtomicNNUEV2 trainer and wire-format support.

The historical trainer remains at the repository root.  This package is the
additive SFNNv15 path and deliberately does not auto-convert legacy V1 nets.
"""

from .contract import BACKEND_NAME, FILE_VERSION, NETWORK_HASH
from .model import AtomicNNUEV2

__all__ = ["AtomicNNUEV2", "BACKEND_NAME", "FILE_VERSION", "NETWORK_HASH"]
