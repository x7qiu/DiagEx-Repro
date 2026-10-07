"""Compatibility import; implementation lives in :mod:`symbol_interpretation`."""
import sys

from diagex.vision import symbol_interpretation as _implementation

sys.modules[__name__] = _implementation
