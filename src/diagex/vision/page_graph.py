"""Compatibility import; implementation lives in :mod:`connection_inference`."""
import sys

from diagex.vision import connection_inference as _implementation

sys.modules[__name__] = _implementation
