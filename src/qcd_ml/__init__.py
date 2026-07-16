"""
qcd_ml - Machine learning tools for QCD
=======================================

A PyTorch-based library for quantum chromodynamics (QCD) machine learning research.

This package provides:
- Gauge-equivariant neural network layers
- QCD operators (Dirac, Wilson, etc.)
- Gauge field observables and smearing
- Path evaluation and hopping operations
- Utility functions for tensor operations

See the README for more information.
"""

from . import base, nn, qcd, util
