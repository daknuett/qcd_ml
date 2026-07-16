"""
qcd_ml.base - Base operations for QCD ML
==========================================

This module provides fundamental operations for working with QCD data:
- Hopping operations (gauge and non-gauge)
- Group operations (SU3 composition, gauge transformations)
- Path evaluation for gauge-equivariant operations
"""

from . import hop, operations, paths
