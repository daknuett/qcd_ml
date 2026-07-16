"""
This module provides neural networks for lattice QCD.
The modules ``ptc`` and ``lptc`` provide parallel transport pooling and local parallel transport pooling, respectively.
The module ``pt_pool`` provides the ``v_ProjectLayer`` class for parallel transport pooling and some utility functions for
paralell transport pooling.
The modules ``dense`` and ``pt`` provide dense layers and parallel transport layers, which can be used to build more general gauge-equivariant neural networks.
"""

from . import dense, lptc, matrix_layers, non_gauge, pt, ptc
