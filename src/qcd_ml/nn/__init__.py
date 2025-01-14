"""
This module provides neural networks for lattice QCD. 
The modules ``ptc`` and ``lptc`` provide parallel transport pooling and local parallel transport pooling, respectively.
The module ``pt_pool`` provides the ``v_ProjectLayer`` class for parallel transport pooling and some utility functions for 
paralell transport pooling.
The modules ``dense`` and ``pt`` provide dense layers and parallel transport layers, which can be used to build more general gauge-equivariant neural networks.
"""

import qcd_ml.nn.dense
import qcd_ml.nn.lptc
import qcd_ml.nn.pt
import qcd_ml.nn.pt_pool
import qcd_ml.nn.ptc
