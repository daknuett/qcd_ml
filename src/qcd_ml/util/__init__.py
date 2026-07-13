"""
qcd_ml.util - Utility functions
===============================

This module provides utility functions for working with tensors and solving linear systems:
- Iterative solvers (GMRES)
- QCD-specific utilities (multigrid)
- Compile-time evaluation utilities
"""

import torch
import qcd_ml.util.qcd
import qcd_ml.util.solver
