"""
This package provides 

- ``v_ProjectLayer``: A parallel transport pooling projection layer. 
- ``get_paths.get_paths_*``: Functions to generate complete sets of paths for a given block size.
- ``pool4d.v_pool4d`` and ``pool4d.v_unpool4d``: Slow implementations of pooling over a 4d spin-color vector field.

For a faster implementation of the pooling and unpooling, install the package ``qcd_ml_accel``.

For a detailed description of the parallel transport pooling, see the paper:
https://arxiv.org/abs/2304.10438
"""
from .pool import v_ProjectLayer

import qcd_ml.nn.pt_pool.get_paths
