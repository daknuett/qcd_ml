"""
qcd_ml.base.paths
=================

Gauge-equivariant parallel transport paths.

The function ``v_evaluate_path`` is memory effective but slow.

The class ``PathBuffer`` can be used to speed up path evaluation
but may be more memory intensve.


"""

from .simple_paths import (
        v_evaluate_path, v_ng_evaluate_path, v_reverse_evaluate_path, v_ng_reverse_evaluate_path
        )

from .path_buffer import PathBuffer
