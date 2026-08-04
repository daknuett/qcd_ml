"""
qcd_ml.base.paths
=================

Gauge-equivariant parallel transport paths.

The function ``v_evaluate_path`` is memory effective but slow.

The class ``PathBuffer`` can be used to speed up path evaluation
but may be more memory intensive.

"""

from typing import List, Tuple

from .simple_paths import (
        v_evaluate_path, v_ng_evaluate_path, v_reverse_evaluate_path, v_ng_reverse_evaluate_path, m_evaluate_path
        )

from .path_buffer import PathBuffer


def path_get_orig_point(path: List[Tuple[int, int]]) -> List[int]:
    """Returns the point that will be transported to the point [0,0,0,0] by the path.
    
    This function computes the origin point in 4D spacetime coordinates by
    reversing the path's displacement. Each path element (mu, nhops) contributes
    a displacement of -nhops in the mu direction.
    """
    point = [0] * 4
    for mu, nhops in path:
        point[mu] -= nhops
    return point
