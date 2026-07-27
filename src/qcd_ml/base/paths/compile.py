"""
Compile non-gauge paths for faster execution.

This is for internal use only!

v_ng_evaluate_path can be optimized by compiling the paths,
such that as few rolls as possible are necessary, because only
start and end point are important.

XXX: Paths CANNOT be compiled when a gauge field is present! 
     In this case, the result depends on the path. Use PathBuffer instead.
"""

from typing import List, Tuple


def compile_path(path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Compiles a path, such that few rolls are necessary to
    v_ng_evaluate_path.

    This optimization works by producing a single transport along every dimension.

    Mostly used internally by PathBuffer.
    """
    shifts = [0] * 4

    for mu, nhops in path:
        shifts[mu] += nhops

    return [(mu, nhops) for mu, nhops in enumerate(shifts) if nhops != 0]
