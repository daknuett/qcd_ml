"""
Compile non-gauge paths for faster execution.

This is for internal use only!

v_ng_evaluate_path can be optimized by compiling the paths,
such that as few rolls as possible are necessary, because only
start and end point are important.

XXX: Paths CANNOT be compiled when a gauge field is present! 
     In this case, the result depends on the path.
"""

from typing import List, Tuple


def compile_path(path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Compiles a path, such that few rolls are necessary to
    v_ng_evaluate_path.

    This optimization works by combining multiple hops in the same direction
    into a single hop, reducing the number of tensor roll operations needed.

    Args:
        path: A list of tuples (mu, nhops) where mu is the direction index
            (0-3 for 4D spacetime) and nhops is the number of hops in that
            direction. Negative nhops indicate hops in the negative direction.

    Returns:
        A compiled list of tuples (mu, nhops) where only directions with
        non-zero net hops are included. The result is equivalent to the
        original path but requires fewer roll operations.

    Note:
        XXX: Do not use when a gauge field is present!
        In this case, the result depends on the path.
    """
    shifts = [0] * 4

    for mu, nhops in path:
        shifts[mu] += nhops

    return [(mu, nhops) for mu, nhops in enumerate(shifts) if nhops != 0]
