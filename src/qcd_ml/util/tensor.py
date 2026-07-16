"""
Utilities related to tensors.

Contains:

    - ``get_permutation_sign``: Computes the sign of a permutation.
    - ``levi_civita_index_and_sign_iterator``: Yields ``(index, element)`` of the epsilon pseudo-tensor.

"""

from typing import Iterator, List, Tuple

def get_permutation_sign(permutation: List[int]) -> int:
    """
    Returns the number of switches of neighboring elements necessary
    to sort ``permutation``.

    This is useful for instance in the case of the Levi Civita symbol
    where the tensor element depends on the sign of the permutation.

    Note:
        This function modifies the input list in place.

    Args:
        permutation: A list of integers representing a permutation.

    Returns:
        The sign of the permutation: 1 for even permutations, -1 for odd permutations.

    Example::

        >>> get_permutation_sign([0, 1, 2])
        1
        >>> get_permutation_sign([1, 0, 2])
        -1

    """
    n_switches = 0
    did_switch = False
    n_elements = len(permutation)

    for i in range(n_elements - 1):
        if permutation[i] > permutation[i+1]:
            did_switch = True
            permutation[i+1], permutation[i] = permutation[i], permutation[i+1]
            n_switches += 1

    while did_switch:
        did_switch = False
        for i in range(n_elements - 1):
            if permutation[i] > permutation[i+1]:
                did_switch = True
                permutation[i+1], permutation[i] = permutation[i], permutation[i+1]
                n_switches += 1
    return (-1) ** n_switches

def _continue_levi_civita_permutations(
    idx_at: int,
    idx_already: List[int],
    c_sgn: int,
    remaining_idcs: List[int]
) -> Iterator[Tuple[List[int], int]]:
    """
    Recursive helper generator for generating Levi-Civita permutations and their signs.

    Args:
        idx_at: Current position in the permutation being built.
        idx_already: List of indices already placed in the permutation.
        c_sgn: Current sign multiplier (1 or -1).
        remaining_idcs: List of remaining indices to be placed.

    Yields:
        Tuples of (permutation, sign) for all permutations of the input indices.
        Each permutation is a list of indices, and sign is either 1 or -1.
    """
    if len(remaining_idcs) == 1:
        cix = remaining_idcs[0]
        yield idx_already + remaining_idcs, c_sgn
        return
    for i, cix in enumerate(remaining_idcs):
        this_idx_already = idx_already + [cix]
        this_c_sgn = c_sgn * (-1) ** i

        yield from _continue_levi_civita_permutations(idx_at + 1
                                                      , this_idx_already
                                                      , this_c_sgn
                                                      , remaining_idcs[:i] + remaining_idcs[i+1:])

def levi_civita_index_and_sign_iterator(nd: int) -> Iterator[Tuple[List[int], int]]:
    r"""
    Iterator over the indices and (non-zero) elements of the Levi Civita symbol (or epsilon pseudo tensor).
    Yields ``index, element``, i.e., :math:`((i,j,k), \epsilon_{i,j,k})`.
    ``nd`` is the number of dimensions.

    Args:
        nd: The number of dimensions (size of the permutation).

    Yields:
        Tuples of (index, element) where index is a permutation of ``range(nd)``
        and element is the corresponding Levi-Civita symbol value (1 or -1).

    Example::

        >>> list(levi_civita_index_and_sign_iterator(3))
        [([0, 1, 2], 1),
         ([0, 2, 1], -1),
         ([1, 0, 2], -1),
         ([1, 2, 0], 1),
         ([2, 0, 1], 1),
         ([2, 1, 0], -1)]
    """
    remaining_idcs = list(range(nd))
    yield from _continue_levi_civita_permutations(0, [], 1, remaining_idcs)
