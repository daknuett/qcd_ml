"""
Utilities related to tensors.

Contains:

    - ``get_permutation_sign``: Computes the sign of a permutation.
    - ``levi_civita_index_and_sign_iterator``: Yields ``(index, element)`` of the epsilon pseudo-tensor.

"""

def get_permutation_sign(permutation):
    """
    Returns the number of switches of neighboring elements necessary
    to sort ``permutation``.

    This is useful for instance in the case of the Levi Civita symbol
    where the tensor element depends on the sign of the permutation.

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


def _continue_levi_civita_permutations(idx_at, idx_already, c_sgn, remaining_idcs):
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


def levi_civita_index_and_sign_iterator(nd):
    r"""
    Iterator over the indices and (non-zero) elements of the Levi Civita symbol (or epsilon pseudo tensor).
    Yields ``index, element``, i.e., :math:`((i,j,k), \epsilon_{i,j,k})`.
    ``nd`` is the number of dimensions.

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
