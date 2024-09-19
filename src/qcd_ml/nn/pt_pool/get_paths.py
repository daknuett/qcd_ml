import itertools
import numpy as np

def get_paths_lexicographic(block_size):
    paths = []
    for position in itertools.product(*(range(bs) for bs in block_size)):
        path = sum([[(mu, -1)] for mu, n in enumerate(position) for _ in range(n)], start=[])
        paths.append(path)
    return paths


def get_paths_reverse_lexicographic(block_size):
    return [list(reversed(pth)) for pth in get_paths_lexicographic(block_size)]


def get_paths_one_step_lexicographic(block_size):
    paths = []
    for position in itertools.product(*(range(bs) for bs in block_size)):
        path = []
        pos = np.array(position)
        while pos.any():
            for mu in range(pos.shape[0]):
                if pos[mu] > 0:
                    path.append((mu, -1))
                    pos[mu] -= 1
        paths.append(path)
    return paths


def get_paths_one_step_reverse_lexicographic(block_size):
    """
    For a reference point (1,1,1,1) and a point (x_1 + 1, x_2 + 1, x_3 + 1, x_4 + 1) the path 
    is generated as such:
    .. math::

        \\cdots H_{-1}H_{-2}H_{-3}H_{-4}H_{-1}H_{-2}H_{-3}H_{-4}
    """
    paths = []
    for position in itertools.product(*(range(bs) for bs in block_size)):
        path = []
        pos = np.array(position)
        while pos.any():
            for mu in reversed(range(pos.shape[0])):
                if pos[mu] > 0:
                    path.append((mu, -1))
                    pos[mu] -= 1
        paths.append(path)
    return paths
