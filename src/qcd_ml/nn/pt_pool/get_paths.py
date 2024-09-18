import itertools


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
    return [list(reversed(pth)) for pth in get_paths_one_step_lexicographic(block_size)]
