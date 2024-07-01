from .hop import v_hop

def v_evaluate_path(U, path, v):
    """
    paths is a list of paths. Every path is a list [(mu, nhops)].
    An empty list is the path that does not perform any hops.
    
    If nhops is negative, the hop is made in negative mu direction.
    """
    for mu, nhops in path:
        if nhops < 0:
            direction = -1
            nhops *= -1
        else:
            direction = 1

        for _ in range(nhops):
            v = v_hop(U, mu, direction, v)
    return v
