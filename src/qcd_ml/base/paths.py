import torch

from .hop import v_hop, v_ng_hop
from .operations import SU3_group_compose
from .operations import v_gauge_transform

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

def v_ng_evaluate_path(path, v):
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
            v = v_ng_hop(mu, direction, v)
    return v


def v_reverse_evaluate_path(U, path, v):
    """
    paths is a list of paths. Every path is a list [(mu, nhops)].
    An empty list is the path that does not perform any hops.
    
    If nhops is negative, the hop is made in negative mu direction.
    """
    for mu, nhops in reversed(path):
        nhops *= -1
        if nhops < 0:
            direction = -1
            nhops *= -1
        else:
            direction = 1

        for _ in range(nhops):
            v = v_hop(U, mu, direction, v)
    return v

def v_ng_reverse_evaluate_path(path, v):
    """
    paths is a list of paths. Every path is a list [(mu, nhops)].
    An empty list is the path that does not perform any hops.
    
    If nhops is negative, the hop is made in negative mu direction.
    """
    for mu, nhops in reversed(path):
        nhops *= -1
        if nhops < 0:
            direction = -1
            nhops *= -1
        else:
            direction = 1

        for _ in range(nhops):
            v = v_ng_hop(mu, direction, v)
    return v


class PathBuffer:
    """
    This class brings the same functionality as v_evaluate_path and
    v_reverse_evaluate_path but pre-computes the costly gauge transport matrix
    multiplications.
    """
    def __init__(self, U, path):
        self.path = path

        self.accumulated_U = torch.zeros_like(U[0])
        self.accumulated_U[:,:,:,:] = torch.complex(
                torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.double)
                , torch.zeros(3, 3, dtype=torch.double)
                )

        print(self.path)
        for mu, nhops in self.path:
            if nhops < 0:
                direction = -1
                nhops *= -1
            else:
                direction = 1

            for _ in range(nhops):
                if direction == -1:
                    U = torch.roll(U, 1, mu + 1) # mu + 1 because U is (mu, x, y, z, t)
                    self.accumulated_U = SU3_group_compose(U[mu], self.accumulated_U)
                else:
                    self.accumulated_U = SU3_group_compose(U[mu].adjoint(), self.accumulated_U)
                    U = torch.roll(U, -1, mu + 1)

    def v_transport(self, v):
        v = v_gauge_transform(self.accumulated_U, v)
        v = v_ng_evaluate_path(self.path, v)
        return v

    def v_reverse_transport(self, v):
        v = v_ng_reverse_evaluate_path(self.path, v)
        v = v_gauge_transform(self.accumulated_U.adjoint(), v)
        return v
