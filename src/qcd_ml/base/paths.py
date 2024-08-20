"""
qcd_ml.base.paths
=================

Gauge-equivariant parallel transport paths.

The function ``v_evaluate_path`` is memory effective but slow.

The class ``PathBuffer`` can be used to speed up path evaluation
but may be more memory intensve.


"""
import torch

from .hop import v_hop, v_ng_hop
from .operations import SU3_group_compose
from .operations import v_gauge_transform

def v_evaluate_path(U, path, v):
    """
    Gauge-equivariantly evaluate a path on a vector-like field.

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
    Evaluate a path on a vector-like field without gauge degrees of freedom.

    paths is a list of paths. Every path is a list [(mu, nhops)].
    An empty list is the path that does not perform any hops.
    
    If nhops is negative, the hop is made in negative mu direction.
    """
    if len(path) > 0:
        mus = [mu for mu,_ in path]
        hops = [nhops for _,nhops in path]
        return torch.roll(v, shifts=hops, dims=mus)
    return v


def slow_v_ng_evaluate_path(path, v):
    """
    XXX: deprecated; only used for testing purposes.

    Evaluate a path on a vector-like field without gauge degrees of freedom.

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
    Gauge-equivariantly evaluate a path on a vector-like field.
    This is the inverse of ``v_evaluate_path``.

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
    Inverse of ``v_ng_evaluate_path``.
    """
    if len(path) > 0:
        mus = [mu for mu,_ in path]
        hops = [-nhops for _,nhops in path]
        return torch.roll(v, shifts=hops, dims=mus)
    return v


def slow_v_ng_reverse_evaluate_path(path, v):
    """
    XXX: Deprecated; used for testing.
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
        if isinstance(U, list):
            # required by torch.roll below.
            U = torch.stack(U)
        self.path = path

        if len(self.path) == 0:
            # save computational cost and memory.
            self._is_identity = True
        else:
            self._is_identity = False

            self.accumulated_U = torch.zeros_like(U[0])
            self.accumulated_U[:,:,:,:] = torch.complex(
                    torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.double)
                    , torch.zeros(3, 3, dtype=torch.double)
                    )

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
        """
        Gauge-equivariantly transport the vector-like field ``v`` along the path.
        """
        if not self._is_identity:
            v = v_gauge_transform(self.accumulated_U, v)
            v = v_ng_evaluate_path(self.path, v)
        return v

    def v_reverse_transport(self, v):
        """
        Inverse of ``v_transport``.
        """
        if not self._is_identity:
            v = v_ng_reverse_evaluate_path(self.path, v)
            v = v_gauge_transform(self.accumulated_U.adjoint(), v)
        return v
