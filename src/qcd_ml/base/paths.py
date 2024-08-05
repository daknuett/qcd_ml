import torch

from .hop import v_hop, v_ng_hop
from .operations import SU3_group_compose
from .operations import v_gauge_transform

@torch.compile
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

@torch.compile
def v_ng_evaluate_path(path, v):
    """
    paths is a list of paths. Every path is a list [(mu, nhops)].
    An empty list is the path that does not perform any hops.
    
    If nhops is negative, the hop is made in negative mu direction.
    """
    if len(path) > 0:
        mus = [mu for mu,_ in path]
        hops = [nhops for _,nhops in path]
        return torch.roll(v, shifts=hops, dims=mus)
    return v


@torch.compile
def slow_v_ng_evaluate_path(path, v):
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


@torch.compile
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


@torch.compile
def v_ng_reverse_evaluate_path(path, v):
    """
    paths is a list of paths. Every path is a list [(mu, nhops)].
    An empty list is the path that does not perform any hops.
    
    If nhops is negative, the hop is made in negative mu direction.
    """
    if len(path) > 0:
        mus = [mu for mu,_ in path]
        hops = [-nhops for _,nhops in path]
        return torch.roll(v, shifts=hops, dims=mus)
    return v


@torch.compile
def slow_v_ng_reverse_evaluate_path(path, v):
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
    This class is buffer for paths using a given gauge configuration.
    It is intended to be used as such::

        pb = PathBuffer(U)

        transport_1 = pb.path([(1, 2), (0, -2),  (1, -2), (0, 2)])
        transport_2 = pb.path([(1, 2), (3, -2),  (1, -2), (3, 2)])

        vec = transport_1.v_transport(vec)
        vec = transport_2.v_transport(vec)

    This has the advantage that
    
    - the gauge transport matrices are pre-computed
    - the gauge transport matrices are stored once per path buffer, 
      i.e., when a path is re-used the memory is not duplicated.
    """
    def __init__(self, U):
        if isinstance(U, list):
            # required by torch.roll below.
            U = torch.stack(U)
        self.U = U

        self.accumulated_U = dict()

    def _check_path_descr(self, path_descr):
        spacetime_dim = self.U.shape[0]
        for i, (mu, nhops) in enumerate(path_descr):
            if mu < 0 or mu > spacetime_dim:
                raise ValueError(f"path component {i}: mu ({mu}) is out of bounds (0, {spacetime_dim}); path: {path_descr}")
        return True


    def path(self, path_descr):
        if len(path_descr) == 0:
            return NoopPathBufferTransporter()

        self._check_path_descr(path_descr)
        path_indicator = tuple(path_descr)

        U = self.U
        accumulated_U = torch.zeros_like(U[0])
        accumulated_U[:,:,:,:] = torch.complex(
                torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.double)
                , torch.zeros(3, 3, dtype=torch.double)
                )

        for mu, nhops in path_descr:
            if nhops < 0:
                direction = -1
                nhops *= -1
            else:
                direction = 1

            for _ in range(nhops):
                if direction == -1:
                    U = torch.roll(U, 1, mu + 1) # mu + 1 because U is (mu, x, y, z, t)
                    accumulated_U = SU3_group_compose(U[mu], accumulated_U)
                else:
                    accumulated_U = SU3_group_compose(U[mu].adjoint(), accumulated_U)
                    U = torch.roll(U, -1, mu + 1)

        self.accumulated_U[path_indicator] = accumulated_U
        return PathBufferTransporter(self, path_indicator)



class NoopPathBufferTransporter:
    __slots__ = ("path_indicator",)
    def __init__(self):
        self.path_indicator = tuple()

    @torch.compile
    def v_transport(self, v):
        return v

    @torch.compile
    def v_reverse_transport(self, v):
        return v


class PathBufferTransporter:
    __slots__ = ("path_buffer", "path_indicator")
    def __init__(self, path_buffer, path_indicator):
        self.path_buffer = path_buffer
        self.path_indicator = path_indicator

    @torch.compile
    def v_transport(self, v):
        v = v_gauge_transform(self.path_buffer.accumulated_U[self.path_indicator], v)
        v = v_ng_evaluate_path(self.path_indicator, v)
        return v

    @torch.compile
    def v_reverse_transport(self, v):
        v = v_ng_reverse_evaluate_path(self.path_indicator, v)
        v = v_gauge_transform(self.path_buffer.accumulated_U[self.path_indicator].adjoint(), v)
        return v
