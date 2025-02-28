import torch

from .simple_paths import v_ng_evaluate_path, v_ng_reverse_evaluate_path
from ..operations import v_gauge_transform, SU3_group_compose, m_gauge_transform
from .compile import compile_path

class PathBuffer:
    """
    This class brings the same functionality as v_evaluate_path and
    v_reverse_evaluate_path but pre-computes the costly gauge transport matrix
    multiplications.

    To access the gauge transport matrix, use ``PathBuffer(U, path).gauge_transport_matrix``.
    """
    def __init__(self, U, path
                 , gauge_group_compose=SU3_group_compose
                 , v_gauge_transform=v_gauge_transform
                 , m_gauge_transform=m_gauge_transform
                 , adjoin=lambda x: x.adjoint()
                 , gauge_identity=torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.cdouble)):
        if isinstance(U, list):
            # required by torch.roll below.
            U = torch.stack(U)
        self.path = path

        self.gauge_group_compose = gauge_group_compose
        self.v_gauge_transform = v_gauge_transform
        self.m_gauge_transform = m_gauge_transform
        self.adjoin = adjoin

        if len(self.path) == 0:
            # save computational cost.
            self._is_identity = True
            self.accumulated_U = torch.zeros_like(U[0])
            self.accumulated_U[:,:,:,:] = torch.clone(gauge_identity)
        else:
            self._is_identity = False

            self.accumulated_U = torch.zeros_like(U[0])
            self.accumulated_U[:,:,:,:] = torch.clone(gauge_identity)

            for mu, nhops in self.path:
                if nhops < 0:
                    direction = -1
                    nhops *= -1
                else:
                    direction = 1

                for _ in range(nhops):
                    if direction == -1:
                        U = torch.roll(U, 1, mu + 1) # mu + 1 because U is (mu, x, y, z, t)
                        self.accumulated_U = self.gauge_group_compose(U[mu], self.accumulated_U)
                    else:
                        self.accumulated_U = self.gauge_group_compose(self.adjoin(U[mu]), self.accumulated_U)
                        U = torch.roll(U, -1, mu + 1)

            self.path = compile_path(self.path)

    @property
    def gauge_transport_matrix(self):
        return self.accumulated_U

    def v_transport(self, v):
        """
        Gauge-equivariantly transport the vector-like field ``v`` along the path.
        """
        if not self._is_identity:
            v = self.v_gauge_transform(self.accumulated_U, v)
            v = v_ng_evaluate_path(self.path, v)
        return v

    def v_reverse_transport(self, v):
        """
        Inverse of ``v_transport``.
        """
        if not self._is_identity:
            v = v_ng_reverse_evaluate_path(self.path, v)
            v = self.v_gauge_transform(self.adjoin(self.accumulated_U), v)
        return v

    def m_transport(self, m):
        if not self._is_identity:
            m = self.m_gauge_transform(self.accumulated_U, m)
            m = v_ng_evaluate_path(self.path, m)
        return m

    def m_reverse_transport(self, m):
        """
        Inverse of ``m_transport``.
        """
        if not self._is_identity:
            m = v_ng_reverse_evaluate_path(self.path, m)
            m = self.m_gauge_transform(self.adjoin(self.accumulated_U), m)
        return m
