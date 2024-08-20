import torch

from .simple_paths import v_ng_evaluate_path, v_ng_reverse_evaluate_path
from ..operations import v_gauge_transform, SU3_group_compose
from .compile import compile_path

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

            self.path = compile_path(self.path)

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

