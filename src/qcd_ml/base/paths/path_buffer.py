import torch
from typing import List, Tuple, Callable

from .simple_paths import v_ng_evaluate_path, v_ng_reverse_evaluate_path
from ..operations import v_gauge_transform, SU3_group_compose, m_gauge_transform
from .compile import compile_path

class PathBuffer:
    """
    This class brings the same functionality as v_evaluate_path and
    v_reverse_evaluate_path but pre-computes the costly gauge transport matrix
    multiplications.

    To access the pre-computed gauge transport matrix, use
    ``PathBuffer(U, path).gauge_transport_matrix``.
    """

    def __init__(
        self,
        U: torch.Tensor,
        path: List[Tuple[int, int]],
        gauge_group_compose: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = SU3_group_compose,
        v_gauge_transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = v_gauge_transform,
        m_gauge_transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = m_gauge_transform,
        adjoin: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.adjoint(),
        gauge_identity: torch.Tensor = torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.cdouble
        ),
    ) -> None:
        """
        Initialize the PathBuffer with gauge field and path.

        Pre-computes the gauge transport matrix multiplications for efficient
        transport operations along the specified path.

        Args:
            U: Gauge field tensor of shape (4, Lx, Ly, Lz, Lt, Nc, Nc) where 4 is the number
                of spacetime dimensions and Nc is the number of colors (usually 3).
            path: List of path elements as tuples (mu, nhops) where mu is the
                direction index and nhops is the number of hops.
            gauge_group_compose: Function to compose gauge group elements.
                Defaults to qcd_ml.base.operations.SU3_group_compose.
            v_gauge_transform: Function to gauge transform vector-like fields.
                Defaults to qcd_ml.base.operations.v_gauge_transform.
            m_gauge_transform: Function to gauge transform matrix-like fields.
                Defaults to qcd_ml.base.operations.m_gauge_transform.
            adjoin: Function to compute the adjoint of a gauge group element. Defaults to
                lambda x: x.adjoint().
            gauge_identity: The identity element of the gauge group as a tensor.
                Defaults to the 3x3 complex identity matrix.

        """
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
    def gauge_transport_matrix(self) -> torch.Tensor:
        """
        Get the pre-computed gauge transport matrix.
        """
        return self.accumulated_U

    def v_transport(self, v: torch.Tensor) -> torch.Tensor:
        """
        Gauge-equivariantly transport the vector-like field ``v`` along the path.

        Args:
            v: The vector-like field tensor to transport.

        Returns:
            The transported vector-like field tensor.
        """
        if not self._is_identity:
            v = self.v_gauge_transform(self.accumulated_U, v)
            v = v_ng_evaluate_path(self.path, v)
        return v

    def v_reverse_transport(self, v: torch.Tensor) -> torch.Tensor:
        """
        Inverse of ``v_transport``, i.e, transport ``v`` along the reversed path.

        Args:
            v: The vector-like field tensor to transport.

        Returns:
            The reverse transported vector-like field tensor.
        """
        if not self._is_identity:
            v = v_ng_reverse_evaluate_path(self.path, v)
            v = self.v_gauge_transform(self.adjoin(self.accumulated_U), v)
        return v

    def m_transport(self, m: torch.Tensor) -> torch.Tensor:
        """
        Gauge-equivariantly transport the matrix-like field ``m`` along the path.

        Args:
            m: The matrix-like field tensor to transport.

        Returns:
            The transported matrix-like field tensor.
        """
        if not self._is_identity:
            m = self.m_gauge_transform(self.accumulated_U, m)
            m = v_ng_evaluate_path(self.path, m)
        return m

    def m_reverse_transport(self, m: torch.Tensor) -> torch.Tensor:
        """
        Inverse of ``m_transport``, i.e., transport ``m`` along the reversed path.

        Args:
            m: The matrix-like field tensor to reverse transport.

        Returns:
            The reverse transported matrix-like field tensor.
        """
        if not self._is_identity:
            m = v_ng_reverse_evaluate_path(self.path, m)
            m = self.m_gauge_transform(self.adjoin(self.accumulated_U), m)
        return m
