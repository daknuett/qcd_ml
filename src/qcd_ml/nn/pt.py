"""
pt
==

Parallel Transport Layers.
"""

import torch
from typing import List, Tuple, Optional, Dict, Any

from ..base.paths import PathBuffer

class v_PT(torch.nn.Module):
    """
    Parallel Transport Layer for objects that transform vector-like.

    It has no weights.

    paths is a list of paths. Every path is a list [(direction, nhops)].
    An empty list is the path that does not perform any hops.

    For a PT layer with all 0- and 1-hop paths, construct the layer like this::

        U = torch.tensor(np.load("path/to/gauge/config.npy"))

        paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(mu, -1)] for mu in range(4)]
        layer = v_PT(paths, U)
    """

    def __init__(
        self,
        paths: List[List[Tuple[int, int]]],
        U: torch.Tensor,
        **path_buffer_kwargs: Any
    ) -> None:
        """Initialize the v_PT parallel transport layer.

        Args:
            paths: List of paths. Each path is a list of tuples (direction, nhops)
                where direction is the spacetime dimension index and nhops is the
                number of hops (positive or negative) in that direction.
                An empty list [] represents a path with no hops.
            U: Gauge field tensor of shape (4, Lx, Ly, Lz, Lt, 3, 3) where 4 is the
                number of spacetime dimensions and 3x3 represents SU(3) matrices.
            **path_buffer_kwargs: Additional keyword arguments passed to PathBuffer
                instances for each path. These may include:
                - gauge_group_compose: Function to compose gauge group elements
                - v_gauge_transform: Function to gauge transform vector-like fields
                - m_gauge_transform: Function to gauge transform matrix-like fields
                - adjoin: Function to compute adjoint
                - gauge_identity: Identity element of the gauge group
        """
        super().__init__()
        self.n_feature_in = len(paths)
        self.n_feature_out = len(paths)
        self.path_buffer_kwargs = path_buffer_kwargs
        self.path_buffers = [
            PathBuffer(U, pi, **path_buffer_kwargs) for pi in paths
        ]

    def forward(self, features_in: torch.Tensor) -> torch.Tensor:
        """Forward parallel transport of input features along all paths.

        Each input feature is transported along its corresponding path using
        the pre-computed gauge transport matrices.

        Args:
            features_in: Input tensor of shape (n_paths, ...) where n_paths is the
                number of paths (self.n_feature_in). Each feature corresponds to
                one path and will be transported along that path.

        Returns:
            Output tensor of shape (n_paths, ...) containing the transported features.
            The output at position i corresponds to the input at position i transported
            along path i.

        Raises:
            ValueError: If the number of input features does not match the number of paths.
        """
        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(
                f"shape mismatch: got {features_in.shape[0]} but expected {self.n_feature_in}"
            )

        features_out = [None] * self.n_feature_out

        for i, p in enumerate(self.path_buffers):
            features_out[i] = p.v_transport(features_in[i])

        return torch.stack(features_out)

    def reverse(self, features_in: torch.Tensor) -> torch.Tensor:
        """Reverse parallel transport of input features along all paths.

        Each input feature is transported in the reverse direction along its
        corresponding path using the pre-computed gauge transport matrices.

        Args:
            features_in: Input tensor of shape (n_paths, ...) where n_paths is the
                number of paths (self.n_feature_in). Each feature corresponds to
                one path and will be transported in reverse along that path.

        Returns:
            Output tensor of shape (n_paths, ...) containing the reverse-transported
            features. The output at position i corresponds to the input at position i
            transported in reverse along path i.

        Raises:
            ValueError: If the number of input features does not match the number of paths.
        """
        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(
                f"shape mismatch: got {features_in.shape[0]} but expected {self.n_feature_in}"
            )

        features_out = [None] * self.n_feature_out

        for i, p in enumerate(self.path_buffers):
            features_out[i] = p.v_reverse_transport(features_in[i])

        return torch.stack(features_out)

    def gauge_transform_using_transformed(self, U_transformed: torch.Tensor) -> None:
        """
        Update the v_PT layer: The old gauge field U is replaced by
        U_transformed. The weights are kept.

        NOTE: This does not create a transformed copy of the layer!
              Instead the layer is updated.

        Mostly used for testing.

        Args:
            U_transformed: Transformed gauge field tensor of shape (4, Lx, Ly, Lz, Lt, 3, 3)
                to replace the current gauge field.
        """
        for i, pi in enumerate(self.path_buffers):
            self.path_buffers[i] = PathBuffer(
                U_transformed, pi.path, **self.path_buffer_kwargs
            )
