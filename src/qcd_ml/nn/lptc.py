"""
qcd_ml.nn.lptc
==============

Local Parallel Transport Convolutions.
"""

import torch
from typing import List, Tuple, Optional, Dict, Any

from ..base.paths import v_ng_evaluate_path, PathBuffer
from ..base.operations import v_spin_transform, v_ng_spin_transform


class v_LPTC(torch.nn.Module):
    """
    Local Parallel Transport Convolution for objects that 
    transform vector-like.

    Weights are stored as [feature_in, feature_out, path].

    paths is a list of paths. Every path is a list [(direction, nhops)].
    An empty list is the path that does not perform any hops.
    """

    def __init__(
        self,
        n_feature_in: int,
        n_feature_out: int,
        paths: List[List[Tuple[int, int]]],
        U: torch.Tensor,
        **path_buffer_kwargs: Dict[str, Any]
    ) -> None:
        """
        Initialize the v_LPTC layer.

        Args:
            n_feature_in: Number of input features.
            n_feature_out: Number of output features.
            paths: List of paths. Each path is a list of tuples (direction, nhops).
            U: Gauge field tensor.
            **path_buffer_kwargs: Additional keyword arguments for PathBuffer initialization.

        Notes:
            Weights are initialized as a random tensor with shape
            [n_feature_in, n_feature_out, len(paths), *U[0].shape[0:4], 4, 4]
            and dtype torch.cdouble.
            The path_buffers are created for each path in paths.
        """
        super().__init__()
        self.weights = torch.nn.Parameter(
                torch.randn(n_feature_in, n_feature_out, len(paths), *tuple(U[0].shape[0:4]), 4, 4, dtype=torch.cdouble)
                )

        self.n_feature_in = n_feature_in
        self.n_feature_out = n_feature_out
        self.path_buffer_kwargs = path_buffer_kwargs
        self.path_buffers = [PathBuffer(U, pi, **path_buffer_kwargs) for pi in paths]

    def forward(self, features_in: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the v_LPTC layer.

        Args:
            features_in: Input features tensor with shape [n_feature_in, ...].

        Returns:
            Output features tensor with shape [n_feature_out, ...].

        Raises:
            ValueError: If the number of input features does not match n_feature_in.
        """
        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(f"shape mismatch: got {features_in.shape[0]} but expected {self.n_feature_in}")

        features_out = [torch.zeros_like(features_in[0]) for _ in range(self.n_feature_out)]

        for fi, wfi in zip(features_in, self.weights):
            for io, wfo in enumerate(wfi):
                for pi, wi in zip(self.path_buffers, wfo):
                    features_out[io] = features_out[io] + v_spin_transform(wi, pi.v_transport(fi))

        return torch.stack(features_out)

    def gauge_transform_using_transformed(self, U_transformed: torch.Tensor) -> None:
        """
        Update the v_LPTC layer: The old gauge field U is replaced by
        U_transformed. The weights are kept.

        NOTE: This does not create a transformed copy of the layer!
              Instead the layer is updated.

        Mostly used for testing.

        Args:
            U_transformed: The new gauge field tensor to replace the old U.
        """
        for i, pi in enumerate(self.path_buffers):
            self.path_buffers[i] = PathBuffer(U_transformed, pi.path, **self.path_buffer_kwargs)


class v_LPTC_NG(torch.nn.Module):
    """
    Local Parallel Transport Convolution for objects that
    transform vector-like but with no gauge degrees of freedom.

    Weights are stored as [feature_in, feature_out, path].

    paths is a list of paths. Every path is a list [(direction, nhops)].
    An empty list is the path that does not perform any hops.
    """

    def __init__(
        self,
        n_feature_in: int,
        n_feature_out: int,
        paths: List[List[Tuple[int, int]]],
        grid_dims: Tuple[int, ...],
        internal_dof: int
    ) -> None:
        """
        Initialize the v_LPTC_NG layer.

        Args:
            n_feature_in: Number of input features.
            n_feature_out: Number of output features.
            paths: List of paths. Each path is a list of tuples (direction, nhops).
            grid_dims: Dimensions of the grid.
            internal_dof: Internal degrees of freedom.

        Notes:
            Weights are initialized as a random tensor with shape
            [n_feature_in, n_feature_out, len(paths), *grid_dims, internal_dof, internal_dof]
            and dtype torch.cdouble.
        """
        super().__init__()
        self.weights = torch.nn.Parameter(
                torch.randn(n_feature_in, n_feature_out, len(paths), *tuple(grid_dims), internal_dof, internal_dof, dtype=torch.cdouble)
                )

        self.n_feature_in = n_feature_in
        self.n_feature_out = n_feature_out
        self.paths = paths
        self.internal_dof = internal_dof
        self.grid_dims = grid_dims

    def forward(self, features_in: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the v_LPTC_NG layer.

        Args:
            features_in: Input features tensor with shape [n_feature_in, ...].

        Returns:
            Output features tensor with shape [n_feature_out, ...].

        Raises:
            ValueError: If the number of input features does not match n_feature_in.
        """
        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(f"shape mismatch: got {features_in.shape[0]} but expected {self.n_feature_in}")

        features_out = [torch.zeros_like(features_in[0]) for _ in range(self.n_feature_out)]

        for fi, wfi in zip(features_in, self.weights):
            for io, wfo in enumerate(wfi):
                for pi, wi in zip(self.paths, wfo):
                    features_out[io] = features_out[io] + v_ng_spin_transform(wi, v_ng_evaluate_path(pi, fi))

        return torch.stack(features_out)
