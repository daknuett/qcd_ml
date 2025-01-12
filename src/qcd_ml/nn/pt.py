"""
pt
==

Parallel Transport Layers.
"""

import torch
from qcd_ml.base.paths import PathBuffer


class v_PT(torch.nn.Module):
    """
    Parallel Transport Layer for objects that transform vector-like.

    It has no weights.

    paths is a list of paths. Every path is a list [(direction, nhops)].
    An empty list is the path that does not perform any hops.
    """

    def __init__(self, paths, U):
        super().__init__()
        self.n_feature_in = len(paths)
        self.n_feature_out = len(paths)
        # FIXME: This is more memory intensive compared to the
        # implementation using v_evaluate_path, because instead of one
        # copy of U, all gauge transport matrices are stored.
        # On the other hand this may not be a big deal in most cases,
        # because for 1h, the number of gauge fields is identical.
        self.path_buffers = [PathBuffer(U, pi) for pi in paths]

    def forward(self, features_in):
        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(
                f"shape mismatch: got {features_in.shape[0]} but expected {self.n_feature_in}"
            )

        features_out = [
            torch.zeros_like(features_in[0]) for _ in range(self.n_feature_out)
        ]

        for i, p in enumerate(self.path_buffers):
            features_out[i] = p.v_transport(features_in[i])

        return torch.stack(features_out)
