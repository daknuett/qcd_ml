"""
Dense
=====

Dense Layers.
"""

import torch

from ..base.operations import v_spin_const_transform


class v_Dense(torch.nn.Module):
    """
    Dense Layer for vectors.

    Weights are stored as [feature_in, feature_out].

    The output features are a linear combination of input features, multiplied
    by weights in the form of 4x4 spin matrices.
    """

    def __init__(self, n_feature_in, n_feature_out):
        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.randn(n_feature_in, n_feature_out, 4, 4, dtype=torch.cdouble)
        )

        self.n_feature_in = n_feature_in
        self.n_feature_out = n_feature_out

    def forward(self, features_in):
        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(
                f"shape mismatch: got {features_in.shape[0]} but expected {self.n_feature_in}"
            )

        return torch.einsum("iojk,iabcdkG->oabcdjG", self.weights, features_in)
