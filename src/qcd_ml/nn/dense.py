"""
Dense
=====

Dense Layers.
"""

import torch

from qcd_ml.base.operations import v_spin_const_transform


class v_Dense(torch.nn.Module):
    """
    Dense Layer for objects that transform vector-like.

    Weights are stored as [feature_in, feature_out].
    """

    def __init__(self, n_feature_in, n_feature_out):
        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.randn(n_feature_in, n_feature_out, 4, 4, dtype=torch.cdouble)
        )
        self.weights.data = 0.001 * torch.randn_like(
            self.weights.data, dtype=torch.cdouble
        )
        self.weights.data[0, 0] += torch.eye(4)

        self.n_feature_in = n_feature_in
        self.n_feature_out = n_feature_out

    def forward(self, features_in):
        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(
                f"shape mismatch: got {features_in.shape[0]} but expected {self.n_feature_in}"
            )

        features_out = [
            torch.zeros_like(features_in[0]) for _ in range(self.n_feature_out)
        ]

        for fi, wfi in zip(features_in, self.weights):
            for io, wfo in enumerate(wfi):
                features_out[io] += v_spin_const_transform(wfo, fi)

        return torch.stack(features_out)


class Copy(torch.nn.Module):
    """
    Copy Layer for objects that transform vector-like.

    It has no weights.
    """

    def __init__(self, n_feature_out):
        super().__init__()
        self.n_feature_in = 1
        self.n_feature_out = n_feature_out

    def forward(self, features_in):
        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(
                f"shape mismatch: got {features_in.shape[0]} but expected {self.n_feature_in}"
            )

        features_out = [features_in[0] for _ in range(self.n_feature_out)]

        return torch.stack(features_out)


class Add(torch.nn.Module):
    """
    Add Layer for objects that transform vector-like.

    It has no weights.
    """

    def __init__(self, n_feature_in):
        super().__init__()
        self.n_feature_in = n_feature_in
        self.n_feature_out = 1

    def forward(self, features_in):
        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(
                f"shape mismatch: got {features_in.shape[0]} but expected {self.n_feature_in}"
            )

        feature_out = torch.zeros_like(features_in[0])

        for fi in features_in:
            feature_out += fi

        return torch.stack([feature_out])
