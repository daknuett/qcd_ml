"""
qcd_ml.nn.dense
===============

This module provides dense linear layers. Currently ``v_Dense``
for vector-like objects is provided.
"""

import torch

from ..base.operations import v_spin_const_transform


class v_Dense(torch.nn.Module):
    r"""
    Dense Layer for vectors.
    
    ``v_Dense.forward(features_in)`` computes
    
    .. math::

        \phi_o(x) = \sum\limits_i W_{io} \phi_i(x)

    where :math:`W_{io}` are spin matrices.


    ``v_Dense.reverse(features_in)`` computes the
    hermitian adjoint operation, i.e.,

    ..math::

        \phi_i(x) = \sum\limits_o} W_{io}^\dagger \phi_o(x).
    """

    def __init__(self, n_feature_in: int, n_feature_out: int) -> None:
        """Initialize the dense layer.

        Args:
            n_feature_in: Number of input features.
            n_feature_out: Number of output features.
        """
        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.randn(n_feature_in, n_feature_out, 4, 4, dtype=torch.cdouble)
        )

        self.n_feature_in = n_feature_in
        self.n_feature_out = n_feature_out

    def forward(self, features_in: torch.Tensor) -> torch.Tensor:
        r"""
        Apply the dense layer to input features.
        
        .. math::

            \phi_o(x) = \sum\limits_i W_{io} \phi_i(x)
        
        Args:
            features_in: Input features tensor of shape (n_feature_in, ...).
            
        Returns:
            Output features tensor of shape (n_feature_out, ...).
        """
        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(
                f"shape mismatch: got {features_in.shape[0]} but expected {self.n_feature_in}"
            )

        return torch.einsum("iojk,iabcdkG->oabcdjG", self.weights, features_in)

    def reverse(self, features_in: torch.Tensor) -> torch.Tensor:
        r"""
        Hermitian adjoint operation of ``forward``.

        .. math::

            \phi_i(x) = \sum\limits_o W_{io}^\dagger \phi_o(x).
        
        Args:
            features_in: Input features tensor of shape (n_feature_out, ...).
            
        Returns:
            Output features tensor of shape (n_feature_in, ...).
        """
        if features_in.shape[0] != self.n_feature_out:
            raise ValueError(
                f"shape mismatch: got {features_in.shape[0]} but expected {self.n_feature_out}"
            )

        return torch.einsum("iojk,oabcdkG->iabcdjG", self.weights.adjoint(), features_in)
